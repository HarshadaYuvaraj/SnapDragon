"""Antipodal parallel-jaw grasp synthesis on segmented object clusters.

For a camera looking down on a table, side walls are seen at grazing incidence, so per-point
PCA normals on walls are noisy. We therefore build the antipodal test on the object's
footprint contour, expressed in metric table-plane coordinates (accurate 2-D normals), and use
the 3-D PCA normals as a secondary consistency term. Each candidate is then checked against a
finger-volume collision model using the full scene cloud and the table plane.
"""
from dataclasses import dataclass, field
import numpy as np
import cv2
from scipy.spatial import cKDTree
from ..config import GripperSpec


@dataclass
class Grasp:
    center: np.ndarray            # (3,) camera frame
    R: np.ndarray                 # (3,3) columns: closing axis, y, approach
    width: float
    score: float
    object_id: int
    parts: dict = field(default_factory=dict)
    contacts: np.ndarray = None   # (2,3) camera frame

    @property
    def approach(self):
        return self.R[:, 2]

    def transformed(self, T):
        g = Grasp(T[:3, :3] @ self.center + T[:3, 3], T[:3, :3] @ self.R, self.width,
                  self.score, self.object_id, self.parts,
                  None if self.contacts is None else self.contacts @ T[:3, :3].T + T[:3, 3])
        return g

    def to_dict(self, T=None):
        g = self if T is None else self.transformed(T)
        return dict(position_m=[round(float(x), 4) for x in g.center],
                    approach_vector=[round(float(x), 4) for x in g.R[:, 2]],
                    closing_axis=[round(float(x), 4) for x in g.R[:, 0]],
                    rotation_matrix=[[round(float(x), 5) for x in r] for r in g.R],
                    gripper_width_m=round(float(g.width), 4), score=round(float(g.score), 4),
                    object_id=int(g.object_id),
                    score_terms={k: round(float(v), 3) for k, v in g.parts.items()})


def _circ_smooth(x, w):
    k = np.ones(w) / w
    pad = w // 2
    xp = np.concatenate([x[-pad:], x, x[:pad]])
    return np.stack([np.convolve(xp[:, i], k, "valid") for i in range(x.shape[1])], 1)


def _finger_collision(pts, c, a, b, up, w, spec, extra=0.0):
    """Count scene points inside the two finger volumes (grasp frame: a closing, b lateral, up)."""
    d = pts - c
    x, y, z = d @ a, d @ b, d @ up
    hw = w / 2
    inx = (np.abs(x) > hw + 0.006) & (np.abs(x) < hw + spec.finger_thickness + extra)
    iny = np.abs(y) < spec.finger_width / 2 + extra
    inz = (z > -spec.finger_below - extra) & (z < spec.finger_above)
    return int(np.sum(inx & iny & inz))


def synthesize_grasps(cluster, all_obj_points, plane, spec: GripperSpec = None,
                      max_contour_pts=140, top_k=1):
    spec = spec or GripperSpec()
    n_up, d_pl = plane
    n_up = n_up.astype(np.float64)
    m8 = cluster.mask.astype(np.uint8)
    cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return []
    cnt = max(cnts, key=len)[:, 0, :]
    # 3-D contour points from the (organised) cluster points
    H, W = cluster.mask.shape
    grid = np.full((H, W, 3), np.nan, np.float32)
    grid[cluster.mask] = cluster.points
    P3 = grid[cnt[:, 1], cnt[:, 0]].astype(np.float64)
    ok = np.isfinite(P3).all(1)
    P3 = P3[ok]
    if len(P3) < 20:
        return []
    h_pt = P3 @ n_up + d_pl                                  # height of each contour point
    # metric in-plane basis
    ex = np.array([1.0, 0, 0]); e1 = ex - (ex @ n_up) * n_up; e1 /= np.linalg.norm(e1)
    e2 = np.cross(n_up, e1)
    Q = np.stack([P3 @ e1, P3 @ e2], 1)
    sel = np.linspace(0, len(Q), min(len(Q), max_contour_pts), endpoint=False).astype(int)
    Qs = _circ_smooth(Q, 5)
    # contour tangent / outward normal in metric plane
    T = np.roll(Qs, -3, 0) - np.roll(Qs, 3, 0)
    T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-9
    Nrm = np.stack([T[:, 1], -T[:, 0]], 1)
    cen2 = Qs.mean(0)
    if np.mean(np.einsum("ij,ij->i", Nrm, Qs - cen2)) < 0:
        Nrm = -Nrm
    Qs_, N_, P_, h_ = Qs[sel], Nrm[sel], P3[sel], h_pt[sel]

    diff = Qs_[None, :, :] - Qs_[:, None, :]
    dist = np.linalg.norm(diff, axis=2)
    u = diff / np.maximum(dist[..., None], 1e-9)
    cos_i = -np.einsum("ijk,ik->ij", u, N_)                 # contact i: pull direction vs -n_i
    cos_j = np.einsum("ijk,jk->ij", u, N_)                  # contact j
    ca = np.cos(np.radians(spec.friction_angle_deg))
    valid = (cos_i > ca) & (cos_j > ca) & (dist > spec.min_width) \
        & (dist < spec.max_width - 0.006)
    iu = np.triu(np.ones_like(valid, bool), 1)
    ii, jj = np.nonzero(valid & iu)
    if len(ii) == 0:
        return []

    h_top = cluster.top_height
    h_g = max(0.5 * h_top, min(0.02, 0.8 * h_top))          # grasp height above table
    obj_area = cv2.contourArea(Qs.astype(np.float32).reshape(-1, 1, 2))
    r_obj = np.sqrt(max(obj_area, 1e-6) / np.pi)
    cen3 = cluster.centroid
    cen_pl = np.array([cen3 @ e1, cen3 @ e2])

    scene = all_obj_points
    # cheap vectorised pre-score for every valid pair, then full checks on the best few
    ii, jj = np.nonzero(valid & iu)
    w_all = dist[ii, jj]
    antip_all = 0.5 * ((cos_i[ii, jj] - ca) + (cos_j[ii, jj] - ca)) / (1 - ca)
    mid = 0.5 * (Qs_[ii] + Qs_[jj])
    cen_all = np.clip(1 - np.linalg.norm(mid - cen_pl, axis=1) / (0.6 * r_obj), 0, 1)
    wfit_all = np.clip(1 - np.abs(w_all - 0.5 * (spec.min_width + spec.max_width))
                       / (0.5 * spec.max_width), 0, 1)
    cheap = 0.30 * antip_all + 0.25 * cen_all + 0.15 * wfit_all
    order = np.argsort(-cheap)[:60]
    ntree = cKDTree(cluster.points) if cluster.normals is not None else None

    grasps = []
    if h_g - spec.finger_below < spec.table_margin:          # fingertips would hit the table
        return []
    for o in order:
        i, j = ii[o], jj[o]
        w = float(w_all[o])
        pi = P_[i] - n_up * (h_[i] - h_g)
        pj = P_[j] - n_up * (h_[j] - h_g)
        c = 0.5 * (pi + pj)
        a = pj - pi; a -= (a @ n_up) * n_up; a /= np.linalg.norm(a)
        b = np.cross(n_up, a)
        if _finger_collision(scene, c, a, b, n_up, w, spec) > 6:
            continue
        clear = float(np.clip(1 - _finger_collision(scene, c, a, b, n_up, w, spec, 0.015) / 60.0, 0, 1))
        nc = 0.5
        if ntree is not None:                                # 3-D normal agreement at contacts
            hit = []
            for pc in (P_[i], P_[j]):
                nb = ntree.query(pc, k=12)[1]
                nn_ = cluster.normals[nb]
                nh = nn_ - np.outer(nn_ @ n_up, n_up)
                mag = np.linalg.norm(nh, axis=1)
                m = mag > 0.3
                if m.any():
                    hit.append(np.mean(np.abs((nh[m] / mag[m, None]) @ a)))
            if hit:
                nc = float(np.mean(hit))
        parts = dict(antipodal=float(antip_all[o]), centering=float(cen_all[o]),
                     width_fit=float(wfit_all[o]), clearance=clear, normal_agree=nc)
        score = cheap[o] + 0.15 * clear + 0.15 * nc
        R = np.stack([a, np.cross(-n_up, a), -n_up], 1)
        grasps.append(Grasp(c.astype(np.float32), R.astype(np.float32), w, float(score),
                            cluster.id, parts, np.stack([pi, pj]).astype(np.float32)))
        if len(grasps) >= 8:                                 # enough verified candidates
            break
    grasps.sort(key=lambda g: -g.score)
    return grasps[:top_k]
