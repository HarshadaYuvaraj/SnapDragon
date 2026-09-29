"""Headless matplotlib visualisation (dark theme) + kinematic grasp animation."""
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from .config import GripperSpec

BG, FG, ACC, RED = "#0d1117", "#e6edf3", "#58a6ff", "#ff3b3b"
PALETTE = ["#ffb000", "#3fb950", "#a371f7", "#39c5cf"]


def _box_edges(x0, x1, y0, y1, z0, z1):
    c = np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)])
    idx = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
    return [(c[a], c[b]) for a, b in idx]


def gripper_segments(center, R, width, spec=None, back=0.0):
    """Wire-frame of a parallel-jaw gripper in the frame of `center`/`R` (approach = R[:,2])."""
    spec = spec or GripperSpec()
    hw, ft, fw = width / 2, spec.finger_thickness, spec.finger_width / 2
    loc = []
    loc += _box_edges(-hw - ft, -hw, -fw, fw, -0.06, spec.finger_below)
    loc += _box_edges(hw, hw + ft, -fw, fw, -0.06, spec.finger_below)
    loc += _box_edges(-hw - ft, hw + ft, -fw, fw, -0.075, -0.06)
    loc += _box_edges(-0.008, 0.008, -0.008, 0.008, -0.16, -0.075)
    o = np.asarray(center, float) - R[:, 2] * back
    return [(o + R @ a, o + R @ b) for a, b in loc]


def _style3d(ax):
    ax.set_facecolor(BG)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.set_pane_color((0.07, 0.09, 0.12, 1)); a.label.set_color(FG)
    ax.tick_params(colors="#8b949e", labelsize=7)
    ax.grid(True)


def _to_base(pts, T):
    return pts @ T[:3, :3].T + T[:3, 3]


def draw_cloud(ax, res, img_bgr, T, best_only=False, sub=2, lift=None, exclude=None):
    P = res.cloud[::sub, ::sub].reshape(-1, 3)
    C = img_bgr[::sub, ::sub][..., ::-1].reshape(-1, 3) / 255.0
    ok = np.isfinite(P).all(1)
    Pb = _to_base(P[ok], T)
    Cb = C[ok]
    if lift is not None:
        cid, dz = lift
        m = np.zeros(res.valid.shape, bool)
        m[res.clusters[cid].mask] = True
        mm = m[::sub, ::sub].reshape(-1)[ok]
        Pb = Pb.copy(); Pb[mm, 2] += dz
    obj = Pb[:, 2] > 0.012
    ax.scatter(*Pb[~obj].T, c=Cb[~obj] * 0.55, s=1.2, depthshade=False, linewidths=0, alpha=0.6)
    ax.scatter(*Pb[obj].T, c=Cb[obj], s=15, depthshade=False, linewidths=0)
    ax.set_xlim(0.34, 0.66); ax.set_ylim(-0.22, 0.22); ax.set_zlim(-0.005, 0.19)
    ax.set_box_aspect((0.32, 0.44, 0.24))
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
    _style3d(ax)


def draw_grasp3d(ax, g, T, spec, color, lw=1.4, back=0.0, contact=True, gripper=True):
    gb = g.transformed(T)
    if gripper:
        segs = gripper_segments(gb.center, gb.R, gb.width, spec, back)
        ax.add_collection3d(Line3DCollection(segs, colors=color, linewidths=lw))
    if contact and gb.contacts is not None:
        ax.scatter(*gb.contacts.T, c=RED, s=46, depthshade=False, edgecolors="white", linewidths=0.8)
        ax.plot(*gb.contacts.T, c=RED, lw=1.5)
    a = gb.center; b = gb.center + gb.R[:, 2] * 0.06
    ax.plot(*np.stack([a - gb.R[:, 2] * 0.0, b]).T, c=RED, lw=0)


def _project(P, rig):
    return np.stack([rig.fx * P[:, 0] / P[:, 2] + rig.cx, rig.fy * P[:, 1] / P[:, 2] + rig.cy], 1)


def overview_figure(res, left_bgr, right_bgr, rig, backend_name, path, spec=None, title_extra=""):
    spec = spec or GripperSpec()
    T = rig.T_base_cam
    fig = plt.figure(figsize=(15.5, 8.6), facecolor=BG)
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 1.25], hspace=0.12, wspace=0.06,
                  left=0.02, right=0.985, top=0.9, bottom=0.03)
    fig.suptitle("StereoEdge-Grasp  |  stereo pair → disparity → point cloud → 6-DoF grasp",
                 color=FG, fontsize=15, fontweight="bold", x=0.02, ha="left")
    fig.text(0.02, 0.925, f"backend: {backend_name}   {title_extra}", color="#8b949e", fontsize=9.5)

    # 1) left image + grasp overlay
    ax = fig.add_subplot(gs[0, 0]); ax.imshow(left_bgr[..., ::-1]); ax.axis("off")
    for k, g in enumerate(res.grasps):
        best = k == 0
        col = RED if best else PALETTE[k % 4]
        p = _project(g.contacts, rig)
        ax.plot(*p.T, "-", c=col, lw=2.2 if best else 1.4)
        ax.scatter(*p.T, c=col, s=55 if best else 26, edgecolors="white", linewidths=0.8, zorder=5)
        if best:
            ax.annotate("BEST", p.mean(0), xytext=(10, -22), textcoords="offset points", color="white",
                        fontsize=9, fontweight="bold", bbox=dict(boxstyle="round,pad=0.25", fc=RED, ec="none"))
    ax.set_title("Left camera + grasp contacts", color=FG, fontsize=10.5)

    # 2) disparity
    ax = fig.add_subplot(gs[0, 1])
    lo, hi = np.nanpercentile(res.disp, [2, 98])
    ax.imshow(np.nan_to_num(res.disp, nan=lo), cmap="turbo", vmin=lo, vmax=hi); ax.axis("off")
    ax.set_title("Disparity (px)   invalid = dark", color=FG, fontsize=10.5)

    # 3) segmentation
    ax = fig.add_subplot(gs[0, 2])
    seg = np.zeros(left_bgr.shape, np.uint8); seg[...] = (left_bgr * 0.35).astype(np.uint8)
    for k, c in enumerate(res.clusters):
        rgb = np.array(matplotlib.colors.to_rgb(PALETTE[k % 4])) * 255
        seg[c.mask] = (0.35 * left_bgr[c.mask] + 0.65 * rgb[::-1]).astype(np.uint8)
    ax.imshow(seg[..., ::-1]); ax.axis("off")
    ax.set_title(f"Table plane removed → {len(res.clusters)} objects", color=FG, fontsize=10.5)

    # 4) 3D
    ax = fig.add_subplot(gs[1, :2], projection="3d")
    draw_cloud(ax, res, left_bgr, T)
    for k, g in enumerate(res.grasps):
        draw_grasp3d(ax, g, T, spec, RED if k == 0 else PALETTE[k % 4], lw=1.6,
                     contact=True, gripper=(k == 0))
    ax.view_init(elev=42, azim=-58)
    ax.set_title("3-D point cloud (robot-base frame) with gripper poses", color=FG, fontsize=10.5)

    # 5) text
    ax = fig.add_subplot(gs[1, 2]); ax.axis("off"); ax.set_facecolor(BG)
    tm = res.timings_ms
    lines = ["LATENCY (this machine)"]
    for k, v in tm.items():
        lines.append(f"  {k:<16s}{v:7.1f} ms")
    lines[-1] = f"  {'TOTAL':<16s}{tm['total']:7.1f} ms  ({1000 / tm['total']:.1f} FPS)"
    if res.best:
        gb = res.best.transformed(T)
        lines += ["", "BEST GRASP  (robot base frame)",
                  f"  position  {np.round(gb.center, 3).tolist()} m",
                  f"  approach  {np.round(gb.R[:, 2], 2).tolist()}",
                  f"  closing   {np.round(gb.R[:, 0], 2).tolist()}",
                  f"  width     {gb.width * 1000:.1f} mm",
                  f"  score     {gb.score:.3f}"]
        lines += [f"    {k:<13s}{v:.2f}" for k, v in gb.parts.items()]
    ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", family="monospace", color=FG,
            fontsize=9.6, linespacing=1.45)
    fig.savefig(path, dpi=130, facecolor=BG)
    plt.close(fig)


def grasp_closeup(res, left_bgr, rig, path, spec=None, elev=42, azim=-58):
    spec = spec or GripperSpec()
    fig = plt.figure(figsize=(9, 6.2), facecolor=BG)
    ax = fig.add_subplot(111, projection="3d")
    draw_cloud(ax, res, left_bgr, rig.T_base_cam, sub=1)
    for k, g in enumerate(res.grasps):
        draw_grasp3d(ax, g, rig.T_base_cam, spec, RED if k == 0 else PALETTE[k % 4],
                     lw=1.8, gripper=(k == 0))
    ax.view_init(elev=elev, azim=azim)
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(path, dpi=150, facecolor=BG)
    plt.close(fig)


def stereo_pair_image(left, right, path):
    sep = np.full((left.shape[0], 6, 3), 255, np.uint8)
    cv2.imwrite(path, np.hstack([left, sep, right]))


def grasp_animation(res, left_bgr, rig, path, spec=None, frames_per_phase=6):
    """Kinematic approach → close → lift sequence for the best grasp (GIF, no physics engine)."""
    from PIL import Image
    import io
    spec = spec or GripperSpec()
    g = res.best
    T = rig.T_base_cam
    seq = []
    for t in np.linspace(0.14, 0.0, frames_per_phase):
        seq.append((t, g.width + 0.02, 0.0))                         # descend, open
    for w in np.linspace(g.width + 0.02, g.width, 3):
        seq.append((0.0, w, 0.0))                                    # close
    for dz in np.linspace(0.0, 0.10, frames_per_phase):
        seq.append((-dz, g.width, dz))                               # lift (back<0 => up)
    imgs = []
    for back, w, lift in seq:
        fig = plt.figure(figsize=(6.4, 4.8), facecolor=BG)
        ax = fig.add_subplot(111, projection="3d")
        draw_cloud(ax, res, left_bgr, T, sub=2, lift=(g.object_id, lift) if lift else None)
        gg = type(g)(g.center, g.R, w, g.score, g.object_id, g.parts, g.contacts)
        gb = gg.transformed(T)
        segs = gripper_segments(gb.center, gb.R, w, spec, back=back)
        ax.add_collection3d(Line3DCollection(segs, colors=RED, linewidths=1.6))
        ax.view_init(elev=30, azim=-58); fig.subplots_adjust(0, 0, 1, 1)
        buf = io.BytesIO(); fig.savefig(buf, dpi=80, facecolor=BG, format="png"); plt.close(fig)
        imgs.append(Image.open(io.BytesIO(buf.getvalue())).convert("P", palette=Image.ADAPTIVE))
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=110, loop=0)
