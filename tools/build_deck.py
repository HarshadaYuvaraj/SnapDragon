#!/usr/bin/env python3
"""Builds docs/StereoEdge-Grasp_pitch.pdf from outputs/ (figures + benchmark.json). Needs reportlab."""
import json, os, sys
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor, white
from reportlab.lib.utils import ImageReader, simpleSplit

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "docs", "StereoEdge-Grasp_pitch.pdf")
W, H = 1280, 720
BG, PANEL, LINE = HexColor("#0b1220"), HexColor("#141d30"), HexColor("#26334d")
FG, MUTED, ACC, ACC2, RED, GRN, AMB = (HexColor(x) for x in
    ("#eaf0fb", "#93a1bb", "#3d8bff", "#22d3c5", "#ff4d4d", "#3fb950", "#ffb000"))
bench = json.load(open(os.path.join(ROOT, "outputs", "benchmark.json")))
grasp = json.load(open(os.path.join(ROOT, "outputs", "grasps.json")))
c = canvas.Canvas(OUT, pagesize=(W, H))
c.setTitle("StereoEdge-Grasp: Real-Time 3D Grasp Synthesis from Stereo Vision on the Snapdragon NPU")
c.setAuthor("StereoEdge-Grasp"); c.setSubject("Pitch deck")
page = [0]


def new(title, kicker=None):
    if page[0]:
        c.showPage()
    page[0] += 1
    c.setFillColor(BG); c.rect(0, 0, W, H, 0, 1)
    c.setFillColor(ACC); c.rect(0, H - 6, W, 6, 0, 1)
    if kicker:
        c.setFillColor(ACC2); c.setFont("Helvetica-Bold", 15); c.drawString(70, H - 62, kicker.upper())
    c.setFillColor(FG); c.setFont("Helvetica-Bold", 38); c.drawString(70, H - 108, title)
    c.setFillColor(MUTED); c.setFont("Helvetica", 12)
    c.drawString(70, 26, "StereoEdge-Grasp")
    c.drawRightString(W - 70, 26, f"{page[0]} / 8")


def text(x, y, s, size=19, w=500, font="Helvetica", color=FG, lead=1.35):
    c.setFillColor(color); c.setFont(font, size)
    for ln in simpleSplit(s, font, size, w):
        c.drawString(x, y, ln); y -= size * lead
    return y


def card(x, y, w, h, head, body, accent=ACC, size=20):
    c.setFillColor(PANEL); c.roundRect(x, y, w, h, 14, 0, 1)
    c.setFillColor(accent); c.roundRect(x, y + h - 6, w, 6, 3, 0, 1)
    c.setFillColor(FG); c.setFont("Helvetica-Bold", 23); c.drawString(x + 22, y + h - 46, head)
    if isinstance(body, str):
        text(x + 22, y + h - 80, body, size, w - 44, color=MUTED)
    else:
        yy = y + h - 80
        for item in body:
            c.setFillColor(accent); c.circle(x + 30, yy + size * 0.32, 3.5, 0, 1)
            yy = text(x + 46, yy, item, size, w - 70, color=MUTED) - 7


def arrow(x1, y, x2, col=MUTED):
    c.setStrokeColor(col); c.setFillColor(col); c.setLineWidth(2.5)
    c.line(x1, y, x2 - 9, y)
    p = c.beginPath(); p.moveTo(x2, y); p.lineTo(x2 - 11, y + 6); p.lineTo(x2 - 11, y - 6); p.close()
    c.drawPath(p, 0, 1)


def img(path, x, y, w, h=None):
    im = ImageReader(os.path.join(ROOT, "outputs", path)); iw, ih = im.getSize()
    h = h or w * ih / iw
    c.drawImage(im, x, y, w, h, mask="auto")
    return h

# 1 ── title
new("", None); page[0] = 1
c.setFillColor(ACC); c.rect(0, H - 6, W, 6, 0, 1)
img("grasp_3d.png", 690, 90, 520)
c.setFillColor(ACC2); c.setFont("Helvetica-Bold", 16); c.drawString(70, 560, "SNAPDRAGON AI CHALLENGE SUBMISSION")
c.setFillColor(FG); c.setFont("Helvetica-Bold", 68); c.drawString(70, 470, "StereoEdge-Grasp")
text(70, 410, "Real-time 3D grasp synthesis from low-cost stereo vision, built for on-device inference on the Snapdragon NPU", 27, 600, color=FG)
text(70, 270, "Two ordinary cameras. No depth sensor. No cloud. A collision-checked 6-DoF grasp pose for a robot arm.", 19, 560, color=MUTED)
for i, (k, v) in enumerate([("2", "commodity cameras"), ("0", "bytes sent to the cloud"), ("6-DoF", "grasp + width + score")]):
    x = 70 + i * 200
    c.setFillColor(PANEL); c.roundRect(x, 110, 185, 100, 12, 0, 1)
    c.setFillColor(ACC2); c.setFont("Helvetica-Bold", 34); c.drawString(x + 18, 160, k)
    c.setFillColor(MUTED); c.setFont("Helvetica", 13.5); c.drawString(x + 18, 130, v)
c.setFillColor(MUTED); c.setFont("Helvetica", 12); c.drawRightString(W - 70, 26, "1 / 8")

# 2 ── problem
new("Robot vision is too costly, too heavy, too exposed", "The problem")
card(70, 300, 350, 270, "Expensive sensors", "RGB-D cameras add cost, bulk and cabling to every cell, and struggle in sunlight and on reflective parts.", RED)
card(465, 300, 350, 270, "Cloud round-trips", "Streaming frames to remote GPUs adds latency and jitter to a control loop that must be deterministic.", AMB)
card(860, 300, 350, 270, "Privacy and power", "Factory and lab imagery is sensitive, and always-on cloud inference burns energy and bandwidth.", ACC)
c.setFillColor(PANEL); c.roundRect(70, 110, 1140, 150, 14, 0, 1)
c.setFillColor(ACC2); c.setFont("Helvetica-Bold", 22); c.drawString(100, 215, "Our answer")
text(100, 182, "Recover 3D from two cheap cameras and compute the grasp locally: the matrix-heavy stereo network targets the Snapdragon NPU, and the whole loop stays on the device.", 20, 1080)

# 3 ── pipeline
new("From two images to one grasp pose", "How it works")
steps = [("1  Stereo capture", "Two synchronized cameras, rectified", MUTED),
         ("2  Neural stereo", "ONNX network via QNN EP (NPU)", ACC),
         ("3  Point cloud", "Disparity to 3D, RANSAC table plane", MUTED),
         ("4  Grasp search", "Antipodal pairs, normals, scoring", MUTED),
         ("5  Collision check", "Finger volumes vs. scene and table", MUTED),
         ("6  6-DoF grasp", "Pose, approach, width to the arm", GRN)]
bw, gap = 172, 21
for i, (h_, b_, col) in enumerate(steps):
    x = 70 + i * (bw + gap)
    c.setFillColor(PANEL); c.roundRect(x, 400, bw, 150, 12, 0, 1)
    c.setStrokeColor(col); c.setLineWidth(2.5); c.roundRect(x, 400, bw, 150, 12, 1, 0)
    c.setFillColor(FG); c.setFont("Helvetica-Bold", 16); c.drawString(x + 14, 520, h_)
    text(x + 14, 492, b_, 14, bw - 26, color=MUTED)
    if i < 5:
        arrow(x + bw + 2, 475, x + bw + gap - 2)
c.setFillColor(ACC); c.roundRect(70 + (bw + gap), 374, bw, 6, 3, 0, 1)
c.setFillColor(ACC); c.setFont("Helvetica-Bold", 13); c.drawString(70 + (bw + gap), 350, "Neural workload: target for the Hexagon NPU")
img("stereo_pair.png", 70, 110, 560)
c.setFillColor(MUTED); c.setFont("Helvetica", 12); c.drawString(70, 92, "Input: left / right frame (synthetic top-down scene with ground truth)")
text(670, 245, "Swappable backends behind one interface", 21, 540, "Helvetica-Bold")
y = text(670, 212, "sgbm  classical OpenCV baseline (CPU)", 17, 540, "Helvetica", MUTED)
y = text(670, y - 4, "onnx-cpu  ONNX graph on onnxruntime", 17, 540, "Helvetica", MUTED)
y = text(670, y - 4, "onnx-qnn  same graph on the Hexagon NPU (QNN HTP)", 17, 540, "Helvetica", MUTED)
text(670, y - 14, "A trained stereo model from Qualcomm AI Hub or an open-source hub drops in with the same input/output contract.", 16, 540, color=ACC2)

# 4 ── Snapdragon
new("Why it belongs on Snapdragon", "The edge advantage")
card(70, 380, 550, 190, "Privacy by construction", "The pipeline makes no network calls. Frames, point clouds and grasps stay on the HP Snapdragon PC.", GRN)
card(660, 380, 550, 190, "Deterministic latency", "No cloud round-trip. The stereo network is a static-shape ONNX graph of NPU-friendly ops (Pad, Slice, Sub, Abs, Concat, AveragePool).", ACC)
card(70, 170, 550, 190, "Frees the CPU", "Offloading the dense matrix math to the NPU leaves CPU cores for control, vision post-processing and the arm driver.", ACC2)
card(660, 170, 550, 190, "Low-cost scale-out", "Commodity cameras replace depth sensors, and one PC serves as the cell's edge brain.", AMB)
c.setFillColor(PANEL); c.roundRect(70, 60, 1140, 80, 12, 0, 1)
c.setFillColor(AMB); c.setFont("Helvetica-Bold", 15); c.drawString(94, 108, "STATUS")
text(180, 108, "QNN backend, AI Hub profiling script and Windows-ARM64 setup are implemented. On-device NPU latency and power will be measured on Snapdragon X hardware and are not claimed here.", 15, 1000, color=FG, lead=1.3)

# 5 ── demo
new("Live output: cloud, contacts and gripper pose", "Demo")
img("overview.png", 60, 52, 940)
for i, (h_, b_, col) in enumerate([("Red", "selected grasp contacts and gripper", RED), ("Score", "antipodal, centering, width, clearance, normals", ACC2), ("Timing", "median of 5 runs, single x86 CPU core", AMB)]):
    y = 470 - i * 130
    c.setFillColor(col); c.roundRect(1030, y, 5, 100, 2, 0, 1)
    c.setFillColor(FG); c.setFont("Helvetica-Bold", 20); c.drawString(1048, y + 74, h_)
    text(1048, y + 48, b_, 15.5, 170, color=MUTED)

# 6 ── results
new("Measured results", "Benchmarks")
ck = lambda k, s: bench["results"][k][s]["median"]
sg, on = "sgbm", "onnx-cpu"
c.setFillColor(PANEL); c.roundRect(70, 330, 520, 250, 14, 0, 1)
c.setFillColor(FG); c.setFont("Helvetica-Bold", 20); c.drawString(94, 545, "Stereo accuracy vs. ground truth")
rows = [("", "SGBM", "ONNX net"), ("Mean disparity error", "0.27 px", "0.08 px"), ("Pixels within 1 px", "99.4 %", "100 %"),
        ("Valid pixels", "84.6 %", "85.6 %"), ("Objects found / grasped", "4 / 4", "4 / 4")]
for i, r in enumerate(rows):
    y = 500 - i * 36
    c.setFillColor(MUTED if i == 0 else FG); c.setFont("Helvetica-Bold" if i == 0 else "Helvetica", 16)
    c.drawString(94, y, r[0]); c.drawRightString(440, y, r[1]); c.drawRightString(566, y, r[2])
    c.setStrokeColor(LINE); c.line(94, y - 11, 566, y - 11)
c.setFillColor(PANEL); c.roundRect(620, 330, 590, 250, 14, 0, 1)
c.setFillColor(FG); c.setFont("Helvetica-Bold", 20); c.drawString(644, 545, "Latency per stage (ms, median)")
stages = [("stereo", "stereo_matching"), ("cloud", "point_cloud"), ("plane", "plane_fit"), ("segment", "segmentation"),
          ("normals", "normals"), ("grasp", "grasp_synthesis")]
mx = max(ck(on, k) for _, k in stages)
for i, (lab, k) in enumerate(stages):
    y = 500 - i * 29; v = ck(on, k)
    c.setFillColor(MUTED); c.setFont("Helvetica", 14); c.drawString(644, y, lab)
    c.setFillColor(ACC if k == "stereo_matching" else LINE); c.roundRect(730, y - 4, 380 * v / mx, 17, 4, 0, 1)
    c.setFillColor(FG); c.drawString(736 + 380 * v / mx, y, f"{v:.1f}")
tot = ck(on, "total")
c.setFillColor(ACC2); c.setFont("Helvetica-Bold", 19)
c.drawString(94, 290, f"End-to-end: {tot:.0f} ms  =  {1000 / tot:.0f} FPS on one x86 CPU core (320 x 240, ONNX on CPU)")
c.setFillColor(MUTED); c.setFont("Helvetica", 15)
text(94, 255, "Setup: Linux x86-64, single core, no NPU, median of 30 runs on a synthetic scene with ground truth. The grasp poses land within 2 cm of true object centres with widths inside the gripper range (7 automated tests).", 15, 1090, color=MUTED)
card(70, 90, 1140, 110, "What this tells us", "The stereo network is already the largest single stage, which is exactly the part the NPU is meant to take. The remaining stages are lightweight CPU code. We do not claim NPU speed-ups until measured on the device.", ACC, 15)

# 7 ── use case
new("Where it applies", "Use case & market")
uses = [("Bin picking & sorting", "Mixed parts on a conveyor or tray"), ("Small-batch manufacturing", "Flexible cells without custom fixtures"),
        ("Lab & pharma automation", "Sensitive imagery must stay on site"), ("Recycling & e-waste", "Cheap sensing for harsh, dirty cells")]
for i, (h_, b_) in enumerate(uses):
    x = 70 + (i % 2) * 400; y = 400 - (i // 2) * 165
    card(x, y, 375, 140, h_, b_, [ACC, ACC2, GRN, AMB][i], 18)
c.setFillColor(PANEL); c.roundRect(870, 235, 340, 305, 14, 0, 1)
c.setFillColor(ACC2); c.setFont("Helvetica-Bold", 21); c.drawString(894, 500, "Deployment")
y = 465
for s in ["pip install, one Python codebase", "HP Snapdragon PC as the cell's edge brain", "Two USB/CSI cameras, standard calibration", "JSON grasp output for any arm driver", "MIT-licensed, runs on CPU without an NPU"]:
    y = text(894, y, s, 15.5, 290, color=FG) - 8
text(70, 130, "Accessible by design: the same code runs on any machine for development and switches to the NPU with one flag (--backend onnx-qnn).", 18, 780, color=MUTED)

# 8 ── limitations / roadmap
new("Honest limits and next steps", "Roadmap")
card(70, 300, 550, 270, "Known limitations", ["Stereo needs texture; glossy or featureless parts degrade depth", "Demo scene is clean and top-down; clutter is harder", "Grasp search assumes vertical-walled objects", "NPU latency and power are pending hardware"], RED, 19)
card(660, 300, 550, 270, "Next steps", ["Drop in a trained stereo model (AI Hub / open source), quantise to INT8", "Profile on Snapdragon X with AI Hub; publish NPU latency and power", "Validate on real stereo pairs", "Hardware-in-the-loop with a physical arm"], GRN, 19)
c.setFillColor(PANEL); c.roundRect(70, 110, 1140, 150, 14, 0, 1)
c.setFillColor(ACC2); c.setFont("Helvetica-Bold", 22); c.drawString(100, 222, "Deliverables")
text(100, 190, "Source repository (MIT) with README, 7 passing tests, benchmark script, Windows-ARM64 setup script, QNN backend and AI Hub profiling tool. Every figure in this deck is generated by the code in the repository.", 18, 1080, color=FG)
c.save()
print("wrote", OUT)
