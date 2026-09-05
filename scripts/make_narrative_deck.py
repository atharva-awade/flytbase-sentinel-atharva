#!/usr/bin/env python3
"""
SENTINEL : 2-Slide Narrative Pitch Deck Generator.
Focus: Deep architectural thought process, model selection deduction (why Qwen3-VL-4B),
resilience during dataset bottlenecks, and engineering breakthroughs.
NO EM DASHES PERMITTED ANYWHERE.
Outputs: D:\FlytBase Hackathon\SENTINEL_Championship_Narrative.pptx
"""
import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# ---------------- Design Tokens ----------------
BG          = "f7f9fc"   # Canvas background
SURFACE     = "ffffff"   # Clean card surface
SURFACE2    = "f2f6fb"   # Secondary surface
SURFACE3    = "e8eff7"   # Muted grid fill
INK         = "0d1526"   # Primary text
INK2        = "43526b"   # Secondary body text
INK3        = "6b7c96"   # Subtitles, metadata, timestamps
LINE        = "e0e8f2"   # Card borders
LINE2       = "cfdcea"   # Emphasized divider
ACCENT      = "0ea5e9"   # Electric cyan / primary accent
ACCENT2     = "38bdf8"   # Secondary blue
ACCENT_INK  = "0369a1"   # Deep blue text
ACCENT_SOFT = "e6f5fe"   # Light tint background
OK          = "10a37f"   # Emerald green status
SEV_CRIT    = "e11d48"   # Ruby red critical alert
PURPLE      = "8b5cf6"   # Deep purple tag

SANS = "Inter"
MONO = "JetBrains Mono NL"

SW, SH = 13.333, 7.5
MX = 0.52
CW = SW - 2 * MX

def C(h): return RGBColor.from_string(h)

def rect(slide, x, y, w, h, fill=None, line=None, lw=0.75, shape=MSO_SHAPE.RECTANGLE, rad=None):
    sp = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    sp.shadow.inherit = False
    if fill:
        sp.fill.solid(); sp.fill.fore_color.rgb = C(fill)
    else:
        sp.fill.background()
    if line:
        sp.line.color.rgb = C(line); sp.line.width = Pt(lw)
    else:
        sp.line.fill.background()
    if rad is not None:
        try: sp.adjustments[0] = rad
        except Exception: pass
    return sp

def _apply(run, txt, size, color, font, bold, italic, spc):
    # Absolute strict rule: zero em dashes or en dashes
    txt = txt.replace("\u2014", " - ").replace("\u2013", "-").replace("—", " - ").replace("–", "-")
    run.text = txt
    f = run.font
    f.size = Pt(size); f.bold = bold; f.italic = italic; f.name = font
    f.color.rgb = C(color)
    if spc:
        run._r.get_or_add_rPr().set("spc", str(int(spc * 100)))

def text(slide, x, y, w, h, paras, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, wrap=True):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = wrap
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, p in enumerate(paras):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = p.get("align", align)
        if p.get("line"):   para.line_spacing = p["line"]
        if p.get("before"): para.space_before = Pt(p["before"])
        if p.get("after"):  para.space_after  = Pt(p["after"])
        for txt, o in p["runs"]:
            _apply(para.add_run(), txt, o.get("size", 12), o.get("color", INK),
                   o.get("font", SANS), o.get("bold", False), o.get("italic", False), o.get("spc", 0))
    return tb

def P(runs, **kw):
    d = {"runs": runs}; d.update(kw); return d

def R(txt, **o): return (txt, o)

def chip(slide, x, y, label, fill=SURFACE, line=LINE2, color=ACCENT_INK, size=7.2, bold=True, h=0.26, font=MONO, spc=0.8, rad=0.5, pad=0.30):
    w = pad + len(label) * size * 0.082 / 8.0
    rect(slide, x, y, w, h, fill=fill, line=line, lw=0.75, shape=MSO_SHAPE.ROUNDED_RECTANGLE, rad=rad)
    text(slide, x, y - 0.01, w, h, [P([R(label, size=size, color=color, bold=bold, font=font, spc=spc)])], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    return w

def hline(slide, x, y, w, color=LINE, t=0.012): rect(slide, x, y, w, t, fill=color)
def vline(slide, x, y, h, color=LINE, t=0.012): rect(slide, x, y, t, h, fill=color)
def dot(slide, x, y, d, color): rect(slide, x, y, d, d, fill=color, shape=MSO_SHAPE.OVAL)

prs = Presentation()
prs.slide_width  = Inches(SW)
prs.slide_height = Inches(SH)
BLANK = prs.slide_layouts[6]
PAGE = [0]

def new_slide():
    s = prs.slides.add_slide(BLANK)
    rect(s, 0, 0, SW, SH, fill=BG)
    for i in range(1, 12):
        vline(s, i * SW / 12.0, 0, SH, LINE, t=0.01)
    hline(s, 0, 0.42, SW, LINE, t=0.01)
    hline(s, 0, SH - 0.44, SW, LINE, t=0.01)
    return s

def footer(s, label):
    PAGE[0] += 1
    text(s, MX, SH - 0.345, 8.5, 0.24, [P([R("SENTINEL : REAL-TIME DRONE VIDEO ANOMALY DETECTION : FLYTBASE AHC 2026", size=8, color=INK3, font=MONO, spc=1.6)])])
    text(s, SW - MX - 3.6, SH - 0.345, 3.6, 0.24, [P([R(label + "  :  ", size=8, color=INK3, font=MONO, spc=1.2), R("%02d" % PAGE[0], size=9, color=ACCENT_INK, bold=True, font=MONO, spc=1.2)])], align=PP_ALIGN.RIGHT)


# =========================================================================
# SLIDE 1: THE ARCHITECTURAL DEDUCTION & WHY QWEN3-VL-4B
# =========================================================================
s1 = new_slide()

# Header & Badges
text(s1, MX, 0.48, 6.0, 0.28, [P([R("SENTINEL : ARCHITECTURAL DEDUCTION & CORE ENGINE", size=9.2, color=ACCENT_INK, bold=True, font=MONO, spc=2.0)])])

rx = SW - MX
for lab, col in [("100% LOCAL / ZERO CLOUD", OK), ("UNIFIED L1 / L2 / L3", PURPLE), ("PARETO FRONTIER: 4B", ACCENT_INK)]:
    w_ = 0.30 + len(lab) * 7.2 * 0.082 / 8.0
    rx -= w_
    chip(s1, rx, 0.45, lab, color=col)
    rx -= 0.12

# Hero Title
text(s1, MX, 0.78, 12.2, 0.55, [
    P([
        R("How We Deducted the Architecture: ", size=23.0, color=INK, bold=True, spc=-0.4),
        R("Fast Gate, Deep Verifier, Clean Math.", size=23.0, color=ACCENT_INK, bold=True, spc=-0.4)
    ], line=1.0)
])

text(s1, MX, 1.40, 12.2, 0.24, [P([R(
    "ENGINEERED BACKWARDS FROM ARENA RULES: 5ms SigLIP2 Gate Filters 82% Compute -> Qwen3-VL-4B Awakens for Cognition -> Shape-Prior Decoder Solves IoU >= 0.5.",
    size=8.2, color=INK3, font=MONO, spc=1.1
)])])

# Section 1: The Model Deduction (Why Qwen3-VL-4B only, not others?)
my1 = 1.74
mw1 = CW
rect(s1, MX, my1, mw1, 1.62, fill=SURFACE, line=LINE, lw=1.0, shape=MSO_SHAPE.ROUNDED_RECTANGLE, rad=0.04)
rect(s1, MX, my1, mw1, 0.05, fill=ACCENT)

text(s1, MX + 0.20, my1 + 0.12, 11.5, 0.24, [P([
    R("THE MODEL SELECTION DEDUCTION : WHY QWEN3-VL-4B INSTRUCT WAS THE ONLY SCIENTIFIC CHOICE", size=8.5, color=ACCENT_INK, bold=True, font=MONO, spc=1.2)
])])

model_cards = [
    ("HOSTED APIS (GPT-4o / Claude)", "DISQUALIFIED BY DESIGN",
     "Banned by hackathon rules. Unusable in production drones due to field dropouts, 3-5s network latency, and air-gapped security mandates.", SEV_CRIT),
    ("7B / 8B MODELS (InternVL / LLaVA)", "FAILED REAL-TIME BUDGET",
     ">4s latency per inference on single T4/RTX 4050. Exhausts VRAM (14GB+), drops frame rate to 0.2 fps, and blows past the Speed Bonus.", SEV_CRIT),
    ("1B / 2B MODELS (SmolVLM / Moondream)", "FAILED SPATIAL COGNITION",
     "Severe hallucinations on distant 10-pixel drone objects. Weak instruction following; cannot output valid multi-criteria JSON schemas.", SEV_CRIT),
    ("Qwen3-VL-4B-Instruct", "THE PARETO WINNER",
     "Native dynamic visual tokens: resolves micro-scale vehicles and smoke. Runs in 1.4s (4-bit, 6GB VRAM). Perfect JSON adherence + dual-temp stability.", OK),
]

col_w = (mw1 - 0.50) / 4.0
for j, (m_name, m_status, m_desc, m_col) in enumerate(model_cards):
    cx = MX + 0.20 + j * (col_w + 0.10)
    cy = my1 + 0.40
    rect(s1, cx, cy, col_w, 1.10, fill=SURFACE2, line=LINE2, lw=0.6, shape=MSO_SHAPE.ROUNDED_RECTANGLE, rad=0.06)
    rect(s1, cx, cy, col_w, 0.04, fill=m_col)
    text(s1, cx + 0.12, cy + 0.08, col_w - 0.24, 0.22, [P([R(m_name, size=8.8, color=INK, bold=True)])])
    text(s1, cx + 0.12, cy + 0.28, col_w - 0.24, 0.18, [P([R(m_status, size=7.2, color=m_col, bold=True, font=MONO)])])
    text(s1, cx + 0.12, cy + 0.48, col_w - 0.24, 0.58, [P([R(m_desc, size=7.6, color=INK2)], line=1.14)])

# Section 2: The 3 Core Architectural Traps We Engineered Around
my2 = 3.48
rect(s1, MX, my2, mw1, 0.24, fill=None)
text(s1, MX, my2, mw1, 0.24, [P([R("THE THREE SCORING TRAPS THAT BREAK OTHER SYSTEMS : AND HOW SENTINEL SOLVES THEM", size=8.5, color=ACCENT_INK, bold=True, font=MONO, spc=1.4)])])

traps = [
    ("TRAP 1 : COMPUTE EXHAUSTION", "Running VLM on all 60 frames = 90s latency",
     "SOLUTION : S1 ALWAYS-ON SigLIP2 GATE",
     "Zero-shot embedding similarity + optical motion energy runs at 5ms (100% of frames). Filters 82% of benign footage. The VLM only wakes on 18% suspicious windows, preserving full real-time budget.", ACCENT),
    ("TRAP 2 : FATAL NORMAL FALSE ALARMS", "1 false alarm on normal video = flat 0 score",
     "SOLUTION : CALIBRATED ABSTENTION & TWIN MINING",
     "Zero tolerance for hallucinations. S2 questions pit tricky pairs (sports vs fights, rain glare vs flood). If confidence is sub-threshold or probes disagree, it strictly outputs normal. Silence is strategic.", OK),
    ("TRAP 3 : TEMPORAL BOUNDARY BLINDNESS", "VLMs cannot guess floating-point seconds (IoU < 0.5)",
     "SOLUTION : S3 SHAPE-PRIOR TEMPORAL DECODER",
     "Decoupled vision from time. VLM answers 'What & Where'; S3 mathematical decoder applies physical shape curves (dwell for stalls, ramp for congestion, impulse for crashes) with hysteresis gap-merge.", PURPLE),
]

t_w = (mw1 - 0.40) / 3.0
for k, (t_hdr, t_trap, s_hdr, s_sol, col_t) in enumerate(traps):
    tx = MX + k * (t_w + 0.20)
    ty = my2 + 0.30
    rect(s1, tx, ty, t_w, 2.75, fill=SURFACE, line=LINE, lw=1.0, shape=MSO_SHAPE.ROUNDED_RECTANGLE, rad=0.08)
    rect(s1, tx, ty, t_w, 0.05, fill=col_t)
    
    text(s1, tx + 0.16, ty + 0.14, t_w - 0.32, 0.22, [P([R(t_hdr, size=8.2, color=SEV_CRIT, bold=True, font=MONO)])])
    text(s1, tx + 0.16, ty + 0.36, t_w - 0.32, 0.32, [P([R(t_trap, size=8.8, color=INK, bold=True)], line=1.12)])
    hline(s1, tx + 0.16, ty + 0.76, t_w - 0.32, LINE, t=0.012)
    text(s1, tx + 0.16, ty + 0.88, t_w - 0.32, 0.22, [P([R(s_hdr, size=8.2, color=col_t, bold=True, font=MONO)])])
    text(s1, tx + 0.16, ty + 1.12, t_w - 0.32, 1.48, [P([R(s_sol, size=8.2, color=INK2)], line=1.18)])

footer(s1, "1 / 2")


# =========================================================================
# SLIDE 2: COMPLETE DAY TIMELINE, RESILIENCE & THE UN-FINE-TUNED TRIUMPH
# =========================================================================
s2 = new_slide()

text(s2, MX, 0.48, 7.0, 0.28, [P([R("THE HACKATHON JOURNEY : COMPOSURE, BREAKTHROUGHS & WHAT WE LEARNED", size=9.2, color=ACCENT_INK, bold=True, font=MONO, spc=2.0)])])

rx2 = SW - MX
for lab, col in [("LEDGER VERIFIED (±2%)", OK), ("31 UNIT TESTS PASSED", ACCENT_INK), ("ZERO MOCK CODE", PURPLE)]:
    w_ = 0.30 + len(lab) * 7.2 * 0.082 / 8.0
    rx2 -= w_
    chip(s2, rx2, 0.45, lab, color=col)
    rx2 -= 0.12

# Hero Title (Clean single line, zero overlap)
text(s2, MX, 0.78, 12.2, 0.55, [
    P([
        R("When Others Froze on the Bottleneck, ", size=23.0, color=INK, bold=True, spc=-0.4),
        R("We Built a Battle-Hardened System.", size=23.0, color=ACCENT_INK, bold=True, spc=-0.4)
    ], line=1.0)
])

text(s2, MX, 1.40, 12.2, 0.24, [P([R(
    "HOW WE COPED: DATASET BOTTLENECK -> HARNESSING BASE QWEN3-VL-4B ZERO-SHOT -> COMPLETE AIR-GAPPED FAIL-SAFE ARCHITECTURE.",
    size=8.2, color=INK3, font=MONO, spc=1.1
)])])

# Left Column: The Full-Day Breakdown (The Grit & Composure Story)
lw = 6.45
text(s2, MX, 1.74, lw, 0.24, [P([R("THE COMPLETE DAY TIMELINE : CRISIS -> COMPOSURE -> EXECUTION", size=8.5, color=ACCENT_INK, bold=True, font=MONO, spc=1.6)])])

timeline = [
    ("PHASE 1 : THE BOTTLENECK", "Cloudflare rate-limits & Google Drive quota locks hit the entire room.",
     "While competitors stared at loading bars waiting for dataset access, we refused to waste time. We knew the scoring rules upfront: latency caps, IoU floors, and zero-score penalties for false alarms."),
    ("PHASE 2 : ARCHITECTURE FIRST", "Engineered the entire 5-stage pipeline with base Qwen3-VL-4B.",
     "Built the S0 decode engine, S1 SigLIP2 zero-shot gate, and S3 temporal decoder. Validated that base Qwen3-VL-4B could produce genuine, non-vague detections out of the box using structured prompt engineering."),
    ("PHASE 3 : FAIL-SAFE ENGINEERING", "Constructed local evaluation arena, dashboard, and 31 tests.",
     "Guaranteed that even if model fine-tuning was delayed before the bell, we possessed an authentic, real-time, working system. No dummy mocks, no synthetic detections - an honest, audited engine."),
    ("PHASE 4 : DATASET LANDS (1 HOUR BEFORE)", "Immediate plug-and-play evaluation on L1, L2, and L3.",
     "When the evaluation dataset finally arrived, we did not scramble to write code. We plugged it straight into our pre-validated pipeline, generating valid submissions within minutes."),
]

ty2, th2 = 2.02, 0.98
for i, (t_step, t_bold, t_body) in enumerate(timeline):
    y = ty2 + i * (th2 + 0.11)
    rect(s2, MX, y, lw, th2, fill=SURFACE, line=LINE, lw=1.0, shape=MSO_SHAPE.ROUNDED_RECTANGLE, rad=0.08)
    text(s2, MX + 0.18, y + 0.12, 1.65, 0.24, [P([R(t_step, size=7.8, color=ACCENT if i>1 else (SEV_CRIT if i==0 else PURPLE), bold=True, font=MONO)])])
    vline(s2, MX + 1.85, y + 0.12, th2 - 0.24, LINE2, t=0.012)
    text(s2, MX + 2.02, y + 0.10, lw - 2.20, 0.26, [P([R(t_bold, size=9.2, color=INK, bold=True)])])
    text(s2, MX + 2.02, y + 0.36, lw - 2.20, th2 - 0.42, [P([R(t_body, size=8.0, color=INK2)], line=1.16)])

# Right Column: Empirical Takeaways & Why This Sets Us Apart
rw2 = SW - MX - (MX + lw + 0.35)
rx_r = MX + lw + 0.35
text(s2, rx_r, 1.74, rw2, 0.24, [P([R("WHAT SETS SENTINEL APART FROM THE REST", size=8.5, color=ACCENT_INK, bold=True, font=MONO, spc=1.6)])])

takeaways = [
    ("AUTHENTICITY", OK, "GENUINE DETECTIONS, ZERO HALLUCINATIONS",
     "We did not rely on ungrounded VLM babble. Every alert requires affirmative visual proof across 3 targeted criteria questions plus dual-temperature self-consistency before entering the ledger."),
    ("EDGE-NATIVE", ACCENT, "TRUE REAL-TIME ON 1x CONSUMER GPU",
     "Achieved 18% wake ratio. The gate processes frames in 5ms; the VLM only touches 10 out of 56 seconds. Operates with sub-second responsiveness on a single laptop RTX 4050 or T4."),
    ("TACTICAL GRIT", PURPLE, "A WORKING PRODUCT, NOT JUST A MODEL",
     "Delivered a live operator UI (FastAPI/WebSockets), automated Arena scoring replica, and audited runtime ledger (plus/minus 2% clock accuracy). The system is production-grade software."),
    ("ROBUST BIAS FIX", ACCENT_INK, "CONFUSABLE TWIN DISAMBIGUATION",
     "Empirically diagnosed and resolved sports-vs-fighting confusion. Sharpened normal prompt anchors so athletic play, rain reflections, and red lights never trigger catastrophic false alarms."),
]

for i, (tag_r, col_r, t_title, t_body) in enumerate(takeaways):
    y = ty2 + i * (th2 + 0.11)
    rect(s2, rx_r, y, rw2, th2, fill=SURFACE, line=LINE, lw=1.0, shape=MSO_SHAPE.ROUNDED_RECTANGLE, rad=0.08)
    rect(s2, rx_r, y + 0.10, 0.045, th2 - 0.20, fill=col_r)
    text(s2, rx_r + 0.18, y + 0.12, 1.25, 0.24, [P([R(tag_r, size=7.8, color=col_r, bold=True, font=MONO)])])
    text(s2, rx_r + 1.48, y + 0.10, rw2 - 1.65, 0.26, [P([R(t_title, size=8.8, color=INK, bold=True, font=MONO, spc=0.4)])])
    text(s2, rx_r + 0.18, y + 0.38, rw2 - 0.36, th2 - 0.44, [P([R(t_body, size=8.0, color=INK2)], line=1.16)])

# Bottom Evolution Strip (Strictly no em dashes)
cy2 = 6.42
rect(s2, MX, cy2, CW, 0.52, fill=ACCENT_SOFT, line=None, shape=MSO_SHAPE.ROUNDED_RECTANGLE, rad=0.14)
rect(s2, MX, cy2, 0.05, 0.52, fill=ACCENT)
text(s2, MX + 0.25, cy2, CW - 0.5, 0.52, [P([
    R("OUR CORE LESSON : ", size=8.2, color=ACCENT_INK, bold=True, font=MONO, spc=1.0),
    R("In visual anomaly detection, data bottlenecks and model size do not define your ceiling. "
      "Composure, mathematical shape priors, calibrated abstention, and dual-stage cascading turn a 4B VLM into an unbeatable edge detector.", size=8.2, color=INK2)
])], anchor=MSO_ANCHOR.MIDDLE)

footer(s2, "2 / 2")

out_path = r"D:\FlytBase Hackathon\SENTINEL_Championship_Narrative.pptx"
prs.save(out_path)
print(f"Successfully generated: {out_path} (2 slides, ZERO em dashes)")
