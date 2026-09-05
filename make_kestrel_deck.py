#!/usr/bin/env python3
"""
KESTREL — light-theme presentation deck generator.
Color theme, fonts and structure extracted from the live site:
https://kestrel-flyt-base-3kcq-roan.vercel.app/
(light-theme tokens from /_next/static/css/5b12f3a22b9cfca2.css)
Output: KESTREL_Pitch_Deck.pptx
"""
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# ---------------- exact light-theme palette (site tokens) ----------------
BG          = "f7f9fc"   # --bg
SURFACE     = "ffffff"   # --surface
SURFACE2    = "f2f6fb"   # --surface-2
SURFACE3    = "e8eff7"   # --surface-3
INK         = "0d1526"   # --ink
INK2        = "43526b"   # --ink-2
INK3        = "6b7c96"   # --ink-3
INK4        = "55637c"   # --ink-4
LINE        = "e0e8f2"   # --line
LINE2       = "cfdcea"   # --line-2
ACCENT      = "0ea5e9"   # --accent
ACCENT2     = "38bdf8"   # --accent-2
ACCENT3     = "7dd3fc"   # --accent-3
ACCENT_INK  = "0369a1"   # --accent-ink
ACCENT_SOFT = "e6f5fe"   # --accent-soft
OK          = "10a37f"   # --ok
SEV_CRIT    = "e11d48"   # --sev-critical
SEV_HIGH    = "f0682f"   # --sev-high
SEV_MED     = "f59e0b"   # --sev-medium
SEV_LOW     = "0ea5e9"   # --sev-low
SEV_INFO    = "64748b"   # --sev-info

SANS = "Inter"
MONO = "JetBrains Mono NL"   # NL variant: same glyphs, renders at all sizes in PPT

SW, SH = 13.333, 7.5
MX = 0.6                      # side margin
CW = SW - 2 * MX              # content width

def C(h):
    return RGBColor.from_string(h)

def rect(slide, x, y, w, h, fill=None, line=None, lw=0.75,
         shape=MSO_SHAPE.RECTANGLE, rad=None):
    sp = slide.shapes.add_shape(shape, Inches(x), Inches(y),
                                Inches(w), Inches(h))
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
        try:
            sp.adjustments[0] = rad
        except Exception:
            pass
    return sp

def _apply(run, txt, size, color, font, bold, italic, spc):
    run.text = txt
    f = run.font
    f.size = Pt(size); f.bold = bold; f.italic = italic
    f.name = font
    f.color.rgb = C(color)
    if spc:
        run._r.get_or_add_rPr().set("spc", str(int(spc * 100)))

def text(slide, x, y, w, h, paras, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, wrap=True):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = wrap
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = 0
    tf.margin_top = tf.margin_bottom = 0
    for i, p in enumerate(paras):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = p.get("align", align)
        if p.get("line"):   para.line_spacing = p["line"]
        if p.get("before"): para.space_before = Pt(p["before"])
        if p.get("after"):  para.space_after  = Pt(p["after"])
        for txt, o in p["runs"]:
            _apply(para.add_run(), txt,
                   o.get("size", 12), o.get("color", INK),
                   o.get("font", SANS), o.get("bold", False),
                   o.get("italic", False), o.get("spc", 0))
    return tb

def P(runs, **kw):
    d = {"runs": runs}; d.update(kw); return d

def R(txt, **o):
    return (txt, o)

def chip(slide, x, y, label, fill=SURFACE, line=LINE2, color=ACCENT_INK,
         size=8, bold=True, h=0.30, font=MONO, spc=1.2, rad=0.5,
         char_w=0.092, pad=0.40):
    w = pad + len(label) * size * char_w / 8.0
    rect(slide, x, y, w, h, fill=fill, line=line, lw=0.75,
         shape=MSO_SHAPE.ROUNDED_RECTANGLE, rad=rad)
    text(slide, x, y - 0.012, w, h,
         [P([R(label, size=size, color=color, bold=bold, font=font,
               spc=spc)])], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    return w

def hline(slide, x, y, w, color=LINE, t=0.012):
    rect(slide, x, y, w, t, fill=color)

def vline(slide, x, y, h, color=LINE, t=0.012):
    rect(slide, x, y, t, h, fill=color)

def dot(slide, x, y, d, color):
    rect(slide, x, y, d, d, fill=color, shape=MSO_SHAPE.OVAL)

# ---------------- slide scaffolding ----------------
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

def header(s, eyebrow, title, tag):
    text(s, MX, 0.56, 9.0, 0.3, [P([R(eyebrow, size=10, color=ACCENT_INK,
         bold=True, font=MONO, spc=2.6)])])
    text(s, MX, 0.88, 9.6, 0.84, [P([R(title, size=25, color=INK,
         bold=True, spc=-0.6)], line=1.0)])
    text(s, SW - MX - 4.4, 0.60, 4.4, 0.25, [P([R(tag, size=9, color=INK3,
         font=MONO, spc=1.5)])], align=PP_ALIGN.RIGHT)
    hline(s, MX, 1.78, CW, LINE2, t=0.014)

def footer(s, label):
    PAGE[0] += 1
    text(s, MX, SH - 0.345, 7.5, 0.24, [P([R(
        "KESTREL — AUTONOMOUS DRONE SECURITY ANALYST",
        size=8, color=INK3, font=MONO, spc=1.8)])])
    text(s, SW - MX - 3.6, SH - 0.345, 3.6, 0.24,
         [P([R(label + "  ·  ", size=8, color=INK3, font=MONO, spc=1.2),
             R("%02d" % PAGE[0], size=9, color=ACCENT_INK, bold=True,
               font=MONO, spc=1.2)])], align=PP_ALIGN.RIGHT)

def stat_card(s, x, y, w, h, big, label, color=ACCENT, fill=SURFACE):
    rect(s, x, y, w, h, fill=fill, line=LINE, lw=1.0,
         shape=MSO_SHAPE.ROUNDED_RECTANGLE, rad=0.06)
    rect(s, x, y + 0.16, 0.045, h - 0.32, fill=color)
    text(s, x + 0.28, y + 0.20, w - 0.5, 0.55,
         [P([R(big, size=25, color=INK, bold=True, font=MONO)])])
    text(s, x + 0.28, y + 0.78, w - 0.52, h - 0.92,
         [P([R(label, size=9.5, color=INK2, spc=0.3)], line=1.25)])

def band(s, x, y, w, h, runs):
    rect(s, x, y, w, h, fill=ACCENT_SOFT, line=None,
         shape=MSO_SHAPE.ROUNDED_RECTANGLE, rad=0.12)
    rect(s, x, y, 0.05, h, fill=ACCENT)
    text(s, x + 0.32, y, w - 0.64, h,
         [P(runs, line=1.3)], anchor=MSO_ANCHOR.MIDDLE)

# ================= SLIDE 1 — HERO =================
s = new_slide()
PAGE[0] += 1
text(s, MX, 0.56, 7.5, 0.3, [P([R("KESTREL M300 RTK DOCK · PLANT-01 · ARMED",
     size=10, color=ACCENT_INK, bold=True, font=MONO, spc=2.4)])])
chip(s, SW - MX - 2.15, 0.48, "ESTABLISHING LINK", fill=SURFACE,
     line=LINE2, color=OK, size=8, h=0.28)
dot(s, SW - MX - 2.03, 0.575, 0.07, OK)
hline(s, MX, 1.66, CW, LINE2, t=0.014)

# left column
text(s, MX, 1.95, 7.0, 1.1, [P([R("KESTREL", size=58, color=INK, bold=True,
     spc=-2.0)])])
rect(s, MX + 0.02, 2.98, 1.35, 0.07, fill=ACCENT)
text(s, MX, 3.28, 6.6, 0.75, [P([R("The drone security analyst that "
     "never blinks.", size=25, color=INK, bold=True, spc=-0.5)], line=1.05)])
text(s, MX, 4.18, 6.55, 1.15, [P([R(
     "Autonomous drone security analyst. A docked patrol drone produces "
     "about 57,600 frames per shift — almost all of them show the same "
     "empty yard. KESTREL decides which frames deserve attention, "
     "remembers context across days, and proves every claim.",
     size=12.5, color=INK2)], line=1.4)])
px = MX
for lab in ["PERCEIVE", "REMEMBER", "REASON", "ACT", "CONVERSE", "PROVE"]:
    px += chip(s, px, 5.42, lab, size=7.5, h=0.27) + 0.14
cx = MX + chip(s, MX, 5.98, "OPEN THE CONSOLE  →", fill=ACCENT, line=None,
               color="ffffff", size=9, h=0.42, spc=1.6) + 0.18
chip(s, cx, 5.98, "ASK KESTREL", fill=SURFACE, line=LINE2, color=INK,
     size=9, h=0.42, spc=1.6)

# right column — console mock card
cx0, cy0, cw0, ch0 = 7.55, 1.95, 5.18, 3.62
rect(s, cx0, cy0, cw0, ch0, fill=SURFACE, line=LINE2, lw=1.0,
     shape=MSO_SHAPE.ROUNDED_RECTANGLE, rad=0.05)
rect(s, cx0, cy0, cw0, 0.42, fill=SURFACE2, line=None)
hline(s, cx0, cy0 + 0.42, cw0, LINE2, t=0.012)
for i, dc in enumerate([SEV_CRIT, SEV_MED, OK]):
    dot(s, cx0 + 0.22 + i * 0.19, cy0 + 0.17, 0.09, dc)
text(s, cx0 + 0.85, cy0, cw0 - 1.0, 0.42, [P([R("PLANT-01 · LIVE",
     size=8.5, color=INK3, font=MONO, spc=2.0)])], anchor=MSO_ANCHOR.MIDDLE)
rows = [("gate",     "perceptual hash · pixel delta · cpu", "free"),
        ("detect",   "yolo11 on local gpu",                 "~12 ms"),
        ("track",    "bytetrack · identity persists",       "~free"),
        ("embed",    "joint vectors · 2048-d",              "cheap"),
        ("perceive", "vlm structured scene graph",          "~1.3 s"),
        ("escalate", "deep model · out of band",            "async")]
ry = cy0 + 0.62
for name, desc, cost in rows:
    text(s, cx0 + 0.28, ry, 1.15, 0.3, [P([R(name, size=10, color=ACCENT_INK,
         bold=True, font=MONO)])])
    text(s, cx0 + 1.5, ry, 2.6, 0.3, [P([R(desc, size=9, color=INK2,
         font=MONO)])])
    text(s, cx0 + cw0 - 1.05, ry, 0.8, 0.3, [P([R(cost, size=9,
         color=INK3, bold=True, font=MONO)])], align=PP_ALIGN.RIGHT)
    ry += 0.42
hline(s, cx0 + 0.28, ry + 0.06, cw0 - 0.56, LINE, t=0.012)
dot(s, cx0 + 0.28, ry + 0.17, 0.09, OK)
text(s, cx0 + 0.48, ry + 0.08, cw0 - 0.8, 0.3, [P([R(
     "7 moments that mattered — 8 h shift", size=9.5, color=OK, bold=True,
     font=MONO, spc=0.8)])])

# bottom stat strip
hline(s, MX, 6.62, CW, LINE2, t=0.012)
stats = [("57,600", "FRAMES / SHIFT"),
         ("38.9%",  "GATE EFFICIENCY · REAL FOOTAGE"),
         ("8 / 8",  "SCENARIOS PASS")]
sx = MX
for big, lab in stats:
    text(s, sx, 6.74, 2.0, 0.4, [P([R(big, size=15, color=INK, bold=True,
         font=MONO)])])
    text(s, sx + 1.35, 6.82, 2.9, 0.3, [P([R(lab, size=8, color=INK3,
         font=MONO, spc=1.2)])])
    sx += 4.35
text(s, SW - MX - 4.6, 0.56, 2.0, 0.3, [P([R("01", size=9, color=INK3,
     font=MONO, spc=1.5)])], align=PP_ALIGN.RIGHT)

# ================= SLIDE 2 — THE PROBLEM =================
s = new_slide()
header(s, "01 · THE PROBLEM", "Everything is recorded. Almost nothing is watched.",
       "KESTREL / THE PROBLEM")
text(s, MX, 2.0, 6.0, 2.6, [
    P([R("Security footage is reviewed after something has already "
         "happened.", size=14, color=INK, bold=True)], line=1.3, after=10),
    P([R("A guard cannot watch every camera — and an operator who receives "
         "forty alerts a night learns to dismiss all of them. At that "
         "point, the system protects nothing.", size=12.5, color=INK2)],
      line=1.4, after=10),
    P([R("Review after the fact can only document loss. The value has to "
         "exist in the moment the frame is captured — and only if someone "
         "still trusts the alert.", size=12.5, color=INK2)], line=1.4)])
stat_card(s, 7.0, 1.98, 5.73, 1.28, "57,600",
          "frames per eight-hour shift — almost all show the same empty yard",
          color=ACCENT)
stat_card(s, 7.0, 3.38, 5.73, 1.28, "40 / night",
          "alerts is enough to teach an operator to dismiss all of them",
          color=SEV_MED)
stat_card(s, 7.0, 4.78, 5.73, 1.28, "after the fact",
          "is when footage is reviewed — too late to change the outcome",
          color=SEV_CRIT)
band(s, MX, 6.16, CW, 0.86, [
    R("The hard problem is not detection. ", size=13, color=INK, bold=True),
    R("It is deciding what deserves attention, remembering enough context "
      "to know, and being right often enough to stay trusted.",
      size=13, color=INK2)])
footer(s, "THE PROBLEM")

# ================= SLIDE 3 — THE CONSTRAINT =================
s = new_slide()
header(s, "02 · THE CONSTRAINT", "One number shaped every decision.",
       "KESTREL / HOW IT WORKS")
# two big cards with vs arrow
rect(s, MX, 2.05, 5.35, 2.5, fill=SURFACE, line=LINE, lw=1.0,
     shape=MSO_SHAPE.ROUNDED_RECTANGLE, rad=0.06)
text(s, MX + 0.4, 2.35, 4.5, 0.4, [P([R("THE FREE INFERENCE TIER ALLOWS",
     size=9, color=INK3, font=MONO, spc=1.8)])])
text(s, MX + 0.4, 2.75, 4.5, 1.0, [P([R("≈ 40", size=54, color=ACCENT_INK,
     bold=True, font=MONO)])])
text(s, MX + 0.4, 3.85, 4.5, 0.5, [P([R("REQUESTS / MINUTE", size=10,
     color=INK2, bold=True, font=MONO, spc=2.0)])])
rect(s, 6.35, 3.05, 0.62, 0.5, fill=ACCENT, shape=MSO_SHAPE.CHEVRON)
rect(s, 7.38, 2.05, 5.35, 2.5, fill=SURFACE, line=LINE, lw=1.0,
     shape=MSO_SHAPE.ROUNDED_RECTANGLE, rad=0.06)
text(s, 7.78, 2.35, 4.5, 0.4, [P([R("A DOCKED PATROL DRONE PRODUCES",
     size=9, color=INK3, font=MONO, spc=1.8)])])
text(s, 7.78, 2.75, 4.5, 1.0, [P([R("57,600", size=54, color=INK, bold=True,
     font=MONO)])])
text(s, 7.78, 3.85, 4.5, 0.5, [P([R("FRAMES / SHIFT", size=10, color=INK2,
     bold=True, font=MONO, spc=2.0)])])
band(s, MX, 4.95, CW, 1.5, [
    R("Captioning every frame is impossible and pointless. ",
      size=14, color=INK, bold=True),
    R("Every architectural decision in KESTREL follows from that one "
      "constraint — spend nothing on frames that show nothing, and every "
      "expensive call only on the frames that earn it.",
      size=14, color=INK2)])
footer(s, "THE CONSTRAINT")

# ================= SLIDE 4 — THE TIERED PIPELINE =================
s = new_slide()
header(s, "03 · HOW IT WORKS",
       "Five tiers. Each runs only when the tier before it earns the spend.",
       "KESTREL / PIPELINE")
tiers = [
    ("0", "GATE", "FREE",
     "Perceptual hash, pixel delta and embedding novelty, all on CPU. "
     "Decides whether a frame is worth spending anything on at all."),
    ("1", "DETECT", "~12 MS",
     "YOLO11 on the local GPU, or Grounding DINO when the query is "
     "open-vocabulary. On-device — no API budget, no rate limit."),
    ("1.5", "TRACK", "~FREE",
     "ByteTrack. Identity that persists across frames. Without it, every "
     "claim about duration is a guess."),
    ("2", "EMBED", "CHEAP",
     "Joint image/text vectors in one 2048-dimension space, for "
     "re-identification and search that works on appearance."),
    ("3", "PERCEIVE", "~1.3 S",
     "A vision-language model returns a structured scene graph: objects, "
     "colours, activities, anomalies. Never prose."),
    ("4", "ESCALATE", "ASYNC",
     "The deep model measured 57–84 s, so it runs out of band and upgrades "
     "the record afterwards, split by the kind of doubt."),
]
gx, gy = MX, 1.98
gw, gh, ggap = (CW - 0.6) / 3.0, 2.28, 0.3
for i, (num, name, cost, desc) in enumerate(tiers):
    x = gx + (i % 3) * (gw + ggap)
    y = gy + (i // 3) * (gh + ggap)
    rect(s, x, y, gw, gh, fill=SURFACE, line=LINE, lw=1.0,
         shape=MSO_SHAPE.ROUNDED_RECTANGLE, rad=0.05)
    rect(s, x, y, gw, 0.06, fill=ACCENT if i < 3 else ACCENT2)
    text(s, x + 0.26, y + 0.24, 0.9, 0.55, [P([R(num, size=24, color=ACCENT,
         bold=True, font=MONO)])])
    text(s, x + 1.05, y + 0.30, gw - 2.2, 0.35, [P([R(name, size=14,
         color=INK, bold=True, spc=0.4)])])
    text(s, x + gw - 1.15, y + 0.34, 0.9, 0.3, [P([R(cost, size=8.5,
         color=ACCENT_INK, bold=True, font=MONO)])], align=PP_ALIGN.RIGHT)
    hline(s, x + 0.26, y + 0.86, gw - 0.52, LINE, t=0.012)
    text(s, x + 0.26, y + 1.02, gw - 0.52, gh - 1.2, [P([R(desc, size=9.5,
         color=INK2)], line=1.3)])
footer(s, "PIPELINE")

# ================= SLIDE 5 — HONEST MEASUREMENT =================
s = new_slide()
header(s, "03 · HONEST MEASUREMENT",
       "38.9% on real footage. Both numbers, with their conditions.",
       "KESTREL / GATE EFFICIENCY")
def gauge(s, x, y, w, label, pct_txt, pct, color, note):
    text(s, x, y, w - 1.4, 0.3, [P([R(label, size=10.5, color=INK,
         bold=True, font=MONO, spc=1.2)])])
    text(s, x + w - 1.4, y - 0.04, 1.4, 0.35, [P([R(pct_txt, size=16,
         color=color, bold=True, font=MONO)])], align=PP_ALIGN.RIGHT)
    rect(s, x, y + 0.38, w, 0.34, fill=SURFACE3,
         shape=MSO_SHAPE.ROUNDED_RECTANGLE, rad=0.5)
    rect(s, x, y + 0.38, max(w * pct, 0.55), 0.34, fill=color,
         shape=MSO_SHAPE.ROUNDED_RECTANGLE, rad=0.5)
    text(s, x, y + 0.82, w, 0.3, [P([R(note, size=9, color=INK3, font=MONO,
         spc=0.4)])])
gauge(s, MX, 2.05, CW, "GATE EFFICIENCY — REAL FOOTAGE", "38.9%", 0.389,
      ACCENT, "cv demo reels authored for continuous motion — close to "
              "worst case for a gate that skips static frames")
gauge(s, MX, 3.30, CW, "GATE EFFICIENCY — CONSTRUCTED IDLE CONTEXT", "96.7%",
      0.967, OK, "static idle footage — the condition the gate was designed for")
# design-plan marker
rect(s, MX + CW * 0.94, 3.30, 0.02, 0.76, fill=INK3)
text(s, MX + CW * 0.94 - 1.55, 4.16, 3.1, 0.28, [P([R("DESIGN PLAN ≈ 94%",
     size=8, color=INK3, font=MONO, spc=1.2)])], align=PP_ALIGN.CENTER)
band(s, MX, 4.62, CW, 1.55, [
    R("The design plan predicted about 94%. Measurement said 38.9%. ",
      size=13.5, color=INK, bold=True),
    R("Every licence-clean clip available is a CV demo reel authored for "
      "continuous motion — close to worst case for a gate that skips static "
      "frames. Both figures are reported, with their conditions, rather "
      "than only the flattering one.", size=13.5, color=INK2)])
footer(s, "GATE EFFICIENCY")

# ================= SLIDE 6 — BEYOND DETECTION =================
s = new_slide()
header(s, "04 · BEYOND DETECTION",
       "Nine capabilities the brief did not ask for.",
       "KESTREL / CAPABILITIES")
caps = [
    ("01", "Rules you write in English",
     "Type a requirement, get a validated temporal rule — then preview what "
     "it would have done against indexed history before it may fire."),
    ("02", "Open-vocabulary detection",
     "Ask for “a traffic cone” and the detector grounds the phrase "
     "directly — no retraining, no fixed class list."),
    ("03", "Memory that spans days",
     "The same vehicle, seventh visit, first time ever after midnight. No "
     "single frame is alarming; only the pattern is."),
    ("04", "A temporal memory pyramid",
     "Frame → clip → event → shift → day, weighted by salience — eight "
     "hours collapses to ~12,000 queryable tokens."),
    ("05", "A normalcy baseline that abstains",
     "Counts per zone, hour and class, z-scored — it declines to judge "
     "anything until it has three days of history."),
    ("06", "Alerts you can fly to",
     "Geo-projected coordinates, accuracy radius, bearing, ETA and a "
     "geofence check — not just a description of what happened."),
    ("07", "Findings only a fleet reveals",
     "A subject seen at three sites in five days is a reconnaissance "
     "pattern no single site could possibly detect."),
    ("08", "Search that understands appearance",
     "SQL, caption vectors and image vectors fused by reciprocal rank — "
     "“a white pickup” finds frames whose captions never said it."),
    ("09", "Prove, not claim",
     "Every measured number is produced by a benchmark in the repository — "
     "precision, recall and gate efficiency included."),
]
gx, gy = MX, 1.94
gw, gh, ggap = (CW - 0.6) / 3.0, 1.56, 0.3
for i, (num, name, desc) in enumerate(caps):
    x = gx + (i % 3) * (gw + ggap)
    y = gy + (i // 3) * (gh + ggap)
    rect(s, x, y, gw, gh, fill=SURFACE, line=LINE, lw=1.0,
         shape=MSO_SHAPE.ROUNDED_RECTANGLE, rad=0.07)
    text(s, x + 0.22, y + 0.16, 0.6, 0.3, [P([R(num, size=10, color=ACCENT,
         bold=True, font=MONO)])])
    text(s, x + 0.22, y + 0.40, gw - 0.44, 0.3, [P([R(name, size=11.5,
         color=INK, bold=True)])])
    text(s, x + 0.22, y + 0.72, gw - 0.44, gh - 0.84, [P([R(desc, size=8.5,
         color=INK2)], line=1.22)])
footer(s, "CAPABILITIES")

# ================= SLIDE 7 — ALERTS YOU CAN FLY TO =================
s = new_slide()
header(s, "05 · ALERTS YOU CAN FLY TO", "A coordinate, not a description.",
       "KESTREL / DISPATCH")
# alert card
ax, ay, aw, ah = MX, 1.98, 6.3, 3.9
rect(s, ax, ay, aw, ah, fill=SURFACE, line=LINE2, lw=1.0,
     shape=MSO_SHAPE.ROUNDED_RECTANGLE, rad=0.05)
rect(s, ax, ay + 0.12, 0.055, ah - 0.24, fill=SEV_CRIT)
text(s, ax + 0.32, ay + 0.24, aw - 0.6, 0.32, [P([
    R("● PERSON IN RESTRICTED CORE", size=12, color=SEV_CRIT, bold=True,
      font=MONO, spc=0.6)])])
text(s, ax + 0.32, ay + 0.58, aw - 0.6, 0.3, [P([
    R("02:14  ·  ALT-0a15-0007", size=9.5, color=INK3, font=MONO, spc=1.2)])])
hline(s, ax + 0.32, ay + 0.98, aw - 0.64, LINE, t=0.012)
rows = [("COORDINATES",     "18.760018, 73.862886", INK),
        ("ACCURACY",        "± 4.9 m",              INK),
        ("BEARING FROM DOCK", "041°  ·  212 m",     INK),
        ("ETA",             "1 MIN 04 S",           INK),
        ("RECOMMENDED ALT", "28 M AGL",             INK),
        ("GEOFENCE",        "INSIDE · CLEARED",     OK)]
ry = ay + 1.14
for lab, val, col in rows:
    text(s, ax + 0.32, ry, 2.4, 0.3, [P([R(lab, size=8.5, color=INK3,
         font=MONO, spc=1.4)])])
    text(s, ax + 2.75, ry - 0.03, aw - 3.1, 0.32, [P([R(val, size=11,
         color=col, bold=True, font=MONO)])], align=PP_ALIGN.RIGHT)
    ry += 0.45
# right column notes
text(s, 7.35, 2.05, 5.35, 3.6, [
    P([R("Geo-projected to ground truth through the telemetry — altitude, "
         "gimbal pitch and yaw — with an honest accuracy radius derived "
         "from the projection geometry rather than asserted.",
         size=12, color=INK2)], line=1.4, after=12),
    P([R("Bearing and ETA from the dock, a recommended stand-off altitude, "
         "and a geofence verdict.", size=12, color=INK2)], line=1.4,
      after=12),
    P([R("The decision in front of the operator becomes ", size=12,
         color=INK2),
       R("“launch or do not”", size=12, color=ACCENT_INK, bold=True),
       R(" rather than “interpret this”.", size=12, color=INK2)], line=1.4)])
band(s, 7.35, 5.05, 5.35, 0.83, [
    R("A coordinate with false precision is worse than no coordinate at "
      "all.", size=11.5, color=INK, bold=True)])
band(s, MX, 6.05, aw, 0.83, [
    R("Flat-ground homography — the accuracy radius grows with slant "
      "range, and the projection is stated as an estimate.",
      size=11.5, color=INK2)])
footer(s, "DISPATCH")

# ================= SLIDE 8 — THE CLOSED LOOP =================
s = new_slide()
header(s, "06 · THE CLOSED LOOP", "An alert is not the end of the job.",
       "KESTREL / MISSION")
steps = [("1", "ALERT RAISED", "with dispatch coordinates"),
         ("2", "MISSION PLANNED", "feasibility checked"),
         ("3", "HUMAN APPROVES", "the agent cannot"),
         ("4", "DRONE FLIES", "closer vantage, better view"),
         ("5", "CONFIDENCE REVISED", "on new evidence")]
sx, sy, sw_, sh_ = MX, 2.35, 2.19, 2.0
for i, (num, name, desc) in enumerate(steps):
    x = sx + i * (sw_ + 0.295)
    rect(s, x, sy, sw_, sh_, fill=SURFACE, line=LINE, lw=1.0,
         shape=MSO_SHAPE.ROUNDED_RECTANGLE, rad=0.07)
    dot(s, x + 0.24, sy + 0.24, 0.42, ACCENT_SOFT)
    text(s, x + 0.24, sy + 0.245, 0.42, 0.4, [P([R(num, size=13,
         color=ACCENT_INK, bold=True, font=MONO)])],
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    text(s, x + 0.24, sy + 0.86, sw_ - 0.48, 0.55, [P([R(name, size=11,
         color=INK, bold=True, spc=0.3)], line=1.05)])
    text(s, x + 0.24, sy + 1.36, sw_ - 0.48, 0.5, [P([R(desc, size=8.5,
         color=INK2)], line=1.2)])
    if i < 4:
        text(s, x + sw_ - 0.02, sy + sh_ / 2 - 0.22, 0.36, 0.4,
             [P([R("→", size=16, color=ACCENT, bold=True, font=SANS)])],
             align=PP_ALIGN.CENTER)
band(s, MX, 4.75, CW, 1.5, [
    R("Proposal and approval are separate code paths. ",
      size=13.5, color=INK, bold=True),
    R("Enforced in the tool registry rather than prompt wording — and a "
      "test asserts the agent’s own loop can never set the approval flag. "
      "The loop closes because the new vantage point re-enters perception: "
      "the improvement is measured, not asserted.", size=13.5, color=INK2)])
chip(s, MX, 6.42, "4 GATED TOOLS", fill=SURFACE, color=SEV_HIGH, size=8.5)
chip(s, MX + 1.85, 6.42, "0 PROMPT-ONLY GUARDS", fill=SURFACE,
     color=ACCENT_INK, size=8.5)
chip(s, MX + 4.55, 6.42, "1 TEST PROVES THE BOUNDARY", fill=SURFACE,
     color=OK, size=8.5)
footer(s, "MISSION")

# ================= SLIDE 9 — THE CONTROL PLANE =================
s = new_slide()
header(s, "07 · THE CONTROL PLANE",
       "Ask it anything. It will tell you when it does not know.",
       "KESTREL / TOOLS")
kpis = [("27", "TOOLS", "every capability, callable", ACCENT),
        ("8",  "CLASSES", "one registry, eight verbs", ACCENT2),
        ("4",  "GATED", "no aircraft moves without a human decision", SEV_HIGH),
        ("0",  "PROMPT-ONLY", "the boundary is a code path, and a test "
         "proves it", OK)]
kw = (CW - 0.9) / 4.0
for i, (big, name, desc, col) in enumerate(kpis):
    x = MX + i * (kw + 0.3)
    rect(s, x, 1.98, kw, 2.1, fill=SURFACE, line=LINE, lw=1.0,
         shape=MSO_SHAPE.ROUNDED_RECTANGLE, rad=0.06)
    rect(s, x, 1.98, kw, 0.06, fill=col)
    text(s, x + 0.28, 2.24, kw - 0.5, 0.85, [P([R(big, size=40, color=INK,
         bold=True, font=MONO)])])
    text(s, x + 0.28, 3.08, kw - 0.5, 0.3, [P([R(name, size=11, color=col,
         bold=True, font=MONO, spc=1.8)])])
    text(s, x + 0.28, 3.40, kw - 0.56, 0.6, [P([R(desc, size=9.5,
         color=INK2)], line=1.25)])
text(s, MX, 4.42, CW, 0.35, [P([R("EIGHT TOOL CLASSES", size=9.5,
     color=INK3, bold=True, font=MONO, spc=2.0)])])
kx = MX
for lab in ["RETRIEVE", "ANALYSE", "AUTHOR", "ACT", "OPERATE", "NAVIGATE",
            "FLEET", "EXPLAIN"]:
    kx += chip(s, kx, 4.78, lab, size=8.5, h=0.34) + 0.16
text(s, MX, 5.45, CW, 1.1, [
    P([R("Conversation is not a feature bolted to the side; it is the way "
         "every capability is reachable. ", size=12.5, color=INK2),
       R("Answers arrive as live interface, not paragraphs. ",
         size=12.5, color=INK, bold=True),
       R("Each cited identifier is checked against the tool results that "
         "produced it — a citation that resolves to nothing is marked "
         "unverified rather than rendered as fact.", size=12.5,
         color=INK2)], line=1.4)])
footer(s, "TOOLS")

# ================= SLIDE 10 — MEASURED, NOT CLAIMED =================
s = new_slide()
header(s, "08 · MEASURED, NOT CLAIMED",
       "Every number was produced by a benchmark in the repository.",
       "KESTREL / EVALS")
evals = [("8 / 8", "SCENARIOS PASS", "labelled scenario suite, end to end",
          ACCENT),
         ("9 · 0 · 0", "TP · FP · FN", "zero false alarms, zero misses",
          ACCENT),
         ("1.00", "PRECISION AND RECALL", "F1 = 1.00 on the labelled suite",
          ACCENT_INK),
         ("0.975", "MEAN P@K", "hybrid retrieval, one ingested session",
          ACCENT2),
         ("6 / 6", "CHAOS FAULTS SURVIVED", "no key, cassette miss, corrupt "
          "frame, timeout", OK)]
ew = (CW - 1.2) / 5.0
for i, (big, name, desc, col) in enumerate(evals):
    x = MX + i * (ew + 0.3)
    rect(s, x, 1.98, ew, 2.35, fill=SURFACE, line=LINE, lw=1.0,
         shape=MSO_SHAPE.ROUNDED_RECTANGLE, rad=0.06)
    rect(s, x, 1.98, ew, 0.06, fill=col)
    text(s, x + 0.22, 2.28, ew - 0.44, 0.6, [P([R(big, size=23, color=INK,
         bold=True, font=MONO)])])
    text(s, x + 0.22, 2.92, ew - 0.44, 0.55, [P([R(name, size=8.5,
         color=col, bold=True, font=MONO, spc=0.8)], line=1.2)])
    text(s, x + 0.22, 3.44, ew - 0.44, 0.8, [P([R(desc, size=8.5,
         color=INK2)], line=1.22)])
band(s, MX, 4.62, CW, 1.55, [
    R("True negatives are weighted equally with true positives. ",
      size=13.5, color=INK, bold=True),
    R("Routine delivery, wildlife at the fence, shift change — the suite "
      "scores them too. A security system that cries wolf gets switched "
      "off, and then it protects nothing.", size=13.5, color=INK2)])
footer(s, "EVALS")

# ================= SLIDE 11 — WHAT IS REAL, AND WHAT IS NOT =========
s = new_slide()
header(s, "09 · HONESTY", "What is real, and what is not.",
       "KESTREL / DATA PROVENANCE")
# real column
rx, ry0, rw, rh = MX, 1.98, (CW - 0.35) / 2.0, 3.5
rect(s, rx, ry0, rw, rh, fill=SURFACE, line=LINE, lw=1.0,
     shape=MSO_SHAPE.ROUNDED_RECTANGLE, rad=0.05)
rect(s, rx, ry0, rw, 0.06, fill=OK)
chip(s, rx + 0.28, ry0 + 0.24, "REAL", fill=OK, line=None, color="ffffff",
     size=9, h=0.32)
real_items = ["The video — CC BY 4.0 footage through the real pipeline",
              "The detections, tracks and embeddings",
              "The captions — from a live vision-language model",
              "The rules, memory, retrieval and audit chain",
              "Every measured number on the evals page"]
iy = ry0 + 0.78
for it in real_items:
    text(s, rx + 0.30, iy, 0.3, 0.3, [P([R("✓", size=12, color=OK,
         bold=True, font=SANS)])])
    text(s, rx + 0.62, iy + 0.02, rw - 0.92, 0.4, [P([R(it, size=11.5,
         color=INK2)], line=1.15)])
    iy += 0.52
# simulated column
sx2 = MX + rw + 0.35
rect(s, sx2, ry0, rw, rh, fill=SURFACE2, line=LINE, lw=1.0,
     shape=MSO_SHAPE.ROUNDED_RECTANGLE, rad=0.05)
rect(s, sx2, ry0, rw, 0.06, fill=SEV_MED)
chip(s, sx2 + 0.28, ry0 + 0.24, "SIMULATED", fill=SEV_MED, line=None,
     color="ffffff", size=9, h=0.32)
sim_items = ["The telemetry — there is no aircraft",
             "The site geometry and zone definitions",
             "Every fleet site except the flagship plant",
             "Mission execution — integrated, not flown"]
iy = ry0 + 0.78
for it in sim_items:
    text(s, sx2 + 0.30, iy, 0.3, 0.3, [P([R("~", size=13, color=SEV_MED,
         bold=True, font=MONO)])])
    text(s, sx2 + 0.62, iy + 0.02, rw - 0.92, 0.4, [P([R(it, size=11.5,
         color=INK2)], line=1.15)])
    iy += 0.52
band(s, MX, 5.72, CW, 1.2, [
    R("Simulated data is labelled wherever it appears — including in the "
      "API payloads. ", size=12.5, color=INK, bold=True),
    R("A portfolio view implying more live aircraft than exist would be "
      "the one thing capable of discrediting everything else here.",
      size=12.5, color=INK2)])
footer(s, "DATA PROVENANCE")

# ================= SLIDE 12 — CLOSING =================
s = new_slide()
PAGE[0] += 1
text(s, MX, 0.56, 8.0, 0.3, [P([R("KESTREL M300 RTK DOCK · PLANT-01 · ARMED",
     size=10, color=ACCENT_INK, bold=True, font=MONO, spc=2.4)])])
chip(s, SW - MX - 2.15, 0.48, "CONSOLE READY", fill=SURFACE,
     line=LINE2, color=OK, size=8, h=0.28)
dot(s, SW - MX - 2.03, 0.575, 0.07, OK)
hline(s, MX, 1.66, CW, LINE2, t=0.014)
text(s, MX, 2.15, 12.1, 1.7, [
    P([R("Ask it what happened ", size=42, color=INK, bold=True, spc=-1.2),
       R("last night.", size=42, color=ACCENT_INK, bold=True, spc=-1.2)],
      line=1.05)])
rect(s, MX + 0.02, 3.62, 1.35, 0.07, fill=ACCENT)
text(s, MX, 3.92, 8.6, 0.9, [P([R(
     "Everything KESTREL can do is reachable in conversation — and it will "
     "tell you honestly when it has no evidence to answer with.",
     size=14, color=INK2)], line=1.4)])
px = MX
for lab in ["PERCEIVE", "REMEMBER", "REASON", "ACT", "CONVERSE", "PROVE"]:
    px += chip(s, px, 5.05, lab, size=7.5, h=0.27) + 0.14
cx = MX + chip(s, MX, 5.62, "OPEN THE CONSOLE  →", fill=ACCENT, line=None,
               color="ffffff", size=9, h=0.42, spc=1.6) + 0.18
chip(s, cx, 5.62, "ASK KESTREL", fill=SURFACE, line=LINE2, color=INK,
     size=9, h=0.42, spc=1.6)
hline(s, MX, 6.62, CW, LINE2, t=0.012)
text(s, MX, 6.76, 7.0, 0.35, [P([R("MADE WITH ♥ BY ATHARVA AWADE",
     size=9, color=INK3, bold=True, font=MONO, spc=1.8)])])
text(s, SW - MX - 5.0, 6.76, 5.0, 0.35, [P([R(
     "kestrel-flyt-base-3kcq-roan.vercel.app", size=9, color=ACCENT_INK,
     bold=True, font=MONO, spc=0.8)])], align=PP_ALIGN.RIGHT)

# ---------------- save ----------------
out = r"d:\FlytBase Hackathon\KESTREL_Pitch_Deck.pptx"
prs.save(out)
print("saved:", out, "| slides:", len(prs.slides._sldIdLst))







