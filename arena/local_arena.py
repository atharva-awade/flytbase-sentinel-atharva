"""Local Arena — a replica of the hackathon evaluation site for dry-running submissions.

    python -m arena.local_arena --gt data/test/ground_truth.csv --videos data/test/videos --port 8090
    open http://localhost:8090

Behaves like the real arena:
  * manifest.json / starter template downloads derived from the ground truth you point it at
  * upload → arena-style validation table (a rejected file does NOT use a run)
  * a file only updates the videos it mentions; omitted videos keep their previous answer;
    never-answered videos score as normal; every accepted upload re-scores the whole sheet
    and REPLACES the previous score (no best-of)
  * leaderboard columns: D1 /25 · D2 /35 · D3 /40 · Marks /100 · RTF · Speed · Reason · Total
  * "My Submissions" history with the sheet that currently counts

What is exact: every stated rule (normal-video zero, IoU >= 0.5 one-to-one matching, pooled L1, level points).
What is estimated (the real arena does not publish it): the L2/L3 alert/match/timing weights, the Speed and
Reasoning bonus formulas. Those are labelled "est." in the UI. Treat deltas between runs as reliable, absolute
numbers as approximate.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import uvicorn  # noqa: E402
from fastapi import FastAPI, File, Form, UploadFile  # noqa: E402
from fastapi.responses import HTMLResponse, JSONResponse  # noqa: E402

from sentinel.manifest import gt_by_video, load_ground_truth, load_videos_csv, manifest_from_gt  # noqa: E402
from sentinel.schema import Prediction, Submission  # noqa: E402
from sentinel.scorer import LEVEL_POINTS, score  # noqa: E402
from sentinel.validate import is_acceptable, validate_submission  # noqa: E402

app = FastAPI(title="Local Arena")
STATE_DIR = ROOT / "arena" / "state"
GT: list = []
MANIFEST: list = []
DURATIONS: dict[str, float] = {}


def _durations(videos_dir: Path | None, videos_csv: Path | None) -> dict[str, float]:
    out: dict[str, float] = {}
    by = gt_by_video(GT)
    if videos_dir and videos_dir.exists():
        import cv2
        mapping = load_videos_csv(videos_csv) if videos_csv and videos_csv.exists() else {}
        for vid in by:
            cands = [videos_dir / mapping[vid]] if vid in mapping else []
            cands += [videos_dir / f"{vid}{e}" for e in (".mp4", ".MP4", ".mov", ".avi", ".mkv")]
            for c in cands:
                if c.exists():
                    cap = cv2.VideoCapture(str(c)); fps = cap.get(cv2.CAP_PROP_FPS) or 25; n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
                    cap.release()
                    if n > 0:
                        out[vid] = n / fps
                    break
    for vid, evs in by.items():   # fallback: last event end + 5 s, or 60 s
        if vid not in out:
            ends = [e.end_time_sec for e in evs if e.end_time_sec is not None]
            out[vid] = (max(ends) + 5.0) if ends else 60.0
    return out


def _state_path(user: str) -> Path:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in user) or "participant"
    return STATE_DIR / f"{safe}.json"


def _load(user: str) -> dict:
    p = _state_path(user)
    return json.loads(p.read_text()) if p.exists() else {"user": user, "sheet": {}, "history": [], "runs": 0}


def _save(st: dict) -> None:
    _state_path(st["user"]).write_text(json.dumps(st, indent=1))


def _score_sheet(sheet: dict[str, dict]) -> dict:
    preds = [Prediction.model_validate(p) for p in sheet.values()]
    rep = score(preds, GT, DURATIONS)
    d = {lvl: rep.level_points.get(lvl, 0.0) for lvl in (1, 2, 3)}
    marks = sum(d.values())
    rtf = rep.rtf if rep.rtf is not None else None
    speed = 5.0 * max(0.0, min(1.0, 1.0 - rtf)) if rtf is not None else 0.0     # est.: full bonus at RTF→0, none at ≥1
    reason = 5.0 * rep.explanation_coverage if any(p.events for p in preds) else 0.0   # est.: share of events explained
    return {
        "d1": round(d[1], 2), "d2": round(d[2], 2), "d3": round(d[3], 2), "marks": round(marks, 2),
        "rtf": None if rtf is None else round(rtf, 3), "speed": round(speed, 2), "reason": round(reason, 2),
        "total": round(marks + speed + reason, 2),
        "l1_anomaly_acc": round(rep.l1_anomaly_acc, 3), "l1_class_acc": round(rep.l1_class_acc, 3),
        "false_alarms": [v.video_id for v in rep.videos if v.false_alarm],
        "fragments": int(sum(v.fragments for v in rep.videos)),
        "videos": [{"video_id": v.video_id, "level": v.level, "gt_normal": v.gt_normal, "score": round(v.score, 3),
                    "alert": round(v.alert, 2), "match": round(v.match, 2), "timing": round(v.timing, 2),
                    "matched": v.matched, "n_gt": v.n_gt, "n_pred": v.n_pred, "false_alarm": v.false_alarm,
                    "fragments": v.fragments, "notes": v.notes,
                    "answered": v.video_id in sheet} for v in sorted(rep.videos, key=lambda x: (x.level, x.video_id))],
        "answered": len(sheet), "n_videos": len(MANIFEST),
    }


@app.get("/api/manifest")
def api_manifest():
    return [{"video_id": m.video_id, "level": m.level} for m in MANIFEST]


@app.get("/api/template")
def api_template():
    return {"schema_version": "1.0", "submission_id": "my-run-01", "model_name": "my-model",
            "run_metadata": {"total_wall_time_ms": 0, "hardware": ""},
            "predictions": [{"video_id": m.video_id, "events": [],
                             "runtime_metadata": {"frames_processed": 0, "chunks_processed": 0,
                                                  "end_to_end_internal_time_ms": 0, "model_runtimes": []}} for m in MANIFEST]}


@app.post("/api/submit")
async def api_submit(user: str = Form("participant"), file: UploadFile = File(...)):
    raw = await file.read()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        return JSONResponse({"accepted": False, "problems": [{"video_id": "-", "field": "file", "message": f"not valid JSON: {e}"}]})
    probs = validate_submission(data, MANIFEST, raw_bytes=len(raw))
    if not is_acceptable(probs):
        return JSONResponse({"accepted": False, "problems": [p.__dict__ for p in probs if p.field != "info"]})
    sub = Submission.model_validate(data)
    st = _load(user)
    changed = []
    for p in sub.predictions:
        d = p.model_dump(exclude_none=True)
        if st["sheet"].get(p.video_id) != d:
            changed.append(p.video_id)
        st["sheet"][p.video_id] = d
    st["runs"] += 1
    res = _score_sheet(st["sheet"])
    st["history"].append({"run": st["runs"], "at": time.strftime("%H:%M:%S"), "file": file.filename, "submission_id": sub.submission_id,
                          "model_name": sub.model_name, "videos_in_file": len(sub.predictions), "changed": changed,
                          "total": res["total"], "marks": res["marks"], "d1": res["d1"], "d2": res["d2"], "d3": res["d3"]})
    st["current"] = res
    _save(st)
    return {"accepted": True, "changed": changed, "result": res, "runs": st["runs"]}


@app.get("/api/state")
def api_state(user: str = "participant"):
    st = _load(user)
    if st["sheet"] and "current" not in st:
        st["current"] = _score_sheet(st["sheet"])
    return st


@app.post("/api/reset")
def api_reset(user: str = Form("participant")):
    p = _state_path(user)
    if p.exists():
        p.unlink()
    return {"ok": True}


@app.get("/api/leaderboard")
def api_leaderboard():
    rows = []
    for p in STATE_DIR.glob("*.json") if STATE_DIR.exists() else []:
        st = json.loads(p.read_text())
        cur = st.get("current")
        if cur:
            last = st["history"][-1] if st["history"] else {}
            rows.append({"user": st["user"], "model": last.get("model_name", ""), **{k: cur[k] for k in ("d1", "d2", "d3", "marks", "rtf", "speed", "reason", "total")}})
    rows.sort(key=lambda r: -r["total"])
    return rows


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML


HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Local Arena · AHC Video Anomaly Detection</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500&display=swap" rel="stylesheet">
<style>
:root{--bg:#f4f6f9;--card:#fff;--line:#e5e9ef;--ink:#0f1720;--ink2:#334155;--mute:#64748b;--soft:#98a2b3;--brand:#1f6fd2;--brand2:#e8f1fc;--alert:#e0452b;--alert2:#fdeeea;--ok:#22a058;--ok2:#e8f6ee;--warn:#d98e04;--warn2:#fdf4e3;--sh:0 1px 2px rgba(15,23,32,.04),0 12px 32px -16px rgba(15,23,32,.18)}
*{box-sizing:border-box}[hidden]{display:none!important}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 Inter,system-ui,sans-serif}
.mono{font-family:"JetBrains Mono",ui-monospace,monospace;font-feature-settings:"tnum"}
header{height:60px;display:flex;align-items:center;gap:14px;padding:0 28px;background:#fff;border-bottom:1px solid var(--line);position:sticky;top:0;z-index:5}
header b{font-size:16px;font-weight:800}header small{color:var(--mute)}header .tag{margin-left:auto;font-size:12px;color:var(--warn);background:var(--warn2);padding:4px 10px;border-radius:999px;font-weight:600}
main{max-width:1280px;margin:0 auto;padding:20px 28px;display:grid;grid-template-columns:360px 1fr;gap:18px;align-items:start}
.card{background:#fff;border:1px solid var(--line);border-radius:16px;box-shadow:var(--sh)}.card h2{margin:0;padding:16px 18px 0;font-size:11.5px;letter-spacing:.9px;text-transform:uppercase;color:var(--mute);font-weight:700}.body{padding:14px 18px 18px}
label{display:block;font-size:12px;font-weight:600;color:var(--ink2);margin:10px 0 6px}input[type=text]{width:100%;padding:10px 12px;border:1px solid var(--line);border-radius:10px;font:inherit}
.drop{margin-top:10px;border:1.5px dashed #cfd7e2;border-radius:12px;padding:22px;text-align:center;color:var(--mute);cursor:pointer;background:#fafbfd}.drop.on{border-color:var(--brand);background:var(--brand2)}.drop b{display:block;color:var(--ink)}
.btn{border:0;border-radius:11px;padding:11px 14px;font:inherit;font-weight:700;cursor:pointer;background:var(--brand);color:#fff;width:100%;margin-top:12px}.btn.ghost{background:#fff;color:var(--ink2);border:1px solid var(--line);font-weight:600}.btn.sm{width:auto;padding:8px 11px;font-size:12.5px;margin-top:0}
.row{display:flex;gap:8px;flex-wrap:wrap}
.big{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}.kpi{background:var(--bg);border-radius:12px;padding:10px 12px}.kpi .k{font-size:10.5px;color:var(--mute);text-transform:uppercase;letter-spacing:.6px;font-weight:600}.kpi .v{font-size:22px;font-weight:800;letter-spacing:-.3px}.kpi .v small{font-size:11px;color:var(--mute);font-weight:500}
.kpi.total{background:var(--brand2)}.kpi.total .v{color:var(--brand)}
table{width:100%;border-collapse:collapse;font-size:12.5px}th{font-size:10.5px;color:var(--mute);text-transform:uppercase;letter-spacing:.6px;text-align:left;padding:8px 8px;border-bottom:1px solid var(--line);font-weight:600}td{padding:7px 8px;border-bottom:1px solid #f0f2f5;vertical-align:top}tr:hover td{background:#fafbfd}
.pill{display:inline-block;font-size:10.5px;font-weight:700;padding:2px 8px;border-radius:999px}.pill.ok{background:var(--ok2);color:#176b3c}.pill.bad{background:var(--alert2);color:var(--alert)}.pill.mid{background:var(--warn2);color:var(--warn)}.pill.gray{background:var(--bg);color:var(--mute)}
.rej{background:var(--alert2);border:1px solid #f3c4bb;border-radius:12px;padding:12px 14px;margin-top:12px}.rej b{color:var(--alert)}
.acc{background:var(--ok2);border:1px solid #bfe5cd;border-radius:12px;padding:12px 14px;margin-top:12px;color:#176b3c}
.note{font-size:11.5px;color:var(--soft);margin-top:8px}.bar{height:6px;background:var(--bg);border-radius:3px;overflow:hidden;margin-top:6px}.bar i{display:block;height:100%;background:var(--brand)}
.delta{font-size:12px;font-weight:700}.delta.up{color:var(--ok)}.delta.down{color:var(--alert)}
.hist td{font-size:12px}.cur{background:var(--brand2)!important}
</style></head><body>
<header><b>Local Arena</b><small>· AHC Video Anomaly Detection · replica of the evaluation site</small><span class="tag">Speed / Reason bonuses and L2–L3 weights are estimates</span></header>
<main>
 <div>
  <div class="card"><h2>Submit predictions</h2><div class="body">
   <label>Participant</label><input type="text" id="user" value="sentinel">
   <div class="drop" id="drop"><b>Drop your predictions .json here</b>or click to choose · max 5 MB<input type="file" id="file" accept=".json" hidden></div>
   <div class="note" id="fname"></div>
   <button class="btn" id="submit">Submit and score</button>
   <div class="row" style="margin-top:10px"><a class="btn ghost sm" href="/api/manifest" download="manifest.json" style="text-decoration:none">manifest.json</a><a class="btn ghost sm" href="/api/template" download="starter_template.json" style="text-decoration:none">starter template</a><button class="btn ghost sm" id="reset">Reset my sheet</button></div>
   <div id="verdict"></div>
   <div class="note">Rules replicated: a rejected file does not use a run · a file only updates the videos it mentions · omitted videos keep their previous answer · never-answered videos count as normal · every accepted upload re-scores the whole sheet and replaces the previous score.</div>
  </div></div>
  <div class="card" style="margin-top:18px"><h2>My submissions</h2><div class="body" id="hist"><div class="note">No uploads yet.</div></div></div>
  <div class="card" style="margin-top:18px"><h2>Leaderboard</h2><div class="body" id="lb"><div class="note">Empty.</div></div></div>
 </div>
 <div>
  <div class="card"><h2>Current answer sheet · <span id="answered"></span></h2><div class="body">
   <div class="big" id="kpis"></div>
   <div class="big" style="margin-top:10px" id="kpis2"></div>
  </div></div>
  <div class="card" style="margin-top:18px"><h2>Per video</h2><div class="body" style="padding-top:6px;overflow:auto"><table id="pv"></table></div></div>
 </div>
</main>
<script>
const $=s=>document.querySelector(s);let fileObj=null,prev=null;
const drop=$('#drop');drop.onclick=()=>$('#file').click();drop.ondragover=e=>{e.preventDefault();drop.classList.add('on')};drop.ondragleave=()=>drop.classList.remove('on');drop.ondrop=e=>{e.preventDefault();drop.classList.remove('on');pick(e.dataTransfer.files[0])};$('#file').onchange=e=>pick(e.target.files[0]);
function pick(f){fileObj=f;$('#fname').textContent=f?`${f.name} · ${(f.size/1024).toFixed(0)} KB`:''}
$('#submit').onclick=async()=>{if(!fileObj)return alert('choose a file');const fd=new FormData();fd.append('user',$('#user').value);fd.append('file',fileObj);const r=await (await fetch('/api/submit',{method:'POST',body:fd})).json();
 if(!r.accepted){$('#verdict').innerHTML=`<div class="rej"><b>Rejected — this did not use a run.</b><table style="margin-top:8px"><tr><th>video</th><th>field</th><th>message</th></tr>${r.problems.map(p=>`<tr><td class="mono">${p.video_id}</td><td class="mono">${p.field}</td><td>${p.message}</td></tr>`).join('')}</table></div>`;return}
 const d=prev!=null?r.result.total-prev:null;$('#verdict').innerHTML=`<div class="acc"><b>Accepted · run ${r.runs}</b> · ${r.changed.length} video(s) changed · new total <b>${r.result.total.toFixed(1)}</b> ${d!=null?`<span class="delta ${d>=0?'up':'down'}">(${d>=0?'+':''}${d.toFixed(1)})</span>`:''}${d!=null&&d<0?'<br><b style="color:#e0452b">This upload lowered your standing — there is no fallback to the earlier score.</b>':''}</div>`;load()};
$('#reset').onclick=async()=>{if(!confirm('Clear this participant\'s sheet and history?'))return;const fd=new FormData();fd.append('user',$('#user').value);await fetch('/api/reset',{method:'POST',body:fd});prev=null;$('#verdict').innerHTML='';load()};
$('#user').onchange=()=>{prev=null;load()};
function pill(v){return v>=0.95?'ok':v>=0.5?'mid':'bad'}
async function load(){const st=await (await fetch('/api/state?user='+encodeURIComponent($('#user').value))).json();const c=st.current;
 if(!c){$('#kpis').innerHTML='';$('#kpis2').innerHTML='<div class="note">Upload a file to see your score.</div>';$('#pv').innerHTML='';$('#answered').textContent='';$('#hist').innerHTML='<div class="note">No uploads yet.</div>'}else{prev=c.total;
 $('#answered').textContent=`videos answered ${c.answered} of ${c.n_videos}`;
 $('#kpis').innerHTML=[['D1 /25',c.d1,25],['D2 /35',c.d2,35],['D3 /40',c.d3,40],['Marks /100',c.marks,100]].map(([k,v,m])=>`<div class="kpi"><div class="k">${k}</div><div class="v">${v.toFixed(1)}</div><div class="bar"><i style="width:${v/m*100}%"></i></div></div>`).join('');
 $('#kpis2').innerHTML=`<div class="kpi"><div class="k">RTF</div><div class="v">${c.rtf==null?'–':c.rtf.toFixed(3)}<small> proc ÷ media</small></div></div><div class="kpi"><div class="k">Speed (est.)</div><div class="v">${c.speed.toFixed(1)}<small> /5</small></div></div><div class="kpi"><div class="k">Reason (est.)</div><div class="v">${c.reason.toFixed(1)}<small> /5</small></div></div><div class="kpi total"><div class="k">Total</div><div class="v">${c.total.toFixed(1)}</div></div>`;
 const fa=c.false_alarms.length;$('#pv').innerHTML=`<tr><th>video</th><th>lvl</th><th>gt</th><th>answered</th><th>pred</th><th>matched</th><th>alert</th><th>match</th><th>timing</th><th>score</th><th>notes</th></tr>`+c.videos.map(v=>`<tr class="${v.false_alarm?'':''}"><td class="mono">${v.video_id}</td><td>L${v.level}</td><td>${v.gt_normal?'<span class="pill gray">normal</span>':v.n_gt+' ev'}</td><td>${v.answered?'yes':'<span class="pill gray">no → normal</span>'}</td><td>${v.n_pred}</td><td>${v.level>1?v.matched+'/'+v.n_gt:'–'}</td><td>${v.level>1&&!v.gt_normal?v.alert.toFixed(1):'–'}</td><td>${v.level>1&&!v.gt_normal?v.match.toFixed(2):'–'}</td><td>${v.level>1&&!v.gt_normal?v.timing.toFixed(2):'–'}</td><td><span class="pill ${pill(v.score)}">${v.score.toFixed(2)}</span></td><td style="color:${v.false_alarm?'var(--alert)':'var(--mute)'}">${v.false_alarm?'FALSE ALARM → 0':(v.notes||'')}</td></tr>`).join('')+`<tr><td colspan="11" class="note">L1 anomaly-acc ${c.l1_anomaly_acc} · class-acc ${c.l1_class_acc} · false alarms on normal L2/L3: <b style="color:${fa?'var(--alert)':'inherit'}">${fa}</b> · fragments: ${c.fragments}</td></tr>`;
 const h=st.history.slice().reverse();$('#hist').innerHTML=h.length?`<table class="hist"><tr><th>run</th><th>time</th><th>file</th><th>videos</th><th>total</th><th></th></tr>${h.map((r,i)=>`<tr class="${i==0?'cur':''}"><td>${r.run}</td><td class="mono">${r.at}</td><td title="${r.submission_id}">${r.file}</td><td>${r.videos_in_file} (${r.changed.length} changed)</td><td class="mono">${r.total.toFixed(1)}</td><td>${i==0?'<span class="pill ok">counts</span>':''}</td></tr>`).join('')}</table>`:'<div class="note">No uploads yet.</div>'}
 const lb=await (await fetch('/api/leaderboard')).json();$('#lb').innerHTML=lb.length?`<table><tr><th>#</th><th>participant</th><th>D1</th><th>D2</th><th>D3</th><th>marks</th><th>RTF</th><th>spd</th><th>rsn</th><th>total</th></tr>${lb.map((r,i)=>`<tr><td>${i+1}</td><td><b>${r.user}</b><div class="note" style="margin:0">${r.model}</div></td><td>${r.d1.toFixed(1)}</td><td>${r.d2.toFixed(1)}</td><td>${r.d3.toFixed(1)}</td><td>${r.marks.toFixed(1)}</td><td class="mono">${r.rtf==null?'–':r.rtf.toFixed(2)}</td><td>${r.speed.toFixed(1)}</td><td>${r.reason.toFixed(1)}</td><td><b>${r.total.toFixed(1)}</b></td></tr>`).join('')}</table>`:'<div class="note">Empty.</div>'}
load();
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True, help="ground_truth.csv to score against (public test set)")
    ap.add_argument("--videos", default=None, help="video directory (for exact durations → RTF); optional")
    ap.add_argument("--videos-csv", default=None)
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    global GT, MANIFEST, DURATIONS
    GT = load_ground_truth(a.gt)
    MANIFEST = manifest_from_gt(GT)
    vdir = Path(a.videos) if a.videos else None
    DURATIONS = _durations(vdir, Path(a.videos_csv) if a.videos_csv else (vdir.parent / "videos.csv" if vdir else None))
    print(f"Local Arena: {len(MANIFEST)} videos  L1={sum(m.level==1 for m in MANIFEST)} L2={sum(m.level==2 for m in MANIFEST)} "
          f"L3={sum(m.level==3 for m in MANIFEST)}  ->  http://{a.host}:{a.port}")
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
