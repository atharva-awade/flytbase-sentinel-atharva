"""SENTINEL command line.  `python -m sentinel.cli --help`

Typical day:
  sentinel run        --manifest data/manifest.json --videos data/private --levels 1 --out submissions/L1.json
  sentinel validate   submissions/L1.json --manifest data/manifest.json
  sentinel score      submissions/public.json --gt data/test/ground_truth.csv
  sentinel submit-check submissions/L2.json --manifest data/manifest.json --public submissions/public.json
  sentinel sweep      --gt data/test/ground_truth.csv --videos data/test/videos
  sentinel merge      submissions/L1.json submissions/L2.json submissions/L3.json --out submissions/final.json
"""
from __future__ import annotations

import copy
import itertools
import json
import shutil
import time
from pathlib import Path

import numpy as np
import typer
import yaml
from rich import print as rprint
from rich.table import Table

from . import CLASSES
from .manifest import gt_by_video, load_ground_truth, load_manifest, load_videos_csv, manifest_from_gt
from .pipeline import Pipeline, load_config
from .schema import ManifestEntry, Prediction, Submission
from .scorer import score as score_fn
from .validate import is_acceptable, validate_file

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)


def _files_for(entries: list[ManifestEntry], videos_dir: Path, videos_csv: Path | None) -> dict[str, str]:
    mapping = load_videos_csv(videos_csv) if videos_csv and videos_csv.exists() else {}
    out = {}
    for e in entries:
        cand = []
        if e.file:
            cand.append(videos_dir / e.file)
        if e.video_id in mapping:
            cand.append(videos_dir / mapping[e.video_id])
        cand += [videos_dir / f"{e.video_id}{ext}" for ext in (".mp4", ".MP4", ".mov", ".avi", ".mkv")]
        for c in cand:
            if c.exists():
                out[e.video_id] = str(c); break
        else:
            rprint(f"[yellow]no file for {e.video_id}[/yellow]")
    return out


def _load_entries(manifest: Path | None, gt: Path | None) -> list[ManifestEntry]:
    if manifest:
        return load_manifest(manifest)
    if gt:
        return manifest_from_gt(load_ground_truth(gt))
    raise typer.BadParameter("need --manifest or --gt")


def _progress(e, p, tr):
    evs = ", ".join(f"{x.class_name}[{x.start_time_sec}-{x.end_time_sec}]" if x.start_time_sec is not None else x.class_name
                    for x in p.events) or "—"
    rprint(f"L{e.level} [bold]{e.video_id}[/bold] {tr['duration_sec']:.0f}s wake={tr['wake_ratio']:.0%} "
           f"rt={tr['runtime_ms']/1000:.1f}s  → {evs}")


@app.command()
def run(manifest: Path = typer.Option(None), gt: Path = typer.Option(None), videos: Path = typer.Option(..., help="directory of videos"),
        videos_csv: Path = typer.Option(None), levels: str = typer.Option("1,2,3"), out: Path = typer.Option(...),
        config: Path = typer.Option("configs/default.yaml"), mock: bool = typer.Option(False, help="CPU mock gate+VLM"),
        submission_id: str = typer.Option(None), only: str = typer.Option(None, help="comma list of video_ids")):
    """Run the cascade and write a submission JSON (+ .trace.json for the dashboard)."""
    cfg = load_config(config)
    entries = _load_entries(manifest, gt)
    if only:
        keep = set(only.split(",")); entries = [e for e in entries if e.video_id in keep]
    files = _files_for(entries, videos, videos_csv or (videos.parent / "videos.csv"))
    pipe = Pipeline(cfg, mock=mock)
    sub = pipe.run(entries, files, levels={int(x) for x in levels.split(",")}, submission_id=submission_id, progress=_progress)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(sub.to_json())
    pipe.save_trace(out.with_suffix(".trace.json"))
    probs = validate_file(out, entries)
    ok = is_acceptable(probs)
    rprint(f"\nwrote [bold]{out}[/bold]  videos={len(sub.predictions)}  wake_ratio={sub.run_metadata.vlm_wake_ratio}  "
           f"valid={'[green]yes' if ok else '[red]NO'}[/]")
    for p in probs:
        rprint(f"  {p}")
    if gt:
        rep = score_fn(sub.predictions, load_ground_truth(gt), {v: t["duration_sec"] for v, t in pipe.trace.items()})
        rprint(rep.summary())
        out.with_suffix(".score.txt").write_text(rep.summary())


@app.command()
def validate(file: Path, manifest: Path = typer.Option(None), gt: Path = typer.Option(None)):
    """Arena-style validation table. Exit code 1 if the arena would reject."""
    entries = _load_entries(manifest, gt) if (manifest or gt) else None
    probs = validate_file(file, entries)
    t = Table("video", "field", "message")
    for p in probs:
        t.add_row(p.video_id, p.field, p.message)
    rprint(t if probs else "[green]no problems[/green]")
    raise typer.Exit(0 if is_acceptable(probs) else 1)


@app.command()
def score(file: Path, gt: Path = typer.Option(...), durations: Path = typer.Option(None, help="json {video_id: sec}")):
    """Local replica scoring against a ground_truth.csv."""
    sub = Submission.model_validate_json(file.read_text())
    dur = json.loads(durations.read_text()) if durations else None
    rep = score_fn(sub.predictions, load_ground_truth(gt), dur)
    rprint(rep.summary())


@app.command("submit-check")
def submit_check(file: Path, manifest: Path = typer.Option(...), public: Path = typer.Option(None, help="public-set submission from the same config"),
                 public_gt: Path = typer.Option("data/test/ground_truth.csv"), last: Path = typer.Option("submissions/LAST_UPLOADED.json"),
                 min_delta: float = typer.Option(-0.5)):
    """The upload gate (PLAN.md §8.3): validate, compare to last upload, refuse regressions/false alarms/fragments."""
    entries = load_manifest(manifest)
    probs = validate_file(file, entries)
    if not is_acceptable(probs):
        for p in probs: rprint(f"[red]{p}[/red]")
        raise typer.Exit(1)
    sub = Submission.model_validate_json(file.read_text())
    frag = sum(1 for p in sub.predictions for a, b in itertools.combinations(p.events, 2)
               if a.class_name == b.class_name and a.start_time_sec is not None and b.start_time_sec is not None
               and min(a.end_time_sec, b.end_time_sec) - max(a.start_time_sec, b.start_time_sec) > -3)
    if frag:
        rprint(f"[red]{frag} same-class near-overlapping interval pairs (fragments) — merge them first[/red]"); raise typer.Exit(1)
    if public and Path(public).exists() and Path(public_gt).exists():
        rep = score_fn(Submission.model_validate_json(Path(public).read_text()).predictions, load_ground_truth(public_gt))
        fa = [v.video_id for v in rep.videos if v.false_alarm]
        rprint(rep.summary())
        if fa:
            rprint(f"[red]false alarms on public normal videos {fa} — fix τ_video before uploading[/red]"); raise typer.Exit(1)
        best_f = Path("submissions/BEST_PUBLIC_SCORE.txt")
        best = float(best_f.read_text()) if best_f.exists() else -1e9
        if rep.total_points < best + min_delta:
            rprint(f"[red]public score {rep.total_points:.1f} < best {best:.1f}{min_delta:+.1f} — refusing[/red]"); raise typer.Exit(1)
        best_f.write_text(str(max(best, rep.total_points)))
    if Path(last).exists():
        prev = {p.video_id: p for p in Submission.model_validate_json(Path(last).read_text()).predictions}
        changed = [p.video_id for p in sub.predictions if p.video_id not in prev or
                   [e.model_dump(include={"class_name", "start_time_sec", "end_time_sec"}) for e in p.events] !=
                   [e.model_dump(include={"class_name", "start_time_sec", "end_time_sec"}) for e in prev[p.video_id].events]]
        rprint(f"changes vs last upload: {len(changed)} videos {changed}")
    rprint("[green]READY — upload this file, then run:  sentinel mark-uploaded " + str(file) + "[/green]")


@app.command("mark-uploaded")
def mark_uploaded(file: Path, last: Path = typer.Option("submissions/LAST_UPLOADED.json")):
    """Record what the arena currently holds (merging partial uploads)."""
    new = Submission.model_validate_json(file.read_text())
    if Path(last).exists():
        cur = {p.video_id: p for p in Submission.model_validate_json(Path(last).read_text()).predictions}
    else:
        cur = {}
    for p in new.predictions:
        cur[p.video_id] = p
    new.predictions = list(cur.values())
    Path(last).write_text(new.to_json())
    shutil.copy(file, Path("submissions") / f"uploaded_{time.strftime('%H%M%S')}_{file.name}")
    rprint(f"arena sheet now has {len(cur)} videos")


@app.command()
def merge(files: list[Path], out: Path = typer.Option(...), submission_id: str = typer.Option("sentinel-final")):
    """Compose one sheet from several (later files override earlier ones per video)."""
    cur: dict[str, Prediction] = {}
    base = None
    for f in files:
        s = Submission.model_validate_json(f.read_text()); base = base or s
        for p in s.predictions:
            cur[p.video_id] = p
    base.predictions = list(cur.values()); base.submission_id = submission_id
    out.write_text(base.to_json()); rprint(f"wrote {out} with {len(cur)} videos")


@app.command()
def sweep(gt: Path = typer.Option(...), videos: Path = typer.Option(...), config: Path = typer.Option("configs/default.yaml"),
          mock: bool = typer.Option(False), out: Path = typer.Option("configs/class_priors.sweep.yaml"),
          trace_from: Path = typer.Option(None, help="reuse a run's .trace.json (skips GPU work: decoder-only sweep)")):
    """Grid-search decoder thresholds on the public set. Decoder-only when --trace-from is given (seconds, not minutes)."""
    from .decoder import decode, decode_l1, load_priors
    from .emit import to_prediction
    from .schema import RuntimeMetadata
    gts = load_ground_truth(gt)
    entries = manifest_from_gt(gts)
    if trace_from is None:
        cfg = load_config(config)
        pipe = Pipeline(cfg, mock=mock)
        pipe.run(entries, _files_for(entries, videos, videos.parent / "videos.csv"), progress=_progress)
        trace = pipe.trace
    else:
        trace = json.loads(trace_from.read_text())
    lvl = {e.video_id: e.level for e in entries}
    priors0 = load_priors()
    rt = RuntimeMetadata(frames_processed=1, chunks_processed=1, end_to_end_internal_time_ms=1)
    grid = itertools.product([0.5, 0.6, 0.7], [0.3, 0.4], [0.55, 0.65, 0.75], [0.5, 0.6])  # tau_on, tau_off, tau_video, tau_l1
    results = []
    for tau_on, tau_off, tau_video, tau_l1 in grid:
        P = copy.deepcopy(priors0); P["defaults"].update(tau_on=tau_on, tau_off=tau_off, tau_video=tau_video, tau_l1=tau_l1)
        preds = []
        for vid, tr in trace.items():
            curves = {c: np.asarray(tr["curves"].get(c, np.zeros(tr["T"]))) for c in CLASSES}
            if lvl[vid] == 1:
                preds.append(to_prediction(vid, 1, None, decode_l1(curves, P), rt))
            else:
                preds.append(to_prediction(vid, lvl[vid], decode(curves, lvl[vid], P, tr["duration_sec"]), None, rt))
        rep = score_fn(preds, gts)
        fa = sum(v.false_alarm for v in rep.videos)
        results.append((rep.total_points - 5 * fa, rep.total_points, fa, tau_on, tau_off, tau_video, tau_l1))
    results.sort(reverse=True)
    t = Table("objective", "points", "false alarms", "tau_on", "tau_off", "tau_video", "tau_l1")
    for r in results[:10]:
        t.add_row(*[f"{x:.2f}" if isinstance(x, float) else str(x) for x in r])
    rprint(t)
    best = results[0]
    P = copy.deepcopy(priors0); P["defaults"].update(tau_on=best[3], tau_off=best[4], tau_video=best[5], tau_l1=best[6])
    out.write_text(yaml.safe_dump(P, sort_keys=False)); rprint(f"best priors → {out} (copy over configs/class_priors.yaml to adopt)")


@app.command("derive-priors")
def derive_priors(train_dir: Path = typer.Option("data/train"), out: Path = typer.Option("configs/class_priors.yaml")):
    """Fill median_dur_sec per class from train/*/ground_truth.csv."""
    from .decoder import load_priors
    P = load_priors(out)
    durs: dict[str, list[float]] = {}
    for csv in train_dir.glob("*/ground_truth.csv"):
        for e in load_ground_truth(csv):
            if e.is_anomaly and e.start_time_sec is not None and e.end_time_sec is not None:
                durs.setdefault(e.class_name, []).append(e.end_time_sec - e.start_time_sec)
    for c, xs in durs.items():
        if c in P["classes"]:
            P["classes"][c]["median_dur_sec"] = float(round(np.median(xs), 1))
            rprint(f"{c:<34} n={len(xs):<4} median={np.median(xs):.1f}s  p25={np.percentile(xs,25):.1f} p75={np.percentile(xs,75):.1f}")
    out.write_text(yaml.safe_dump(P, sort_keys=False))


@app.command("build-normal-bank")
def build_normal_bank(train_dir: Path = typer.Option("data/train"), out: Path = typer.Option("banks/normal.npy"),
                      per_video: int = typer.Option(8), max_videos: int = typer.Option(400), config: Path = typer.Option("configs/default.yaml")):
    """Embed frames from train/normal (+ non-event stretches) with SigLIP2 → normality bank."""
    from .gate import build_gate
    from .ingest import load_clip
    cfg = load_config(config)
    g = build_gate(cfg["gate"], cfg["paths"]["prompt_bank"])
    vids = sorted((train_dir / "normal" / "videos").glob("*.mp4"))[:max_videos]
    E = []
    for i, v in enumerate(vids):
        clip = load_clip(v.stem, str(v), sample_fps=0.5, max_side=cfg["ingest"]["max_side"], max_frames=per_video)
        e = g.embed_images(clip.frames, None); E.append(e / np.linalg.norm(e, axis=1, keepdims=True))
        if i % 25 == 0: rprint(f"{i}/{len(vids)}")
    bank = np.concatenate(E).astype(np.float16)
    out.parent.mkdir(exist_ok=True, parents=True); np.save(out, bank); rprint(f"saved {bank.shape} → {out}")


if __name__ == "__main__":
    app()
