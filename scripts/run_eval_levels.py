#!/usr/bin/env python3
"""
Run SENTINEL on the official evaluation dataset with L1, L2, L3 subfolders.

Usage:
  python scripts/run_eval_levels.py --eval-dir "data/evaluation" --out "submissions/submission_final.json"
"""
import csv
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
import typer
from rich import print as rprint

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sentinel.cli import load_config, _progress
from sentinel.manifest import ManifestEntry
from sentinel.pipeline import Pipeline
from sentinel.validate import validate_file, is_acceptable

app = typer.Typer(add_completion=False)


@app.command()
def main(
    eval_dir: Path = typer.Option(..., help="Path to evaluation folder containing L1, L2, L3"),
    out: Path = typer.Option("submissions/submission_final.json", help="Output submission JSON path"),
    config: Path = typer.Option("configs/default.yaml", help="Sentinel config path"),
    submission_id: str = typer.Option("sentinel-ahc-eval-01", help="Submission ID"),
    mock: bool = typer.Option(False, help="Force mock verifier")
):
    eval_path = Path(eval_dir)
    if not eval_path.exists():
        rprint(f"[red]Error: eval_dir '{eval_dir}' does not exist![/red]")
        raise typer.Exit(1)

    # Check if nested inside a child folder (e.g. "Evaluation - Mirror 5")
    if not (eval_path / "L1").exists() and not (eval_path / "l1").exists():
        for sub in eval_path.iterdir():
            if sub.is_dir() and ((sub / "L1").exists() or (sub / "l1").exists()):
                eval_path = sub
                break

    rprint(f"[bold green]Discovering evaluation dataset in:[/bold green] {eval_path}")

    entries: list[ManifestEntry] = []
    files: dict[str, str] = {}

    for lvl_name, lvl_int in [("L1", 1), ("L2", 2), ("L3", 3)]:
        lvl_dir = eval_path / lvl_name
        if not lvl_dir.exists():
            lvl_dir = eval_path / lvl_name.lower()
        
        if not lvl_dir.exists():
            rprint(f"[yellow]Note: Folder '{lvl_name}' not found in {eval_path}, skipping...[/yellow]")
            continue

        videos_dir = lvl_dir / "videos"
        csv_file = lvl_dir / "videos.csv"

        vid_ids = []
        if csv_file.exists():
            with open(csv_file, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    vid = row.get("video_id") or row.get("id") or row.get("name")
                    fn = row.get("filename") or row.get("file") or row.get("video") or f"{vid}.mp4"
                    if vid:
                        vid_ids.append((str(vid), str(fn)))
        
        if not vid_ids and videos_dir.exists():
            for vpath in sorted(videos_dir.glob("*.mp4")):
                vid_ids.append((vpath.stem, vpath.name))

        rprint(f"  Level {lvl_int} ([bold]{lvl_name}[/bold]): found {len(vid_ids)} videos")

        for vid, fn in vid_ids:
            cand = videos_dir / fn
            if not cand.exists():
                alt = lvl_dir / fn
                if alt.exists():
                    cand = alt
                else:
                    alt2 = videos_dir / f"{vid}.mp4"
                    if alt2.exists():
                        cand = alt2
            
            if cand.exists():
                entries.append(ManifestEntry(video_id=vid, level=lvl_int, file=cand.name))
                files[vid] = str(cand)
            else:
                rprint(f"    [red]Warning: video file for {vid} not found at {cand}[/red]")

    if not entries:
        rprint("[red]Error: No evaluation videos found in L1/L2/L3 directories![/red]")
        raise typer.Exit(1)

    rprint(f"\n[bold green]Total Evaluation Videos Queued:[/bold green] {len(entries)}")
    rprint("Initializing SENTINEL pipeline...")

    cfg = load_config(config)
    pipe = Pipeline(cfg, mock=mock)

    rprint("[bold cyan]Processing videos through SENTINEL Cascade...[/bold cyan]")
    sub = pipe.run(entries, files, submission_id=submission_id, progress=_progress)

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(sub.to_json(), encoding="utf-8")
    pipe.save_trace(out.with_suffix(".trace.json"))

    rprint(f"\n[bold green]Submission successfully written to:[/bold green] {out}")
    rprint(f"   Videos Processed: {len(sub.predictions)}")
    rprint(f"   VLM Wake Ratio:   {sub.run_metadata.vlm_wake_ratio:.1%}")

    probs = validate_file(out, entries)
    ok = is_acceptable(probs)
    rprint(f"   Official Arena Schema Validation: {'[bold green]PASS[/bold green]' if ok else '[bold red]FAIL[/bold red]'}")
    for p in probs:
        rprint(f"     {p}")


if __name__ == "__main__":
    app()
