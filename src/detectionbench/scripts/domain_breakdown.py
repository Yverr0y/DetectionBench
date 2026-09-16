r"""
Domain-stratified evaluation for GWHD.

Uses per-image country/growth-stage metadata from ``global_dict.json`` -- a
per-domain/site manifest compiled for this project's own stratified
evaluation (not a file shipped by the original GWHD release), listing which
test images belong to which contributing institution, country, and growth
stage. Kept alongside the raw download for convenience; not part of this
project's canonical COCO/YOLO conversion, which treats ``domain`` as unused
metadata (see ``gwhd.py``'s docstring).

Reuses the exact same evaluation engines already producing each model's
published aggregate mAP -- Ultralytics' own ``model.val()`` for YOLO,
``supervision.MeanAveragePrecision`` for RF-DETR -- just restricted to a
per-group image subset each time, so the stratified numbers are computed
identically to (and therefore reconcilable against) the numbers already on
the model cards, rather than a second, potentially-diverging metric
implementation reading the same predictions.

Not wired into ``detectionbench-evaluate``: this is bespoke analysis tooling
for one dataset's domain metadata, not a general per-dataset command.

Known data quirk: exactly one GWHD test-set image
(``da...4203__1.png``) has no entry anywhere in ``global_dict.json`` -- it's
the disambiguated second occurrence of a duplicate filename in the original
release (see ``gwhd.py``'s ``_unique_output_name``), which predates and is
external to this project's renaming. Excluded from every breakdown below
rather than guessed at; the exclusion count is reported per run.

Usage:
  python -m detectionbench.scripts.domain_breakdown --group-by country
  python -m detectionbench.scripts.domain_breakdown --group-by development_stage
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import cv2
import torch
import yaml
from rich.table import Table

from detectionbench.scripts.evaluate_rfdetr import load_detection_dataset
from detectionbench.utils.rfdetr import load_rfdetr_model_class
from detectionbench.utils.utils import RichConsoleManager

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENTS_DIR = REPO_ROOT / "experiments" / "gwhd"

GWHD_RAW_DIR = Path("/home/neural_debugger/Downloads/datasets/GWHD/gwhd_2021")
DEFAULT_GLOBAL_DICT = GWHD_RAW_DIR / "global_dict.json"
DEFAULT_YOLO_TEST_IMAGES = GWHD_RAW_DIR / "yolo_dataset" / "images" / "test"
DEFAULT_COCO_DATASET_DIR = GWHD_RAW_DIR / "yolo_dataset_coco"

RFDETR_SLUGS = ["rfdetr-nano", "rfdetr-small", "rfdetr-medium"]
YOLO_SLUGS = ["yolo11x", "yolo26m", "yolo26s", "yolov8m", "yolov8s", "yolov8n"]
NUM_CLASSES = 1  # GWHD is single-class (wheat_head)


def build_group_mapping(
    global_dict_path: Path, group_by: str
) -> tuple[dict[str, str], list[tuple[str, str, str]]]:
    """
    Map each domain-tagged test filename to its group value.

    Returns (mapping, conflicts); conflicts lists (file_name, first_value,
    second_value) for any filename whose owning domains disagree on the
    group value -- not currently expected to fire (GWHD's one duplicate
    filename is owned by two domains that happen to agree on both country
    and growth stage), logged rather than silently resolved if it ever does.
    """
    data = json.loads(global_dict_path.read_text())
    mapping: dict[str, str] = {}
    conflicts: list[tuple[str, str, str]] = []
    for entry in data.values():
        if group_by == "country":
            values = entry.get("country") or []
            value = str(values[0]).strip() if values else "Unknown"
        elif group_by == "development_stage":
            value = str(entry.get("development_stage") or "Unknown").strip()
        else:
            raise ValueError(f"Unsupported group_by '{group_by}'")

        for file_name in entry.get("images_in_test_set", []):
            existing = mapping.get(file_name)
            if existing is not None and existing != value:
                conflicts.append((file_name, existing, value))
                continue
            mapping[file_name] = value
    return mapping, conflicts


def resolve_test_ground_truth_files(coco_dataset_dir: Path) -> set[str]:
    """Return every image file_name in the canonical GWHD test split."""
    annotations_path = coco_dataset_dir / "test" / "_annotations.coco.json"
    data = json.loads(annotations_path.read_text())
    return {image["file_name"] for image in data["images"]}


def evaluate_yolo_per_group(  # noqa: PLR0913
    *,
    slug: str,
    checkpoint_path: Path,
    test_images_dir: Path,
    mapping: dict[str, str],
    groups: list[str],
    device: str,
    scratch_dir: Path,
) -> dict[str, dict[str, float]]:
    """Evaluate one YOLO checkpoint once per group via Ultralytics' val() engine."""
    from ultralytics import YOLO as UltralyticsYOLO

    model = UltralyticsYOLO(str(checkpoint_path))
    results_by_group: dict[str, dict[str, float]] = {}

    for group in groups:
        image_paths = [
            str(test_images_dir / file_name)
            for file_name, g in mapping.items()
            if g == group and (test_images_dir / file_name).exists()
        ]
        if not image_paths:
            continue

        safe_group = group.replace(" ", "_").replace("/", "_")
        list_path = scratch_dir / f"{slug}_{safe_group}.txt"
        list_path.write_text("\n".join(image_paths))

        # train/val both required by Ultralytics' data.yaml schema even
        # though only `test` is actually loaded for split="test" below --
        # pointed at the same list rather than a separate dummy path.
        data_yaml_path = scratch_dir / f"{slug}_{safe_group}.yaml"
        data_yaml_path.write_text(
            yaml.safe_dump(
                {
                    "train": str(list_path),
                    "val": str(list_path),
                    "test": str(list_path),
                    "nc": NUM_CLASSES,
                    "names": {0: "wheat_head"},
                }
            )
        )

        val_results = model.val(
            data=str(data_yaml_path),
            device=device,
            split="test",
            save_json=False,
            project=str(scratch_dir),
            name=f"{slug}_{safe_group}_val",
            exist_ok=True,
            verbose=False,
            plots=False,
        )
        results_by_group[group] = {
            "mAP50": float(val_results.box.map50),
            "mAP50_95": float(val_results.box.map),
            "num_images": len(image_paths),
        }
    return results_by_group


def evaluate_rfdetr_per_group(  # noqa: PLR0913
    *,
    slug: str,
    checkpoint_path: Path,
    dataset_dir: Path,
    split: str,
    mapping: dict[str, str],
    groups: list[str],
    device: str,
) -> tuple[dict[str, dict[str, float]], int]:
    """Evaluate one RF-DETR checkpoint in a single pass, bucketing by group."""
    from supervision.metrics import MeanAveragePrecision

    model_class = load_rfdetr_model_class(slug)
    model = model_class(
        device=device,
        num_classes=NUM_CLASSES,
        pretrain_weights=str(checkpoint_path),
    )
    model.optimize_for_inference()

    dataset, _class_names = load_detection_dataset(dataset_dir, split)

    metrics_by_group: dict[str, MeanAveragePrecision] = {
        group: MeanAveragePrecision() for group in groups
    }
    counts_by_group: dict[str, int] = dict.fromkeys(groups, 0)
    excluded = 0

    for image_path, image, ground_truth in dataset:
        file_name = Path(image_path).name
        group = mapping.get(file_name)
        if group is None or group not in metrics_by_group:
            excluded += 1
            continue
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detections = model.predict(rgb_image, threshold=0.0, include_source_image=False)
        metrics_by_group[group].update(detections, ground_truth)
        counts_by_group[group] += 1

    results: dict[str, dict[str, float]] = {}
    for group in groups:
        if counts_by_group[group] == 0:
            continue
        result = metrics_by_group[group].compute()
        results[group] = {
            "mAP50": float(result.map50),
            "mAP50_95": float(result.map50_95),
            "num_images": counts_by_group[group],
        }
    return results, excluded


def print_summary(
    group_by: str, all_results: dict[str, dict[str, dict[str, float]]]
) -> None:
    """Print a Rich table of every model's per-group mAP@50/mAP@50:95."""
    console = RichConsoleManager.get_console()
    groups = sorted({g for r in all_results.values() for g in r})

    table = Table(title=f"GWHD Per-{group_by.replace('_', ' ').title()} mAP@50")
    table.add_column("Model", style="cyan")
    for group in groups:
        table.add_column(group, justify="right")

    for slug, group_results in all_results.items():
        row = [slug]
        for group in groups:
            value = group_results.get(group, {}).get("mAP50")
            row.append(f"{value:.4f}" if value is not None else "-")
        table.add_row(*row)

    console.print(table)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the domain-breakdown script."""
    parser = argparse.ArgumentParser(
        description="Domain-stratified GWHD evaluation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--group-by",
        required=True,
        choices=["country", "development_stage"],
        help="Metadata field from global_dict.json to stratify by",
    )
    parser.add_argument("--global-dict", default=str(DEFAULT_GLOBAL_DICT))
    parser.add_argument("--yolo-test-images", default=str(DEFAULT_YOLO_TEST_IMAGES))
    parser.add_argument("--coco-dataset-dir", default=str(DEFAULT_COCO_DATASET_DIR))
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--scratch-dir",
        default=None,
        help="Where per-group temp .txt/.yaml files for the YOLO path are "
        "written (default: a system temp dir, not under experiments/)",
    )
    return parser.parse_args()


def main() -> None:
    """Run domain-stratified evaluation across every trained GWHD model."""
    args = parse_args()
    console = RichConsoleManager.get_console()

    mapping, conflicts = build_group_mapping(Path(args.global_dict), args.group_by)
    if conflicts:
        console.print(
            f"[yellow]WARNING: {len(conflicts)} filename(s) had "
            f"conflicting group assignments across domains:[/yellow]"
        )
        for file_name, first, second in conflicts:
            console.print(f"  {file_name}: {first!r} vs {second!r}")

    gt_files = resolve_test_ground_truth_files(Path(args.coco_dataset_dir))
    unmapped = sorted(gt_files - set(mapping))
    if unmapped:
        console.print(
            f"[yellow]{len(unmapped)} ground-truth test image(s) have no "
            f"domain metadata and will be excluded from this breakdown:"
            f"[/yellow] {unmapped}"
        )

    groups = sorted(set(mapping.values()))
    console.print(f"[bold]Groups ({args.group_by}):[/bold] {groups}")

    scratch_base = (
        Path(args.scratch_dir)
        if args.scratch_dir
        else Path(tempfile.gettempdir()) / "detectionbench_domain_breakdown"
    )
    scratch_dir = scratch_base / args.group_by
    scratch_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, dict[str, dict[str, float]]] = {}

    for slug in YOLO_SLUGS:
        nested = EXPERIMENTS_DIR / slug / slug
        checkpoint_path = nested / "weights" / "best.pt"
        if not checkpoint_path.exists():
            console.print(
                f"[yellow]Skipping {slug}: no checkpoint at {checkpoint_path}[/yellow]"
            )
            continue
        console.print(
            f"[bold cyan]Evaluating {slug} per-{args.group_by}...[/bold cyan]"
        )
        results = evaluate_yolo_per_group(
            slug=slug,
            checkpoint_path=checkpoint_path,
            test_images_dir=Path(args.yolo_test_images),
            mapping=mapping,
            groups=groups,
            device=args.device,
            scratch_dir=scratch_dir,
        )
        all_results[slug] = results
        out_path = nested / "evaluation" / f"{args.group_by}_breakdown.json"
        out_path.write_text(json.dumps(results, indent=2))
        console.print(f"  Saved to {out_path}")

    for slug in RFDETR_SLUGS:
        checkpoint_path = EXPERIMENTS_DIR / slug / "checkpoint_best_total.pth"
        if not checkpoint_path.exists():
            console.print(
                f"[yellow]Skipping {slug}: no checkpoint at {checkpoint_path}[/yellow]"
            )
            continue
        console.print(
            f"[bold cyan]Evaluating {slug} per-{args.group_by}...[/bold cyan]"
        )
        results, excluded = evaluate_rfdetr_per_group(
            slug=slug,
            checkpoint_path=checkpoint_path,
            dataset_dir=Path(args.coco_dataset_dir),
            split=args.split,
            mapping=mapping,
            groups=groups,
            device=args.device,
        )
        all_results[slug] = results
        out_dir = EXPERIMENTS_DIR / slug / "evaluation"
        out_path = out_dir / f"{args.group_by}_breakdown.json"
        out_path.write_text(json.dumps(results, indent=2))
        console.print(f"  Saved to {out_path} ({excluded} image(s) excluded)")

    print_summary(args.group_by, all_results)


if __name__ == "__main__":
    main()
