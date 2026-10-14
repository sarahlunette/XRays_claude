"""``xrayml`` command line — every pipeline step is one idempotent command.

Each command reads files and writes files (no hidden state), so the same steps run from a
laptop, a DVC stage (``pipelines/dvc.yaml``), an Airflow task or a GitHub Actions job.

    xrayml synth --out data/synth --n 600 --size 128
    xrayml train configs/classification_synth.yaml
    xrayml evaluate --checkpoint runs/cls/best.ckpt --manifest data/synth/manifest.parquet --split val --out reports/
    xrayml select-ops --checkpoint runs/cls/best.ckpt --preds reports/preds_val.parquet --out bundle/thresholds.json
    xrayml export --checkpoint runs/cls/best.ckpt --manifest data/synth/manifest.parquet --thresholds bundle/thresholds.json --out bundle/
    xrayml gate --preds reports/preds_test.parquet --ops bundle/thresholds.json --out reports/gate.json
    xrayml serve --bundle bundle/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def _model_version(checkpoint: str | Path) -> str:
    """Content-addressed model id: same weights -> same version, whatever the file name."""
    import torch

    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ck["ema"] if ck.get("ema") else ck["model"]
    h = hashlib.sha256()
    for k in sorted(state):
        v = state[k]
        if hasattr(v, "numpy"):
            h.update(k.encode())
            h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return f"{ck['config']['task'][:3]}-{h.hexdigest()[:10]}"


# ---------------------------------------------------------------------- data
def cmd_synth(a: argparse.Namespace) -> int:
    from xrayml.data.manifest import assign_splits, save_manifest
    from xrayml.data.quality import datasheet
    from xrayml.data.synthetic import make_synthetic_dataset

    df = make_synthetic_dataset(
        a.out, n_images=a.n, image_size=a.size, seed=a.seed, gap_px=(a.gap_min, a.gap_max)
    )
    df = assign_splits(df, external={"site": [a.external_site]} if a.external_site else None)
    save_manifest(df, Path(a.out) / "manifest.parquet")
    (Path(a.out) / "DATASHEET.md").write_text(datasheet(df, f"synthetic-n{a.n}-s{a.size}-seed{a.seed}"))
    print(df.groupby("split").fractured.agg(["size", "mean"]).to_string())
    return 0


def cmd_download(a: argparse.Namespace) -> int:
    from xrayml.data.fracatlas import download_fracatlas

    print(download_fracatlas(a.dest))
    return 0


def cmd_build_manifest(a: argparse.Namespace) -> int:
    from xrayml.data.fracatlas import build_fracatlas_manifest
    from xrayml.data.manifest import assign_splits, save_manifest

    df = build_fracatlas_manifest(a.root)
    df = assign_splits(df, val_pct=a.val_pct, test_pct=a.test_pct, salt=a.salt)
    # Paths must be relative to the manifest location.
    root, out = Path(a.root).resolve(), Path(a.out).resolve()
    df["path"] = [
        str((root / p).relative_to(out.parent)) if (root / p).is_relative_to(out.parent) else str(root / p)
        for p in df.path
    ]
    save_manifest(df, out)
    print(df.groupby(["split", "body_part"]).fractured.agg(["size", "sum"]).to_string())
    return 0


def cmd_prepare(a: argparse.Namespace) -> int:
    from xrayml.data.manifest import load_manifest, save_manifest
    from xrayml.data.prepare import prepare_cache

    df = prepare_cache(load_manifest(a.manifest), a.out, max_side=a.max_side, workers=a.workers)
    print(save_manifest(df, Path(a.out) / "manifest.parquet"))
    return 0


def cmd_quality(a: argparse.Namespace) -> int:
    from xrayml.data.manifest import assert_no_group_leakage, load_manifest, validate_manifest
    from xrayml.data.quality import cross_split_duplicates, datasheet

    df = load_manifest(a.manifest)
    validate_manifest(df)
    assert_no_group_leakage(df)
    notes = []
    if a.dedupe:
        dups = cross_split_duplicates(df, a.max_hamming)
        notes.append(f"{len(dups)} near-duplicate pairs across splits (pHash Hamming <= {a.max_hamming})")
        if len(dups):
            dups.to_csv(Path(a.manifest).with_name("cross_split_duplicates.csv"), index=False)
    out = Path(a.manifest).with_name("DATASHEET.md")
    out.write_text(datasheet(df, a.version, notes))
    print(out.read_text())
    return 1 if a.dedupe and a.fail_on_duplicates and len(dups) else 0


def cmd_regroup(a: argparse.Namespace) -> int:
    from xrayml.data.manifest import assign_splits, load_manifest, save_manifest
    from xrayml.data.quality import regroup_near_duplicates

    df = load_manifest(a.manifest)
    before = df.group_id.nunique()
    df = regroup_near_duplicates(df, a.max_hamming)
    keep_external = df.split.eq("external_test")
    df = assign_splits(df, val_pct=a.val_pct, test_pct=a.test_pct, salt=a.salt)
    df.loc[keep_external, "split"] = "external_test"
    save_manifest(df, a.out or a.manifest)
    print(f"groups: {before} -> {df.group_id.nunique()} (near-duplicates merged, Hamming <= {a.max_hamming})")
    print(df.groupby("split").fractured.agg(["size", "sum"]).to_string())
    return 0


def cmd_dicom_audit(a: argparse.Namespace) -> int:
    import pandas as pd

    from xrayml.data.dicom import header_record

    rows = [header_record(p) for p in Path(a.folder).rglob("*.dcm")]
    meta = pd.DataFrame(rows)
    meta.to_csv(a.out, index=False)
    if len(meta):
        print(meta.groupby(["Manufacturer", "PhotometricInterpretation"], dropna=False).size().to_string())
    return 0


# ------------------------------------------------------------------ training
def cmd_train(a: argparse.Namespace) -> int:
    from xrayml.config import load_config
    from xrayml.data.manifest import load_manifest
    from xrayml.engine.distributed import cleanup, init_distributed
    from xrayml.engine.trainer import Trainer

    cfg = load_config(a.config, a.overrides)
    rank, world, local_rank = init_distributed()
    df = load_manifest(cfg.data.manifest)
    trainer = Trainer(
        cfg,
        df[df.split == "train"],
        df[df.split == "val"],
        device=a.device,
        ddp_local_rank=local_rank if world > 1 else None,
    )
    if a.resume:
        trainer.resume(a.resume)
    result = trainer.fit()
    if rank == 0:
        print(
            json.dumps(
                {"best": result["best"], "monitor": result["monitor"], "output_dir": str(cfg.output_dir)}
            )
        )
    cleanup()
    return 0


def cmd_evaluate(a: argparse.Namespace) -> int:
    from xrayml.evaluation.evaluate import evaluate_checkpoint

    m = evaluate_checkpoint(a.checkpoint, a.manifest, a.split, a.out, device=a.device, n_boot=a.n_boot)
    print(
        json.dumps({k: v for k, v in m.items() if k not in {"subgroups", "lineage"}}, indent=2, default=str)
    )
    return 0


def cmd_select_ops(a: argparse.Namespace) -> int:
    import pandas as pd
    import torch

    from xrayml.operating_points import select_operating_points

    preds = pd.read_parquet(a.preds)
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    boxes = pd.read_parquet(a.boxes) if a.boxes else None
    ops = select_operating_points(
        preds,
        target_sensitivity=a.target_sensitivity,
        strategy=a.strategy,
        per_body_part=not a.global_only,
        min_positives=a.min_positives,
        version=a.version,
        model_version=_model_version(a.checkpoint),
        dataset_version=ck["lineage"].get("dataset_version", "untracked"),
        tuning_split=str(preds.split.iloc[0]) if "split" in preds else "val",
        box_preds=boxes,
    )
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    ops.save(a.out)
    print(ops.model_dump_json(indent=2))
    return 0


def cmd_export(a: argparse.Namespace) -> int:
    from xrayml.artifacts import ModelCard, OperatingPointSet
    from xrayml.config import TrainConfig
    from xrayml.data.datasets import make_loader
    from xrayml.data.manifest import load_manifest
    from xrayml.engine.checkpoint import load_model_from_checkpoint
    from xrayml.export.onnx_export import check_parity, export_onnx

    model, ck = load_model_from_checkpoint(a.checkpoint)
    cfg = TrainConfig.model_validate(ck["config"])
    if cfg.task == "detection":
        print(
            "detection models are not exported to ONNX here (see export/onnx_export.py docstring)",
            file=sys.stderr,
        )
        return 2
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    export_onnx(model, cfg.data.image_size, out / "model.onnx")
    ops = OperatingPointSet.load(a.thresholds)
    version = _model_version(a.checkpoint)
    if ops.model_version != version:
        print(f"thresholds were tuned for {ops.model_version}, checkpoint is {version}", file=sys.stderr)
        return 1
    if Path(a.thresholds).resolve() != (out / "thresholds.json").resolve():
        ops.save(out / "thresholds.json")
    df = load_manifest(a.manifest)
    loader = make_loader(
        df[df.split.isin(a.parity_split.split(","))],
        cfg.data.model_copy(update={"num_workers": 0}),
        cfg.task,
        train=False,
    )
    parity = check_parity(model, out / "model.onnx", loader, [p.threshold for p in ops.points])
    (out / "parity.json").write_text(json.dumps(parity, indent=2))
    lineage = ck.get("lineage", {})
    ModelCard(
        model_version=version,
        task=cfg.task,
        architecture=f"{cfg.model.name}:{cfg.model.params}",
        image_size=cfg.data.image_size,
        git_sha=lineage.get("git_sha", "unknown"),
        dataset_version=lineage.get("dataset_version", "untracked"),
        dataset_fingerprint=lineage.get("dataset_fingerprint", ""),
        config=ck["config"],
        metrics={"best_val": ck.get("best"), "parity": parity},
    ).save(out / "model_card.json")
    print(json.dumps(parity))
    return 0 if parity["passed"] else 1


def cmd_gate(a: argparse.Namespace) -> int:
    import pandas as pd

    from xrayml.artifacts import OperatingPointSet
    from xrayml.evaluation.release_gate import release_gate

    res = release_gate(
        pd.read_parquet(a.preds),
        OperatingPointSet.load(a.ops),
        pd.read_parquet(a.prod_preds) if a.prod_preds else None,
        OperatingPointSet.load(a.prod_ops) if a.prod_ops else None,
        min_sensitivity_lcb=a.min_sensitivity_lcb,
        min_specificity=a.min_specificity,
        margin=a.margin,
    )
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res.to_dict(), indent=2))
    print(json.dumps(res.to_dict(), indent=2))
    return 0 if res.passed else 1


def cmd_report(a: argparse.Namespace) -> int:
    import pandas as pd

    from xrayml.artifacts import OperatingPointSet
    from xrayml.evaluation.report import validation_report
    from xrayml.operating_points import confirm_on_test

    bundle = Path(a.bundle)
    card = json.loads((bundle / "model_card.json").read_text())
    ops = OperatingPointSet.load(bundle / "thresholds.json")
    metrics = json.loads(Path(a.test_metrics).read_text())
    confirm = confirm_on_test(ops, pd.read_parquet(a.test_preds))
    gate = json.loads(Path(a.gate).read_text()) if a.gate else None
    ext = json.loads(Path(a.external_metrics).read_text()) if a.external_metrics else None
    md = validation_report(card, metrics, ops, confirm, gate, ext)
    Path(a.out).write_text(md)
    print(md)
    return 0


# ---------------------------------------------------------------- operations
def cmd_serve(a: argparse.Namespace) -> int:
    import uvicorn

    from xrayml.serving.app import create_app

    uvicorn.run(create_app(a.bundle), host=a.host, port=a.port)
    return 0


def cmd_drift(a: argparse.Namespace) -> int:
    import pandas as pd

    from xrayml.artifacts import OperatingPointSet
    from xrayml.monitoring.drift import drift_report

    ref = pd.read_parquet(a.reference)
    ops = OperatingPointSet.load(a.ops)
    ref["positive"] = ref.score >= ref.body_part.map(lambda b: ops.threshold_for(b).threshold)
    live = pd.read_json(a.live, lines=True)
    rep = drift_report(ref, live)
    rep.to_csv(a.out, index=False)
    print(rep.to_string(index=False))
    return 1 if a.fail_on_alert and rep.alert.any() else 0


def cmd_label_reports(a: argparse.Namespace) -> int:
    import pandas as pd

    from xrayml.nlp.report_labeler import label_report

    df = pd.read_csv(a.input)
    for f in a.findings.split(","):
        df[f"label_{f}"] = [label_report(t, f) for t in df[a.text_column].fillna("")]
    df.to_csv(a.out, index=False)
    print(df.filter(like="label_").apply(pd.Series.value_counts).fillna(0).astype(int).to_string())
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="xrayml", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("synth", help="generate a synthetic dataset + manifest (CI, demos)")
    s.add_argument("--out", required=True)
    s.add_argument("--n", type=int, default=600)
    s.add_argument("--size", type=int, default=128)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--gap-min", type=float, default=2.0)
    s.add_argument("--gap-max", type=float, default=5.0)
    s.add_argument("--external-site", default="site_c")
    s.set_defaults(fn=cmd_synth)

    s = sub.add_parser("download-fracatlas", help="download + verify FracAtlas (CC BY 4.0)")
    s.add_argument("--dest", default="data/raw")
    s.set_defaults(fn=cmd_download)

    s = sub.add_parser("build-manifest", help="FracAtlas -> manifest.parquet with deterministic splits")
    s.add_argument("--root", default="data/raw/FracAtlas")
    s.add_argument("--out", default="data/fracatlas/manifest.parquet")
    s.add_argument("--val-pct", type=int, default=10)
    s.add_argument("--test-pct", type=int, default=15)
    s.add_argument("--salt", default="split-v1")
    s.set_defaults(fn=cmd_build_manifest)

    s = sub.add_parser("prepare", help="decode once: resized 16-bit PNG cache + rescaled annotations")
    s.add_argument("--manifest", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--max-side", type=int, default=1024)
    s.add_argument("--workers", type=int, default=2)
    s.set_defaults(fn=cmd_prepare)

    s = sub.add_parser("quality", help="schema + leakage checks, near-duplicates, datasheet")
    s.add_argument("--manifest", required=True)
    s.add_argument("--version", default="untracked")
    s.add_argument("--dedupe", action="store_true")
    s.add_argument("--max-hamming", type=int, default=4)
    s.add_argument("--fail-on-duplicates", action="store_true")
    s.set_defaults(fn=cmd_quality)

    s = sub.add_parser("regroup-duplicates", help="merge near-duplicate images into one group, then re-split")
    s.add_argument("--manifest", required=True)
    s.add_argument("--out", help="default: overwrite --manifest")
    s.add_argument("--max-hamming", type=int, default=2)
    s.add_argument("--val-pct", type=int, default=10)
    s.add_argument("--test-pct", type=int, default=15)
    s.add_argument("--salt", default="split-v1")
    s.set_defaults(fn=cmd_regroup)

    s = sub.add_parser("dicom-audit", help="header-only audit of a DICOM folder")
    s.add_argument("folder")
    s.add_argument("--out", default="dicom_audit.csv")
    s.set_defaults(fn=cmd_dicom_audit)

    s = sub.add_parser("train", help="train from a YAML config (+ key=value overrides); torchrun-aware")
    s.add_argument("config")
    s.add_argument("overrides", nargs="*")
    s.add_argument("--resume")
    s.add_argument("--device")
    s.set_defaults(fn=cmd_train)

    s = sub.add_parser("evaluate", help="checkpoint -> predictions, metrics, CIs, subgroups")
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--manifest", required=True)
    s.add_argument("--split", default="test")
    s.add_argument("--out", default="reports")
    s.add_argument("--device", default="cpu")
    s.add_argument("--n-boot", type=int, default=1000)
    s.set_defaults(fn=cmd_evaluate)

    s = sub.add_parser("select-ops", help="choose operating points on the tuning split")
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--preds", required=True)
    s.add_argument("--boxes")
    s.add_argument("--target-sensitivity", type=float, default=0.9)
    s.add_argument("--strategy", choices=["sensitivity", "youden"], default="sensitivity")
    s.add_argument("--global-only", action="store_true")
    s.add_argument("--min-positives", type=int, default=30)
    s.add_argument("--version", default="op-local")
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_select_ops)

    s = sub.add_parser("export", help="ONNX export + parity check + model card -> release bundle")
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--manifest", required=True)
    s.add_argument("--thresholds", required=True)
    s.add_argument("--parity-split", default="test")
    s.add_argument("--out", default="bundle")
    s.set_defaults(fn=cmd_export)

    s = sub.add_parser("gate", help="release gate: acceptance criteria + non-inferiority vs production")
    s.add_argument("--preds", required=True)
    s.add_argument("--ops", required=True)
    s.add_argument("--prod-preds")
    s.add_argument("--prod-ops")
    s.add_argument("--min-sensitivity-lcb", type=float, default=0.80)
    s.add_argument("--min-specificity", type=float, default=0.50)
    s.add_argument("--margin", type=float, default=0.02)
    s.add_argument("--out", default="reports/gate.json")
    s.set_defaults(fn=cmd_gate)

    s = sub.add_parser("report", help="markdown validation report from a bundle + test results")
    s.add_argument("--bundle", required=True)
    s.add_argument("--test-metrics", required=True)
    s.add_argument("--test-preds", required=True)
    s.add_argument("--gate")
    s.add_argument("--external-metrics")
    s.add_argument("--out", default="reports/validation_report.md")
    s.set_defaults(fn=cmd_report)

    s = sub.add_parser("serve", help="run the inference API on a bundle")
    s.add_argument("--bundle", default="bundle")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(fn=cmd_serve)

    s = sub.add_parser("drift", help="compare live prediction logs with the validation reference")
    s.add_argument("--reference", required=True)
    s.add_argument("--ops", required=True)
    s.add_argument("--live", required=True)
    s.add_argument("--out", default="reports/drift.csv")
    s.add_argument("--fail-on-alert", action="store_true")
    s.set_defaults(fn=cmd_drift)

    s = sub.add_parser("label-reports", help="rule-based FR/EN label mining from report text")
    s.add_argument("--input", required=True)
    s.add_argument("--text-column", default="report")
    s.add_argument("--findings", default="fracture,dislocation,effusion")
    s.add_argument("--out", default="reports/labels.csv")
    s.set_defaults(fn=cmd_label_reports)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.fn(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
