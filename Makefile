# One entry point for humans and CI. `make help` lists targets.
UV ?= uv
RUN ?= $(UV) run --no-sync   # `make setup` (or CI) syncs once; commands never re-resolve
EXTRAS := --extra train --extra serve --extra dev

.PHONY: help setup setup-gpu gpu-check lint format typecheck test test-unit test-integration test-gpu cov \
        synth-demo fracatlas-data fracatlas-train fracatlas-gpu serve docker-serve docker-train clean

help:  ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-18s %s\n", $$1, $$2}'

setup:  ## create .venv from uv.lock (CPU torch) + git hooks
	$(UV) sync --frozen $(EXTRAS)
	$(RUN) pre-commit install

# GPU box: same lockfile, then swap the CPU torch wheels for CUDA builds of the *same* version.
# `RUN` uses --no-sync, so later `make` targets never swap them back. cu126 needs NVIDIA driver
# >= 560 (cu130: >= 580); check `nvidia-smi`. Override: make setup-gpu CUDA=cu130
CUDA ?= cu126
setup-gpu:  ## setup + CUDA torch/torchvision (same versions as uv.lock)
	$(UV) sync --frozen $(EXTRAS)
	TORCH=$$($(RUN) python -c "import torch; print(torch.__version__.split('+')[0])"); \
	TV=$$($(RUN) python -c "import torchvision; print(torchvision.__version__.split('+')[0])"); \
	$(UV) pip install --reinstall-package torch --reinstall-package torchvision \
	    "torch==$$TORCH+$(CUDA)" "torchvision==$$TV+$(CUDA)" \
	    --index-url https://download.pytorch.org/whl/$(CUDA)
	@# PyTorch index only: it hosts the CUDA deps too, and in uv an --extra-index-url (PyPI)
	@# would be searched *first* and has no +cuXXX builds
	$(RUN) xrayml gpu-info --require-cuda

gpu-check:  ## is the GPU visible, and what does each knob (AMP, channels_last, compile) buy here?
	$(RUN) xrayml gpu-info --require-cuda
	$(RUN) python scripts/gpu_benchmark.py --no-compile

lint:  ## ruff lint + format check
	$(RUN) ruff check .
	$(RUN) ruff format --check .

format:  ## auto-fix lint and format
	$(RUN) ruff check --fix .
	$(RUN) ruff format .

typecheck:  ## mypy on the package
	$(RUN) mypy

test-unit:  ## fast tests (< 30 s)
	$(RUN) pytest tests/unit -q -m "not slow"

test-integration:  ## end-to-end on synthetic data (CLI -> bundle -> API)
	$(RUN) pytest tests -q -m "integration or slow"

test-gpu:  ## CUDA tests (skipped without a GPU): AMP, CPU/GPU parity, compile, resume
	$(RUN) pytest tests -q -m gpu

test: lint typecheck  ## everything CI runs on a PR
	$(RUN) pytest -q

cov:  ## coverage report
	$(RUN) pytest -q --cov=xrayml --cov-report=term-missing

# ---------------------------------------------------------------- demo: synthetic, ~5 min on CPU
synth-demo:  ## full release path on synthetic fissures: data -> train -> ops -> bundle -> gate -> report
	$(RUN) xrayml synth --out data/synth --n 800 --size 128
	$(RUN) xrayml train configs/classification_synth.yaml
	$(RUN) xrayml evaluate --checkpoint runs/cls_synth/best.ckpt --manifest data/synth/manifest.parquet --split val --out reports/synth
	$(RUN) xrayml evaluate --checkpoint runs/cls_synth/best.ckpt --manifest data/synth/manifest.parquet --split test --out reports/synth
	$(RUN) xrayml evaluate --checkpoint runs/cls_synth/best.ckpt --manifest data/synth/manifest.parquet --split external_test --out reports/synth
	$(RUN) xrayml select-ops --checkpoint runs/cls_synth/best.ckpt --preds reports/synth/preds_val.parquet --target-sensitivity 0.9 --global-only --version op-synth-1 --out bundle/thresholds.json
	$(RUN) xrayml export --checkpoint runs/cls_synth/best.ckpt --manifest data/synth/manifest.parquet --thresholds bundle/thresholds.json --out bundle
	$(RUN) xrayml gate --preds reports/synth/preds_test.parquet --ops bundle/thresholds.json --min-sensitivity-lcb 0.7 --out reports/synth/gate.json
	$(RUN) xrayml report --bundle bundle --test-metrics reports/synth/metrics_test.json --test-preds reports/synth/preds_test.parquet --gate reports/synth/gate.json --external-metrics reports/synth/metrics_external_test.json --out reports/synth/validation_report.md

# ------------------------------------------------------------- real data: FracAtlas (CC BY 4.0)
fracatlas-data:  ## download (338 MB, md5) -> manifest -> 1024 px cache -> dedup regroup -> quality
	$(RUN) xrayml download-fracatlas --dest data/raw
	$(RUN) xrayml build-manifest --root data/raw/FracAtlas --out data/fracatlas/manifest.parquet
	$(RUN) xrayml prepare --manifest data/fracatlas/manifest.parquet --out data/fracatlas_cache --max-side 1024
	$(RUN) xrayml regroup-duplicates --manifest data/fracatlas_cache/manifest.parquet --max-hamming 2
	$(RUN) xrayml quality --manifest data/fracatlas_cache/manifest.parquet --version fracatlas-v7-split-v2-dedup --dedupe

fracatlas-train:  ## classifier on FracAtlas (GPU recommended; see configs/*_fracatlas*.yaml)
	$(RUN) xrayml train configs/classification_fracatlas.yaml

# ----------------------------------------------- real data on GPU: the full release path
# Effective batch = batch_size x accum_steps x NPROC = 32. Two GPUs: NPROC=2 GPU_OVERRIDES="optim.accum_steps=1";
# four: NPROC=4 GPU_OVERRIDES="optim.accum_steps=1 data.batch_size=8".
NPROC ?= 1
GPU_CFG ?= configs/classification_fracatlas_gpu.yaml
GPU_OVERRIDES ?=
GPU_RUN := runs/cls_fracatlas_gpu
GPU_REP := reports/fracatlas_gpu
GPU_BUNDLE := bundle_fracatlas_gpu
fracatlas-gpu:  ## FracAtlas on GPU: train (torchrun) -> evaluate -> ops -> export -> gate -> report
	$(RUN) xrayml gpu-info --require-cuda
	$(RUN) torchrun --standalone --nproc_per_node=$(NPROC) -m xrayml.cli train $(GPU_CFG) output_dir=$(GPU_RUN) $(GPU_OVERRIDES)
	for s in val test; do $(RUN) xrayml evaluate --checkpoint $(GPU_RUN)/best.ckpt \
	    --manifest data/fracatlas_cache/manifest.parquet --split $$s --out $(GPU_REP) --device cuda; done
	$(RUN) xrayml select-ops --checkpoint $(GPU_RUN)/best.ckpt --preds $(GPU_REP)/preds_val.parquet \
	    --target-sensitivity 0.9 --version op-fracatlas-gpu-1 --out $(GPU_BUNDLE)/thresholds.json
	$(RUN) xrayml export --checkpoint $(GPU_RUN)/best.ckpt --manifest data/fracatlas_cache/manifest.parquet \
	    --thresholds $(GPU_BUNDLE)/thresholds.json --out $(GPU_BUNDLE)
	@# a failed gate is a *result* (see docs/results.md): keep going so the report records it
	-$(RUN) xrayml gate --preds $(GPU_REP)/preds_test.parquet --ops $(GPU_BUNDLE)/thresholds.json --out $(GPU_REP)/gate.json
	$(RUN) xrayml report --bundle $(GPU_BUNDLE) --test-metrics $(GPU_REP)/metrics_test.json \
	    --test-preds $(GPU_REP)/preds_test.parquet --gate $(GPU_REP)/gate.json --out $(GPU_REP)/validation_report.md

docker-train:  ## build the CUDA training image and train the GPU classifier on all visible GPUs
	docker build -f docker/Dockerfile.train -t xrayml-train:dev .
	docker run --rm --gpus all --shm-size=16g \
	    -v $(PWD)/data:/workspace/data:ro -v $(PWD)/runs:/workspace/runs \
	    xrayml-train:dev train $(GPU_CFG) $(GPU_OVERRIDES)

serve:  ## run the API on ./bundle
	$(RUN) xrayml serve --bundle bundle --port 8000

docker-serve:  ## build + run the serving image with ./bundle mounted
	docker build -f docker/Dockerfile.serve --target runtime -t xrayml-serve:dev .
	docker run --rm -p 8000:8000 -v $(PWD)/bundle:/bundle:ro xrayml-serve:dev

clean:
	rm -rf runs reports bundle bundle_fracatlas_gpu .pytest_cache .mypy_cache .ruff_cache
