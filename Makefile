# One entry point for humans and CI. `make help` lists targets.
UV ?= uv
RUN ?= $(UV) run --no-sync   # `make setup` (or CI) syncs once; commands never re-resolve
EXTRAS := --extra train --extra serve --extra dev

.PHONY: help setup lint format typecheck test test-unit test-integration cov synth-demo \
        fracatlas-data fracatlas-train serve docker-serve clean

help:  ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-18s %s\n", $$1, $$2}'

setup:  ## create .venv from uv.lock (CPU torch) + git hooks
	$(UV) sync --frozen $(EXTRAS)
	$(RUN) pre-commit install

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

serve:  ## run the API on ./bundle
	$(RUN) xrayml serve --bundle bundle --port 8000

docker-serve:  ## build + run the serving image with ./bundle mounted
	docker build -f docker/Dockerfile.serve --target runtime -t xrayml-serve:dev .
	docker run --rm -p 8000:8000 -v $(PWD)/bundle:/bundle:ro xrayml-serve:dev

clean:
	rm -rf runs reports bundle .pytest_cache .mypy_cache .ruff_cache
