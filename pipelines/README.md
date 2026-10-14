# Data-platform reference code (revision plan chapter 6)

These files show how the same `xrayml` steps plug into the team's stack. They are linted but
not executed in CI (no Airflow / Spark / Mongo in the runner). The executable source of truth
is the CLI; every orchestrator below only *calls* it, so behaviour is identical everywhere.

| File | Role |
|---|---|
| `../dvc.yaml`, `../params.yaml` | Versioned pipeline: data → train → evaluate → ops → export → gate. `dvc.lock` = lineage proof |
| `airflow/dags/dataset_release.py` | Weekly DAG: ingest → de-identify → QC → manifest (Spark) → quality gate → DVC tag |
| `spark/build_manifest.py` | Spark job: filter, dedupe, broadcast join, *the same* split function as `xrayml.data.manifest` |
| `sql/analytics.sql` | Window functions, anti-joins, the "share of studies with a positive label per site" query |
| `mongo/annotations.js` | Document schema, aggregation pipeline, ESR compound index |
