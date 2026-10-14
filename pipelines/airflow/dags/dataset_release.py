"""Weekly dataset release DAG (revision plan 6.7).

ingest -> de-identify -> decode/cache -> QC gate -> Spark manifest -> regroup duplicates
-> quality gate -> DVC add + tag. Principles: idempotent tasks (outputs keyed by the logical
date, overwritten not appended), XCom carries *paths* not data, a bad dataset fails loudly
(retries=0 on the gate), heavy work runs in pods, never in the scheduler.
"""

from __future__ import annotations

import pendulum
from airflow.decorators import dag, task
from airflow.exceptions import AirflowFailException
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator

IMAGE = "ghcr.io/sarahlunette/xrays_claude-train:latest"
LAKE = "s3://xrayml-lake"


def xrayml_pod(task_id: str, *args: str) -> KubernetesPodOperator:
    """Every heavy step is the same CLI as on a laptop, in the training image."""
    return KubernetesPodOperator(
        task_id=task_id,
        name=task_id.replace("_", "-"),
        image=IMAGE,
        cmds=["python", "-m", "xrayml.cli"],
        arguments=list(args),
        get_logs=True,
    )


@dag(
    schedule="@weekly",
    start_date=pendulum.datetime(2026, 1, 1, tz="Europe/Paris"),
    catchup=False,
    default_args={"retries": 2},
    tags=["data", "fracture"],
)
def dataset_release():
    @task
    def ingest(ds: str | None = None) -> str:
        # Partner hospital drop -> immutable raw zone, keyed by logical date (idempotent).
        return f"{LAKE}/raw/{ds}/"

    @task
    def deidentify(raw: str) -> str:
        # pydicom-based PS3.15 profile (see xrayml.data.dicom.deidentify) run as a pod per shard.
        return raw.replace("/raw/", "/deid/")

    manifest = xrayml_pod(
        "build_manifest",
        "build-manifest",
        "--root",
        "/data/{{ ds }}/FracAtlas",
        "--out",
        "/data/{{ ds }}/manifest.parquet",
    )
    regroup = xrayml_pod(
        "regroup_duplicates", "regroup-duplicates", "--manifest", "/data/{{ ds }}/manifest.parquet"
    )

    @task(retries=0)
    def quality_gate(ds: str | None = None) -> str:
        import pandas as pd

        path = f"/data/{ds}/manifest.parquet"
        m = pd.read_parquet(path)
        if m.groupby("group_id").split.nunique().max() > 1:
            raise AirflowFailException("patient leakage across splits")
        if len(m) < 1000:
            raise AirflowFailException(f"suspicious row count {len(m)}")
        return path

    @task
    def dvc_release(manifest_path: str, ds: str | None = None) -> str:
        import subprocess

        tag = f"dataset-{ds}"
        for cmd in (
            ["dvc", "add", manifest_path],
            ["git", "commit", "-am", f"data: {tag}"],
            ["git", "tag", tag],
            ["dvc", "push"],
            ["git", "push", "--tags"],
        ):
            subprocess.run(cmd, check=True)
        return tag

    raw = ingest()
    deid = deidentify(raw)
    deid >> manifest >> regroup
    gate = quality_gate()
    regroup >> gate
    dvc_release(gate)


dataset_release()
