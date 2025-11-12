"""Background scenario job orchestration utilities."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import yaml

ROOT = Path(__file__).resolve().parents[2]
JOBS_DIR = ROOT / "data" / "tmp" / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class ScenarioJobPayload:
    scenario: Dict[str, float]
    rebuild_features: bool
    retrain_models: bool
    rescore_forecasts: bool
    risk_method: str
    members: str
    posenc: str
    lags: int
    member_order: str


def _now() -> str:
    return datetime.utcnow().isoformat()


def submit_job(payload: ScenarioJobPayload, base_risk_config: Path) -> Dict[str, object]:
    job_id = uuid.uuid4().hex
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    job = {
        "id": job_id,
        "status": "pending",
        "created_at": _now(),
        "started_at": None,
        "finished_at": None,
        "current_step": None,
        "progress": 0.0,
        "scenario": payload.scenario,
        "steps": [],
        "error": None,
    }
    _write_job(job_dir, job)

    thread = threading.Thread(
        target=_run_job,
        args=(job_dir, job, payload, base_risk_config),
        daemon=True,
        name=f"scenario-job-{job_id[:8]}",
    )
    thread.start()
    return job


def list_jobs(limit: int = 20) -> List[Dict[str, object]]:
    jobs = []
    for path in sorted(JOBS_DIR.glob("*/job.json"), reverse=True):
        jobs.append(json.loads(path.read_text()))
        if len(jobs) >= limit:
            break
    return jobs


def get_status(job_id: str, tail: int = 40) -> Dict[str, object]:
    job_dir = JOBS_DIR / job_id
    job_path = job_dir / "job.json"
    if not job_path.exists():
        raise FileNotFoundError(f"Unknown job_id {job_id}")
    job = json.loads(job_path.read_text())
    log_path = job_dir / "job.log"
    if log_path.exists() and tail > 0:
        job["log_tail"] = _read_tail(log_path, tail)
    else:
        job["log_tail"] = []
    return job


def _run_job(job_dir: Path, job: Dict[str, object], payload: ScenarioJobPayload, base_risk_config: Path) -> None:
    job_path = job_dir / "job.json"
    log_path = job_dir / "job.log"
    log_path.touch(exist_ok=True)

    steps = _build_steps(payload)
    job["steps"] = [step["label"] for step in steps]
    job["status"] = "running"
    job["started_at"] = _now()
    job["progress"] = 0.0
    _write_job(job_dir, job)

    try:
        risk_config_path = _write_risk_config(job_dir, base_risk_config, payload)
        total = len(steps) + 1  # include risk step
        completed = 0

        for step in steps:
            job["current_step"] = step["label"]
            job["progress"] = completed / total
            _write_job(job_dir, job)
            _run_step(step["label"], step["cmd"], log_path)
            completed += 1

        risk_cmd = _risk_command(payload.risk_method, risk_config_path)
        job["current_step"] = "risk"
        job["progress"] = completed / total
        _write_job(job_dir, job)
        _run_step("risk", risk_cmd, log_path)

        job["status"] = "completed"
        job["finished_at"] = _now()
        job["progress"] = 1.0
        job["current_step"] = None
        _write_job(job_dir, job)
    except Exception as exc:  # pragma: no cover - runtime failures
        job["status"] = "failed"
        job["error"] = str(exc)
        job["finished_at"] = _now()
        job["current_step"] = None
        _write_job(job_dir, job)
        with log_path.open("a", encoding="utf-8") as log_fh:
            log_fh.write(f"\nERROR: {exc}\n")


def _build_steps(payload: ScenarioJobPayload) -> List[Dict[str, object]]:
    py = sys.executable
    steps: List[Dict[str, object]] = []

    if payload.rebuild_features:
        steps.append(
            {
                "label": "features",
                "cmd": [
                    py,
                    "-m",
                    "services.features.build",
                    "--inputs",
                    "data/raw",
                    "--out",
                    "data/proc",
                    "--members",
                    payload.members,
                    "--lags",
                    str(payload.lags),
                    "--posenc",
                    payload.posenc,
                ],
            }
        )

    if payload.retrain_models:
        steps.append(
            {
                "label": "train",
                "cmd": [
                    py,
                    "-m",
                    "services.model.train",
                    "--data",
                    "data/proc",
                    "--out",
                    "models",
                    "--models",
                    "lr,rf,stack",
                    "--targets",
                    "t2m,precip",
                    "--quantiles",
                    "0.9",
                    "--features",
                    f"ensemble_{payload.members}",
                    "--posenc",
                    payload.posenc,
                    "--member_order",
                    payload.member_order,
                ],
            }
        )

    if payload.rescore_forecasts or payload.retrain_models:
        steps.append(
            {
                "label": "score",
                "cmd": [
                    py,
                    "-m",
                    "services.model.score",
                    "--data",
                    "data/proc/test.parquet",
                    "--out",
                    "models/forecasts_latest.parquet",
                    "--models-dir",
                    "models",
                    "--targets",
                    "t2m,precip",
                    "--quantile",
                    "0.9",
                ],
            }
        )

    return steps


def _risk_command(risk_method: str, config_path: Path) -> List[str]:
    py = sys.executable
    return [
        py,
        "-m",
        "services.risk.compute",
        "--portfolio",
        "assets/plants.csv,assets/ports.csv",
        "--forecasts",
        "models/forecasts_latest.parquet",
        "--out",
        "data/risk/latest.parquet",
        "--method",
        risk_method,
        "--risk-config",
        str(config_path),
    ]


def _write_risk_config(job_dir: Path, base_config: Path, payload: ScenarioJobPayload) -> Path:
    base = yaml.safe_load(base_config.read_text()) or {}
    base["risk_method"] = payload.risk_method
    scenario = base.get("scenario", {})
    scenario.update(payload.scenario)
    base["scenario"] = scenario
    out_path = job_dir / "risk_config.yaml"
    out_path.write_text(yaml.safe_dump(base))
    return out_path


def _run_step(label: str, cmd: List[str], log_path: Path) -> None:
    with log_path.open("a", encoding="utf-8") as log_fh:
        log_fh.write(f"\n[{time.strftime('%H:%M:%S')}] Starting {label}: {' '.join(cmd)}\n")
        log_fh.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            log_fh.write(line)
            log_fh.flush()
        ret = proc.wait()
        if ret != 0:
            raise RuntimeError(f"Step {label} failed with exit code {ret}")
        log_fh.write(f"[{time.strftime('%H:%M:%S')}] {label} completed successfully.\n")


def _write_job(job_dir: Path, job: Dict[str, object]) -> None:
    (job_dir / "job.json").write_text(json.dumps(job, indent=2))


def _read_tail(path: Path, lines: int) -> List[str]:
    if lines <= 0:
        return []
    buffer = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh.readlines():
            buffer.append(line.rstrip("\n"))
            if len(buffer) > lines:
                buffer.pop(0)
    return buffer
