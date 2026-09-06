"""
Pretraining Job Microservice.

This does not train inside FastAPI.
It launches the training process safely as a subprocess.

Run:
    uvicorn services.pretrain_service.app:app --host 0.0.0.0 --port 8003
"""

import os
import signal
import subprocess
from typing import Dict

from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI(title="Pretraining Service")

JOBS: Dict[int, subprocess.Popen] = {}


class StartTrainRequest(BaseModel):
    config_path: str = "configs/debug_local.yaml"
    nproc_per_node: int = 1


@app.get("/health")
def health():
    return {"status": "ok", "service": "pretrain"}


@app.post("/start")
def start_training(req: StartTrainRequest):
    """
    Input:
        config_path: YAML pretraining config.
        nproc_per_node: number of GPUs.

    Output:
        OS process ID.
    """
    if req.nproc_per_node > 1:
        cmd = [
            "torchrun",
            f"--nproc_per_node={req.nproc_per_node}",
            "-m",
            "src.train.pretrain",
            "--config",
            req.config_path,
        ]
    else:
        cmd = [
            "python",
            "-m",
            "src.train.pretrain",
            "--config",
            req.config_path,
        ]

    process = subprocess.Popen(cmd, cwd=os.path.join(os.path.dirname(__file__), "../../"))
    JOBS[process.pid] = process

    return {
        "status": "started",
        "pid": process.pid,
        "cmd": " ".join(cmd),
    }


@app.get("/status/{pid}")
def status(pid: int):
    """
    Input:
        pid: process ID.

    Output:
        running/completed/unknown.
    """
    process = JOBS.get(pid)

    if process is None:
        return {"pid": pid, "status": "unknown"}

    code = process.poll()

    if code is None:
        return {"pid": pid, "status": "running"}

    return {"pid": pid, "status": "finished", "return_code": code}


@app.post("/stop/{pid}")
def stop(pid: int):
    """
    Input:
        pid: process ID.

    Output:
        Stops training process if known.
    """
    process = JOBS.get(pid)

    if process is None:
        return {"pid": pid, "status": "unknown"}

    os.kill(pid, signal.SIGTERM)

    return {"pid": pid, "status": "stopped"}
