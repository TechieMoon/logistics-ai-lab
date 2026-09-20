"""Local-only portfolio API. Start from repository root with scripts/serve.ps1."""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
INFERENCE_LOCK = threading.Lock()
app = FastAPI(
    title="Logistics AI Lab",
    version="0.1.0",
    description=(
        "합성 물류 데이터로 실행하는 경로 최적화 · 한국어 RAG 실험. "
        "실제 기업의 정책이나 운영 데이터가 아닙니다. 각 POST 항목의 Try it out으로 실행하세요. "
        "모델 다운로드는 평가 스크립트에서 명시적으로 실행합니다."
    ),
)


class RouteRequest(BaseModel):
    seed: int = Field(default=20260920, ge=0, le=2**32 - 1)
    customers: int = Field(default=15, ge=2, le=50)
    capacity: float = Field(default=30, ge=1, le=1000)
    algorithm: Literal["nearest_neighbor", "two_opt", "learned", "learned_two_opt"] = "two_opt"


class QuestionRequest(BaseModel):
    question: str = Field(default="파손 화물을 접수할 때 사진을 몇 장 남기나요?", min_length=2, max_length=1000)
    mode: Literal["bm25", "dense", "hybrid"] = "bm25"
    generate: bool = False
    device: Literal["cpu", "cuda"] = "cpu"


@app.get("/", include_in_schema=False)
def home():
    return RedirectResponse("/docs")


@app.get("/health", tags=["상태"])
def health():
    import torch
    return {
        "status": "ok", "synthetic_data": True,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "routing_checkpoint": (ROOT / "artifacts/routing/checkpoint.pt").is_file(),
    }


@app.post("/route", tags=["경로 최적화"])
def route(request: RouteRequest):
    from logistics_ai.routing import generate_instance, nearest_neighbor, solve_learned, two_opt
    instance = generate_instance(request.seed, request.customers, request.capacity)
    baseline = nearest_neighbor(instance)
    if request.algorithm.startswith("learned"):
        checkpoint = ROOT / "artifacts/routing/checkpoint.pt"
        if not checkpoint.is_file():
            raise HTTPException(503, "먼저 scripts/train_routing.py로 체크포인트를 생성하세요.")
        with INFERENCE_LOCK:
            solution = solve_learned(instance, checkpoint, device="cpu")
        if request.algorithm == "learned_two_opt":
            solution = two_opt(instance, solution)
    elif request.algorithm == "two_opt":
        solution = two_opt(instance, baseline)
    else:
        solution = baseline
    return {
        "synthetic_data": True, "distance_unit": "unit-square Euclidean distance, not km",
        "instance": instance.to_dict(), "solution": solution.to_dict(),
        "nearest_neighbor_distance": baseline.distance,
        "improvement_vs_nearest_percent": 100 * (baseline.distance - solution.distance) / baseline.distance,
    }


@app.post("/ask", tags=["정책 RAG"])
def ask(request: QuestionRequest):
    from logistics_ai.rag import build_answer
    try:
        # Serialise local model loading/generation to bound GPU memory usage.
        with INFERENCE_LOCK:
            return build_answer(request.question, mode=request.mode, device=request.device,
                                generate=request.generate, local_files_only=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise HTTPException(503, "모델/설정을 확인하세요. 먼저 evaluate_rag.py --allow-download를 "
                            f"실행할 수 있습니다. 오류 종류: {type(exc).__name__}") from exc


@app.get("/results", tags=["실험 결과"])
def results():
    paths = [ROOT / "artifacts/routing/metrics.json", *sorted((ROOT / "artifacts/rag").glob("*/metrics.json"))]
    return {str(path.relative_to(ROOT)).replace("\\", "/"): json.loads(path.read_text(encoding="utf-8"))
            for path in paths if path.is_file()}
