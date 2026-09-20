"""Evaluate frozen synthetic QA with per-query evidence and explicit limitations."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from logistics_ai.rag import (
    DEFAULT_CORPUS,
    DEFAULT_DEV,
    DEFAULT_EVAL,
    EMBEDDING_MODEL,
    EMBEDDING_REVISION,
    GENERATION_MODEL,
    GENERATION_REVISION,
    LocalGenerator,
    Retriever,
    answer_question,
    calibrate_abstention,
    load_documents,
    read_jsonl,
    sha256_file,
    validate_questions,
)


def retrieval_metrics(retrieved_ids: list[str], relevant_ids: list[str], ks=(1, 3, 5)) -> dict:
    if not relevant_ids:
        return {**{f"recall@{k}": None for k in ks}, **{f"mrr@{k}": None for k in ks}}
    relevant = set(relevant_ids)
    result = {}
    for k in ks:
        found = retrieved_ids[:k]
        result[f"recall@{k}"] = len(set(found) & relevant) / len(relevant)
        result[f"mrr@{k}"] = next(
            (1.0 / rank for rank, doc_id in enumerate(found, 1) if doc_id in relevant), 0.0
        )
    return result


def mean_or_none(values):
    values = [value for value in values if value is not None]
    return statistics.mean(values) if values else None


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retriever", choices=("bm25", "dense", "hybrid"), default="bm25")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--allow-download", action="store_true", help="Explicitly allow pinned public model downloads."
    )
    parser.add_argument(
        "--generate", action="store_true", help="Run local Qwen on accepted evaluation queries."
    )
    parser.add_argument(
        "--generation-limit", type=int, default=0, help="0 = all accepted evaluation queries."
    )
    parser.add_argument(
        "--k", type=int, default=3, help="Number of evidence chunks for answerability and generation."
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Default: artifacts/rag/<retriever>.")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--dev", type=Path, default=DEFAULT_DEV)
    parser.add_argument("--evaluation", type=Path, default=DEFAULT_EVAL)
    args = parser.parse_args()
    if args.k < 1 or args.generation_limit < 0:
        parser.error("k must be positive and generation-limit nonnegative")
    if args.output_dir is None:
        args.output_dir = ROOT / "artifacts" / "rag" / args.retriever

    documents = load_documents(args.corpus)
    dev, evaluation = read_jsonl(args.dev), read_jsonl(args.evaluation)
    validate_questions(dev, evaluation, documents)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch = None
    runtime = {"python": platform.python_version(), "platform": platform.platform()}
    if args.retriever != "bm25" or args.generate:
        import torch
        import transformers

        torch.manual_seed(42)
        runtime.update({"torch": torch.__version__, "transformers": transformers.__version__})
        if args.device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but unavailable; no implicit CPU fallback.")
            torch.cuda.reset_peak_memory_stats()
            runtime["gpu"] = torch.cuda.get_device_name(0)
            runtime["cuda_runtime"] = torch.version.cuda

    print(f"Building {args.retriever} retriever for {len(documents)} synthetic policies...", flush=True)
    started = time.perf_counter()
    retriever = Retriever(
        documents, mode=args.retriever, device=args.device, local_files_only=not args.allow_download
    )
    index_ms = (time.perf_counter() - started) * 1000
    calibration = calibrate_abstention(retriever, dev, k=args.k)
    write_json(args.output_dir / "calibration.json", calibration)

    generator = None
    if args.generate:
        print("Loading pinned local Qwen generator...", flush=True)
        generator = LocalGenerator(device=args.device, local_files_only=not args.allow_download)

    records = []
    generation_count = 0
    for number, question in enumerate(evaluation, 1):
        before = time.perf_counter()
        hits = retriever.search(question["question"], k=max(5, args.k))
        retrieval_ms = (time.perf_counter() - before) * 1000
        ids = [hit.document.doc_id for hit in hits]
        answer = answer_question(question["question"], hits[: args.k], calibration["threshold"])
        if (
            generator
            and not answer["abstained"]
            and (args.generation_limit == 0 or generation_count < args.generation_limit)
        ):
            answer["model_draft"] = generator.generate(question["question"], answer["evidence"])
            generation_count += 1
        record = {
            **question,
            "retrieved_doc_ids": ids,
            "retrieval_latency_ms": retrieval_ms,
            **retrieval_metrics(ids, question["relevant_doc_ids"]),
            "response": answer,
        }
        records.append(record)
        # Preserve completed outputs even if an eventual GPU call fails.
        write_json(args.output_dir / "per_query.json", records)
        print(f"[{number}/{len(evaluation)}] {question['id']}: abstained={answer['abstained']}", flush=True)

    known = [row for row in records if row["answerable"]]
    unknown = [row for row in records if not row["answerable"]]
    accepted = [row for row in records if not row["response"]["abstained"]]
    drafts = [row["response"]["model_draft"] for row in records if row["response"]["model_draft"] is not None]
    # The conditional hit rate says whether a relevant chunk was retrieved,
    # NOT whether the model's generated claims are correct.
    retrieval_presence = [
        bool(set(row["retrieved_doc_ids"][: args.k]) & set(row["relevant_doc_ids"])) for row in accepted
    ]
    metrics = {
        "run_at_utc": datetime.now(UTC).isoformat(),
        "status": "completed",
        "dataset": "hand-authored synthetic logistics QA v1.0",
        "synthetic_only": True,
        "retriever": args.retriever,
        "device_requested": args.device,
        "device_used": args.device if args.retriever != "bm25" or args.generate else "cpu",
        "embedding": None
        if args.retriever == "bm25"
        else {"model": EMBEDDING_MODEL, "revision": EMBEDDING_REVISION},
        "generation": {
            "requested": args.generate,
            "model": GENERATION_MODEL if args.generate else None,
            "revision": GENERATION_REVISION if args.generate else None,
            "completed_queries": len(drafts),
            "limit": args.generation_limit,
            "human_factuality_review": "not_performed",
        },
        "runtime": runtime,
        "document_count": len(documents),
        "evaluation_questions": len(records),
        "answerable_questions": len(known),
        "unanswerable_questions": len(unknown),
        "evidence_k": args.k,
        "input_sha256": {
            "corpus": sha256_file(args.corpus),
            "dev": sha256_file(args.dev),
            "eval": sha256_file(args.evaluation),
        },
        "calibration_threshold": calibration["threshold"],
        "retrieval": {
            metric: mean_or_none(row[metric] for row in known)
            for metric in ("recall@1", "recall@3", "recall@5", "mrr@1", "mrr@3", "mrr@5")
        },
        "abstention": {
            "known_query_acceptance_rate": mean_or_none(not row["response"]["abstained"] for row in known),
            "unknown_query_refusal_rate": mean_or_none(row["response"]["abstained"] for row in unknown),
            "unknown_false_accept_count": sum(not row["response"]["abstained"] for row in unknown),
            "known_false_refusal_count": sum(row["response"]["abstained"] for row in known),
            "overall_answer_coverage": len(accepted) / len(records),
            "accepted_query_relevant_chunk_present_rate": mean_or_none(retrieval_presence),
        },
        "latency_ms": {
            "index_build": index_ms,
            "retrieval_p50": percentile([row["retrieval_latency_ms"] for row in records], 0.5),
            "retrieval_p95": percentile([row["retrieval_latency_ms"] for row in records], 0.95),
            "generation_p50": percentile([draft["latency_ms"] for draft in drafts], 0.5),
        },
        "generation_citation_syntax_pass_rate": mean_or_none(
            draft["citation_check"]["valid_syntax_only"] for draft in drafts
        ),
        "limitations": [
            "Tiny hand-authored synthetic corpus and QA by the same author; not production or independent validation.",
            "Development-only threshold is a lexical heuristic; unsupported in-domain questions can still pass.",
            "Retrieval recall, presence of evidence and citation syntax do not establish answer factuality.",
            "No LLM-as-judge or human factuality score is claimed. Inspect per_query.json model drafts manually.",
            "Input hashes identify the exact frozen data used; changing data invalidates direct metric comparisons.",
        ],
    }
    if torch is not None and args.device == "cuda":
        metrics["cuda_peak_allocated_mb"] = torch.cuda.max_memory_allocated() / (1024**2)
    write_json(args.output_dir / "metrics.json", metrics)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "retrieval": metrics["retrieval"],
                "abstention": metrics["abstention"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
