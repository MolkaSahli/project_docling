from __future__ import annotations

import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pfe_ews.config import Settings  # noqa: E402
from pfe_ews.enterprise_api import (  # noqa: E402
    EnterpriseChatClient,
    extract_json_object,
)
from pfe_ews.io_utils import write_json, write_jsonl  # noqa: E402
from pfe_ews.logging_utils import configure_logging  # noqa: E402
from pfe_ews.queries import RETRIEVAL_TASKS  # noqa: E402
from pfe_ews.retrieval import HybridRetriever  # noqa: E402

# -----------------------------------------------------------------------------
# Configuration de l'evaluation
# -----------------------------------------------------------------------------
COUNTERPARTY: str | None = None
TOP_K = 10
PREVIEW_CHARS = 1800

# 0 = non pertinent
# 1 = partiellement pertinent / utile comme contexte secondaire
# 2 = clairement pertinent / contient une preuve directement utile
RELEVANT_THRESHOLD = 1

JUDGE_SYSTEM_PROMPT = """You are a strict evaluator of a credit-risk retrieval system.
Your task is ONLY to judge whether each retrieved passage is relevant to the retrieval query.
Do not infer facts that are absent from the passage.
Use this scale:
0 = irrelevant: does not help answer/investigate the query.
1 = partially relevant: related topic or useful secondary context, but not direct evidence.
2 = highly relevant: directly addresses the query or contains concrete evidence useful for it.
Return JSON only, with exactly this structure:
{
  "judgments": [
    {"rank": 1, "relevance": 0, "reason": "short reason"}
  ]
}
Return one judgment for every supplied rank, in the same order.
"""


def _retrieve(
    retriever: HybridRetriever,
    method: str,
    query: str,
    counterparty: str,
    top_k: int,
) -> list[dict[str, Any]]:
    if method == "dense":
        return retriever.dense_store.search(
            query,
            top_k=top_k,
            counterparty=counterparty,
        )
    if method == "bm25":
        return retriever.lexical_index.search(
            query,
            top_k=top_k,
            counterparty=counterparty,
        )
    if method == "hybrid":
        return retriever.search(
            query,
            counterparty=counterparty,
            final_top_k=top_k,
        )
    raise ValueError(f"Methode inconnue: {method}")


def _judge_hits(
    client: EnterpriseChatClient,
    *,
    query: str,
    hits: list[dict[str, Any]],
    model: str | None,
) -> list[dict[str, Any]]:
    contexts = []
    for rank, hit in enumerate(hits, start=1):
        text = str(hit.get("text", ""))[:PREVIEW_CHARS]
        contexts.append(
            {
                "rank": rank,
                "source_file": hit.get("source_file"),
                "pages": hit.get("page_numbers", []),
                "text": text,
            }
        )

    user_prompt = (
        "Retrieval query:\n"
        f"{query}\n\n"
        "Retrieved passages:\n"
        f"{json.dumps(contexts, ensure_ascii=False, indent=2)}"
    )

    raw_text, _ = client.complete(
        system_prompt=JUDGE_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        model=model,
    )
    payload = extract_json_object(raw_text)
    judgments = payload.get("judgments")
    if not isinstance(judgments, list):
        raise ValueError("Le juge LLM n'a pas retourne 'judgments' sous forme de liste.")

    by_rank: dict[int, dict[str, Any]] = {}
    for item in judgments:
        if not isinstance(item, dict):
            continue
        try:
            rank = int(item["rank"])
            relevance = int(item["relevance"])
        except (KeyError, TypeError, ValueError):
            continue
        if relevance not in {0, 1, 2}:
            continue
        by_rank[rank] = {
            "rank": rank,
            "relevance": relevance,
            "reason": str(item.get("reason", ""))[:500],
        }

    result = []
    for rank in range(1, len(hits) + 1):
        if rank not in by_rank:
            raise ValueError(f"Jugement manquant pour le rang {rank}.")
        result.append(by_rank[rank])
    return result


def _precision_at(relevances: list[int], k: int) -> float:
    selected = relevances[:k]
    if not selected:
        return 0.0
    return sum(value >= RELEVANT_THRESHOLD for value in selected) / len(selected)


def _strict_precision_at(relevances: list[int], k: int) -> float:
    selected = relevances[:k]
    if not selected:
        return 0.0
    return sum(value == 2 for value in selected) / len(selected)


def _success_at(relevances: list[int], k: int) -> float:
    return float(any(value >= RELEVANT_THRESHOLD for value in relevances[:k]))


def _mrr_at(relevances: list[int], k: int) -> float:
    for rank, value in enumerate(relevances[:k], start=1):
        if value >= RELEVANT_THRESHOLD:
            return 1.0 / rank
    return 0.0


def _ndcg_at(relevances: list[int], k: int) -> float:
    selected = relevances[:k]
    if not selected:
        return 0.0

    def dcg(values: list[int]) -> float:
        return sum(
            ((2**rel) - 1) / math.log2(rank + 1)
            for rank, rel in enumerate(values, start=1)
        )

    actual = dcg(selected)
    ideal = dcg(sorted(selected, reverse=True))
    return actual / ideal if ideal > 0 else 0.0


def _unique_source_ratio(hits: list[dict[str, Any]], k: int) -> float:
    selected = hits[:k]
    if not selected:
        return 0.0
    sources = {str(hit.get("source_file", "")) for hit in selected}
    return len(sources) / len(selected)


def _metrics(
    hits: list[dict[str, Any]],
    judgments: list[dict[str, Any]],
    latency_seconds: float,
) -> dict[str, float]:
    relevances = [int(item["relevance"]) for item in judgments]
    return {
        "precision_at_5": _precision_at(relevances, 5),
        "precision_at_10": _precision_at(relevances, 10),
        "strict_precision_at_5": _strict_precision_at(relevances, 5),
        "strict_precision_at_10": _strict_precision_at(relevances, 10),
        "success_at_5": _success_at(relevances, 5),
        "success_at_10": _success_at(relevances, 10),
        "mrr_at_10": _mrr_at(relevances, 10),
        "judged_ndcg_at_10": _ndcg_at(relevances, 10),
        "mean_relevance_at_10": (
            sum(relevances[:10]) / (2 * len(relevances[:10]))
            if relevances[:10]
            else 0.0
        ),
        "unique_source_ratio_at_10": _unique_source_ratio(hits, 10),
        "latency_seconds": latency_seconds,
    }


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def main() -> None:
    settings = Settings.from_env()
    settings.ensure_directories()
    configure_logging(settings.logs_dir, "05b_evaluate_retrieval")

    retriever = HybridRetriever(settings)
    judge = EnterpriseChatClient(settings)

    try:
        counterparty = COUNTERPARTY or settings.target_counterparty
        if not counterparty:
            if not retriever.counterparties:
                raise ValueError("Aucune contrepartie indexee.")
            counterparty = retriever.counterparties[0]

        judge_model = settings.llm_review_model or settings.llm_model
        methods = ("dense", "bm25", "hybrid")
        detailed_rows: list[dict[str, Any]] = []

        for task in RETRIEVAL_TASKS:
            for method in methods:
                started = time.perf_counter()
                hits = _retrieve(
                    retriever,
                    method,
                    task.query,
                    counterparty,
                    TOP_K,
                )
                latency = time.perf_counter() - started

                judgments = _judge_hits(
                    judge,
                    query=task.query,
                    hits=hits,
                    model=judge_model,
                )
                metrics = _metrics(hits, judgments, latency)

                judged_hits = []
                for rank, (hit, judgment) in enumerate(
                    zip(hits, judgments, strict=True),
                    start=1,
                ):
                    judged_hits.append(
                        {
                            "rank": rank,
                            "chunk_id": hit.get("chunk_id"),
                            "source_file": hit.get("source_file"),
                            "pages": hit.get("page_numbers", []),
                            "dense_score": hit.get("dense_score"),
                            "lexical_score": hit.get("lexical_score"),
                            "rrf_score": hit.get("rrf_score"),
                            "relevance": judgment["relevance"],
                            "reason": judgment["reason"],
                            "text_preview": str(hit.get("text", ""))[:800],
                        }
                    )

                row = {
                    "counterparty": counterparty,
                    "task_id": task.task_id,
                    "task_title": task.title,
                    "query": task.query,
                    "method": method,
                    "metrics": metrics,
                    "hits": judged_hits,
                }
                detailed_rows.append(row)

                print(
                    f"{task.task_id:28s} | {method:6s} | "
                    f"P@5={metrics['precision_at_5']:.2f} | "
                    f"P@10={metrics['precision_at_10']:.2f} | "
                    f"MRR@10={metrics['mrr_at_10']:.2f} | "
                    f"nDCG@10={metrics['judged_ndcg_at_10']:.2f} | "
                    f"lat={metrics['latency_seconds']:.3f}s"
                )

        summary: dict[str, Any] = {
            "counterparty": counterparty,
            "top_k": TOP_K,
            "judge_model": judge_model,
            "warning": (
                "Evaluation reference-free par LLM-as-a-judge. "
                "Ces scores mesurent la pertinence des resultats retournes, "
                "pas le vrai recall du corpus."
            ),
            "methods": {},
        }

        metric_names = [
            "precision_at_5",
            "precision_at_10",
            "strict_precision_at_5",
            "strict_precision_at_10",
            "success_at_5",
            "success_at_10",
            "mrr_at_10",
            "judged_ndcg_at_10",
            "mean_relevance_at_10",
            "unique_source_ratio_at_10",
            "latency_seconds",
        ]

        for method in methods:
            rows = [row for row in detailed_rows if row["method"] == method]
            summary["methods"][method] = {
                name: _mean([float(row["metrics"][name]) for row in rows])
                for name in metric_names
            }

        output_dir = settings.outputs_dir / "retrieval_evaluation"
        output_dir.mkdir(parents=True, exist_ok=True)
        detailed_path = output_dir / "retrieval_evaluation_details.jsonl"
        summary_path = output_dir / "retrieval_evaluation_summary.json"
        write_jsonl(detailed_path, detailed_rows)
        write_json(summary_path, summary)

        print("\n=== MOYENNES PAR METHODE ===")
        for method, values in summary["methods"].items():
            print(
                f"{method.upper():6s} | "
                f"P@5={values['precision_at_5']:.3f} | "
                f"P@10={values['precision_at_10']:.3f} | "
                f"Success@5={values['success_at_5']:.3f} | "
                f"MRR@10={values['mrr_at_10']:.3f} | "
                f"nDCG@10={values['judged_ndcg_at_10']:.3f} | "
                f"Rel@10={values['mean_relevance_at_10']:.3f} | "
                f"Latency={values['latency_seconds']:.3f}s"
            )

        print(f"\nDetails : {detailed_path}")
        print(f"Resume  : {summary_path}")

    finally:
        retriever.close()


if __name__ == "__main__":
    main()
