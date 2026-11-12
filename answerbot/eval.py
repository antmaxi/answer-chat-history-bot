"""Measure retrieval using fixture text or annotated Telegram message IDs.

    python -m answerbot.eval              # against DB_PATH
    python -m answerbot.eval --fixture    # load + index the fixture, then eval
    python -m answerbot.eval --cases data/search-benchmark.json
    python -m answerbot.eval --misses     # logged questions that abstained or cited nothing
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from math import ceil
from pathlib import Path
from time import perf_counter

from . import db, index, logconfig, retrieve


@dataclass(frozen=True)
class Case:
    question: str
    needles: tuple[str, ...] = ()
    # True if BM25 alone should retrieve the needles (no embed model needed).
    keyword_ok: bool = True
    evidence: tuple[tuple[int, int], ...] = ()
    chat_ids: tuple[int, ...] | None = None
    now: datetime | None = None


# Distinctive terms from the fixture conversations. Paraphrase-only cases are
# marked keyword_ok=False — they need real vectors and are for the CLI.
CASES: tuple[Case, ...] = (
    Case("how much was the ski trip", ("200", "lari")),
    Case("wifi password", ("HappyMonday2024",)),
    Case("why was standup moved", ("10:30",)),
    Case("what database did we decide on", ("postgres",)),
    Case("staging certificate error", ("letsencrypt",)),
    Case("dentist in Tbilisi", ("Kvaratskhelia",)),
    Case("budget for the offsite", ("5000",)),
    Case("when is the ski trip", ("Feb 14-17",)),
    Case("what did Nino say about standup", ("10:30",)),
    Case("why was the morning meeting moved", ("10:30",), keyword_ok=False),
)


@dataclass
class CaseResult:
    case: Case
    hit: bool
    window_ids: list[int]
    candidate_recall: float | None = None
    evidence_recall: float | None = None
    first_relevant_rank: int | None = None
    excerpt_precision: float | None = None
    search_ms: float = 0.0


def load_cases(path: Path) -> tuple[Case, ...]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("expected a non-empty JSON list of benchmark cases")
    cases = []
    for i, item in enumerate(raw, 1):
        try:
            question = item["question"].strip()
            evidence = tuple((int(chat), int(msg)) for chat, msg in item["evidence"])
            scope = item.get("chat_ids")
            now = item.get("now")
            if not question or not evidence:
                raise ValueError("question and evidence must be non-empty")
            cases.append(Case(
                question,
                evidence=evidence,
                chat_ids=None if scope is None else tuple(map(int, scope)),
                now=None if now is None else datetime.fromtimestamp(now, timezone.utc),
            ))
        except (AttributeError, KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"invalid case {i}: {exc}") from exc
    return tuple(cases)


def evaluate(
    conn,
    cases: tuple[Case, ...] | None = None,
    top_k: int | None = None,
    *,
    keyword_only: bool = False,
) -> list[CaseResult]:
    cases = CASES if cases is None else cases
    if keyword_only:
        cases = tuple(c for c in cases if c.keyword_ok)
    out = []
    for case in cases:
        trace = retrieve.SearchTrace() if case.evidence else None
        started = perf_counter()
        hits = retrieve.search(
            conn, case.question, chat_id=case.chat_ids, top_k=top_k, now=case.now, trace=trace
        )
        result = CaseResult(
            case, False, [h.window_id for h in hits], search_ms=(perf_counter() - started) * 1000
        )
        if trace is None:
            blob = "\n".join(h.text for h in hits).lower()
            result.hit = all(n.lower() in blob for n in case.needles)
        else:
            expected = set(case.evidence)
            candidates = {
                (c.chat_id, c.msg_id) for c in trace.candidates
                if c.fusion_rank <= trace.candidate_limit
            }
            found: set[tuple[int, int]] = set()
            relevant = 0
            for rank, hit in enumerate(hits, 1):
                if hit.kind != "chat":
                    continue
                matches = expected.intersection((hit.chat_id, msg) for msg in hit.msg_ids)
                if matches:
                    if result.first_relevant_rank is None:
                        result.first_relevant_rank = rank
                    relevant += 1
                    found.update(matches)
            result.hit = found == expected
            result.candidate_recall = len(expected & candidates) / len(expected)
            result.evidence_recall = len(found) / len(expected)
            result.excerpt_precision = relevant / len(hits) if hits else 0.0
        out.append(result)
    return out


def recall(results: list[CaseResult]) -> float:
    if not results:
        return 0.0
    return sum(r.hit for r in results) / len(results)


def format_miss_report(rows, limit_answer: int = 200) -> str:
    """Text report of logged questions that abstained or cited nothing."""
    if not rows:
        return "no retrieval misses in query_log"
    lines = [f"retrieval misses: {len(rows)}"]
    for row in rows:
        answer = (row["answer"] or "").replace("\n", " ")
        if len(answer) > limit_answer:
            answer = answer[: limit_answer - 1] + "…"
        lines.append(
            f"{row['ts']} abstain={row['abstain']} cited={row['cited_ids']}"
        )
        lines.append(f"  Q: {row['question']}")
        if answer:
            lines.append(f"  A: {answer}")
    return "\n".join(lines)


def main() -> None:
    import argparse
    import sys

    logconfig.setup()
    ap = argparse.ArgumentParser(description="Measure retrieval success@k on the golden set")
    ap.add_argument("--fixture", action="store_true", help="index tests/make_fixture.py into a temp DB")
    ap.add_argument("--cases", type=Path, help="JSON file of questions and expected message IDs")
    ap.add_argument("-k", type=int, default=None, help="override the production excerpt cap")
    ap.add_argument(
        "--misses",
        action="store_true",
        help="print query_log rows that abstained or cited nothing",
    )
    ap.add_argument("--limit", type=int, default=50)
    args = ap.parse_args()
    try:
        cases = load_cases(args.cases) if args.cases else None
    except (OSError, ValueError) as exc:
        ap.error(str(exc))

    if args.misses:
        conn = db.connect()
        print(format_miss_report(db.retrieval_misses(conn, args.limit)))
        return

    if args.fixture:
        from .ingest.export import load_data

        root = Path(__file__).resolve().parents[1]
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from tests.make_fixture import build_export, record_fixture_names

        conn = db.connect(":memory:")
        load_data(conn, build_export(), source="fixture")
        record_fixture_names(conn)
        index.reindex(conn, progress=True)
    else:
        conn = db.connect()

    results = evaluate(conn, cases, top_k=args.k)
    n = len(results)
    hits = sum(r.hit for r in results)
    label = f"@{args.k}" if args.k is not None else " (production cap)"
    print(f"success{label}: {hits}/{n} ({recall(results):.0%})")
    for r in results:
        mark = "ok" if r.hit else "MISS"
        kind = "" if r.case.keyword_ok else " (needs vectors)"
        print(f"  [{mark}] {r.case.question}{kind}")
        if r.evidence_recall is not None:
            print(
                f"         candidate recall {r.candidate_recall:.0%}, "
                f"final recall {r.evidence_recall:.0%}, "
                f"first relevant rank {r.first_relevant_rank or '—'}, {r.search_ms:.0f} ms"
            )
        elif not r.hit:
            print(f"         needles: {', '.join(r.case.needles)}")
    if results:
        times = sorted(r.search_ms for r in results)
        p50, p95 = (times[ceil(n * p) - 1] for p in (0.50, 0.95))
        print(f"search latency p50 {p50:.0f} ms, p95 {p95:.0f} ms")
    annotated = [r for r in results if r.evidence_recall is not None]
    if annotated:
        for label, attr in (
            ("mean candidate recall", "candidate_recall"),
            ("mean final evidence recall", "evidence_recall"),
            ("annotated excerpt precision", "excerpt_precision"),
        ):
            print(f"{label}: {sum(getattr(r, attr) for r in annotated) / len(annotated):.1%}")
        mrr = sum(1 / r.first_relevant_rank for r in annotated if r.first_relevant_rank) / len(annotated)
        print(f"mean reciprocal rank: {mrr:.3f}")


if __name__ == "__main__":
    main()
