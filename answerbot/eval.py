"""Golden-set retrieval eval: did the right window land in top-k?

Cases come from the synthetic export in tests/make_fixture.py — the same
conversations the README uses as examples. Success is "every needle appears
in the concatenated top-k window text", which is what the LLM would see.

    python -m answerbot.eval              # against DB_PATH
    python -m answerbot.eval --fixture    # load + index the fixture, then eval
"""

from __future__ import annotations

from dataclasses import dataclass

from . import config, db, index, retrieve


@dataclass(frozen=True)
class Case:
    question: str
    needles: tuple[str, ...]
    # True if BM25 alone should retrieve the needles (no embed model needed).
    keyword_ok: bool = True


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
    Case("why was the morning meeting moved", ("10:30",), keyword_ok=False),
)


@dataclass
class CaseResult:
    case: Case
    hit: bool
    window_ids: list[int]


def evaluate(
    conn,
    cases: tuple[Case, ...] | None = None,
    top_k: int | None = None,
    *,
    keyword_only: bool = False,
) -> list[CaseResult]:
    cases = cases or CASES
    if keyword_only:
        cases = tuple(c for c in cases if c.keyword_ok)
    top_k = top_k or config.TOP_K
    out = []
    for case in cases:
        hits = retrieve.search(conn, case.question, top_k=top_k)
        blob = "\n".join(h.text for h in hits).lower()
        hit = all(n.lower() in blob for n in case.needles)
        out.append(CaseResult(case, hit, [h.window_id for h in hits]))
    return out


def recall(results: list[CaseResult]) -> float:
    if not results:
        return 0.0
    return sum(r.hit for r in results) / len(results)


def main() -> None:
    import argparse
    import sys
    from pathlib import Path

    ap = argparse.ArgumentParser(description="Measure retrieval success@k on the golden set")
    ap.add_argument("--fixture", action="store_true", help="index tests/make_fixture.py into a temp DB")
    ap.add_argument("-k", type=int, default=None)
    args = ap.parse_args()

    if args.fixture:
        from .ingest.export import load_data

        root = Path(__file__).resolve().parents[1]
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from tests.make_fixture import build_export

        conn = db.connect(":memory:")
        load_data(conn, build_export(), source="fixture")
        index.reindex(conn, progress=True)
    else:
        conn = db.connect()

    results = evaluate(conn, top_k=args.k)
    n = len(results)
    hits = sum(r.hit for r in results)
    k = args.k or config.TOP_K
    print(f"success@{k}: {hits}/{n} ({recall(results):.0%})")
    for r in results:
        mark = "ok" if r.hit else "MISS"
        kind = "" if r.case.keyword_ok else " (needs vectors)"
        print(f"  [{mark}] {r.case.question}{kind}")
        if not r.hit:
            print(f"         needles: {', '.join(r.case.needles)}")


if __name__ == "__main__":
    main()
