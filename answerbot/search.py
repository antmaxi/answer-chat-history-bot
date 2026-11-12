"""Search the indexed history from the command line.

This is the checkpoint that matters: if retrieval is bad here, no prompt will
fix it downstream, and debugging it through Telegram is miserable.
"""

import argparse
import json
import sys
import textwrap
from dataclasses import asdict

from . import db, logconfig, retrieve


def main() -> None:
    logconfig.setup()
    ap = argparse.ArgumentParser(description="Search indexed chat history")
    ap.add_argument("question", nargs="+")
    ap.add_argument("--chat-id", type=int, action="append", help="limit to this chat (repeatable)")
    ap.add_argument("-k", "--top-k", type=int, default=None)
    ap.add_argument("--full", action="store_true", help="print whole windows, not excerpts")
    ap.add_argument("--explain", action="store_true", help="print ranking diagnostics as JSON to stderr")
    args = ap.parse_args()

    question = " ".join(args.question)
    conn = db.connect()
    trace = retrieve.SearchTrace() if args.explain else None
    hits = retrieve.search(conn, question, args.chat_id, args.top_k, trace=trace)
    if trace is not None:
        print(json.dumps(asdict(trace), ensure_ascii=False, indent=2), file=sys.stderr)

    if not hits:
        print("no matches — is the index built? try: python -m answerbot.index")
        return

    for i, hit in enumerate(hits, 1):
        print(f"\n{'─' * 72}")
        print(f"[{i}] {hit.when()}  ·  {hit.speakers}  ·  score {hit.score:.4f}")
        print(f"    {hit.link()}")
        print()
        body = hit.text if args.full else textwrap.shorten(
            hit.text.replace("\n", " ⏎ "), width=600, placeholder=" …"
        )
        print(textwrap.indent(body, "    "))


if __name__ == "__main__":
    main()
