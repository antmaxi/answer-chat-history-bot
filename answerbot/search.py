"""Search the indexed history from the command line.

This is the checkpoint that matters: if retrieval is bad here, no prompt will
fix it downstream, and debugging it through Telegram is miserable.
"""

import argparse
import textwrap

from . import db, retrieve


def main() -> None:
    ap = argparse.ArgumentParser(description="Search indexed chat history")
    ap.add_argument("question", nargs="+")
    ap.add_argument("--chat-id", type=int, default=None)
    ap.add_argument("-k", "--top-k", type=int, default=None)
    ap.add_argument("--full", action="store_true", help="print whole windows, not excerpts")
    args = ap.parse_args()

    question = " ".join(args.question)
    conn = db.connect()
    hits = retrieve.search(conn, question, args.chat_id, args.top_k)

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
