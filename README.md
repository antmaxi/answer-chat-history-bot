# answer-bot

Bot to answer questions on the Telegram chat history.

Full design in [docs/PLAN.md](docs/PLAN.md). The whole pipeline — ingest, index,
search, answer, and the Telegram bot with live ingest — is implemented.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu  # CPU-only, much smaller
.venv/bin/pip install -e .
```

## Usage

Export the chat from Telegram Desktop (chat menu → Export chat history → format
**JSON**, media not needed), then:

```bash
python -m answerbot.ingest.export path/to/result.json   # load messages
python -m answerbot.index                               # build windows + embeddings
python -m answerbot.search "how much was the ski trip"  # inspect retrieval
python -m answerbot.answer "how much was the ski trip"  # grounded answer + sources
```

`search` takes `-k N` for the number of results and `--full` to print whole
windows instead of excerpts. `answer` needs `ANTHROPIC_API_KEY` set.

### Topping up with new history

When you re-export the chat later (or otherwise add messages), don't rebuild
from scratch — re-embedding the whole corpus costs minutes. Load the newer
export and index only what changed:

```bash
python -m answerbot.ingest.export path/to/newer-export.json  # upserts messages
python -m answerbot.index --update                           # embeds only the new tail
```

`--update` re-windows and re-embeds the new tail **plus the last couple of
weeks** of history (`--lookback-days`, default 14), leaving the rest untouched —
seconds, not the minutes a full rebuild costs. The lookback is what catches
*edits* to recent messages, which a pure-tail update would never revisit; older
edits still need a periodic full `index` to reconcile. Use `--lookback-days 0`
for tail-only, or a larger value to reach further back:

```bash
python -m answerbot.index --update                    # tail + last 14 days
python -m answerbot.index --update --lookback-days 30  # reach back a month
```

`python -m answerbot.index` without a flag does a full rebuild — use that only
after changing the windowing or embedding settings. Re-running the export loader
is always safe; messages are upserted on `(chat_id, msg_id)`.

The bot re-windows live messages automatically (tail-only, for speed — see
below), so `--update` is mainly for bulk top-ups from a fresh export.

## Running the bot

1. Create a bot with @BotFather and copy the token.
2. **Turn privacy mode OFF** (BotFather → Bot Settings → Group Privacy), or the
   bot receives no group messages to read or index.
3. Set `TELEGRAM_BOT_TOKEN`, `ANTHROPIC_API_KEY`, and `ADMIN_USER_IDS` (your
   numeric Telegram user id) in `.env`.
4. Add the bot to your group, then run:

```bash
python -m answerbot.bot
```

In a group it answers when @mentioned or replied to. In a DM it answers if you
belong to a chat it has indexed. New group messages are appended live and the
tail is re-windowed every `LIVE_REINDEX_EVERY` messages. `/stats` and (for
admins) `/reindex` are available.

## Switching to a local model

Answer generation is the only Claude call; embeddings are already local. Set
`LLM_PROVIDER=ollama` and `ANSWER_MODEL` to a pulled model — see
[answerbot/llm.py](answerbot/llm.py) `OllamaLLM`.

## Trying it without real data

```bash
python tests/make_fixture.py > /tmp/result.json
DB_PATH=/tmp/demo.db python -m answerbot.ingest.export /tmp/result.json
DB_PATH=/tmp/demo.db python -m answerbot.index
DB_PATH=/tmp/demo.db python -m answerbot.search "why was the morning meeting moved"
```

## Configuration

All optional, via environment or a `.env` file — see [answerbot/config.py](answerbot/config.py).
The ones worth knowing: `DB_PATH`, `EMBED_MODEL`, `WINDOW_GAP_SECONDS`, `TOP_K`.

## Tests

```bash
.venv/bin/python -m pytest tests/ -q
```

Covers parsing, windowing and query construction. Retrieval *quality* is not
unit-tested — check it by eye with `answerbot.search`.
