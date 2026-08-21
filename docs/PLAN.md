# answer-chat-history-bot — system plan

A Telegram bot that answers questions from a chat's own message history.
The v1 pipeline (ingest → index → retrieve → answer → bot + live ingest) is
shipped. This document is the current design, not a build queue.

## Decisions

| Area | Choice |
|---|---|
| History | One-time Telegram Desktop JSON export to seed, Bot API to stay live |
| Answering | Claude, Gemini, Groq, OpenRouter, or local Ollama (`LLM_PROVIDER`) behind the same protocol |
| Embeddings | Local `sentence-transformers` (`intfloat/multilingual-e5-small`) |
| Interface | Group (@mention or reply) **and** private DM |
| Storage | Single SQLite file — FTS5 for keyword, float32 blobs + numpy for vectors |

No vector DB, no Postgres, no queue. Docker is optional packaging of this same
single-process app, not extra infrastructure. At ~100k messages a brute-force
numpy dot product over 384-dim vectors is ~150 MB RAM and ~10 ms per query.
Qdrant/pgvector stay off the table until that is actually slow. A rerank pass
is the same: only if `python -m answerbot.eval` says hybrid RRF is not enough.

## Architecture

```
Telegram export (.json) ──┐
                          ├──> ingest ──> SQLite ──> index ──> retrieve ──> answer ──> Telegram
Bot API live updates ─────┘             (messages)  (windows,  (hybrid +    (Claude / Gemini /
                                                     vectors,   time/speaker) Groq / OpenRouter /
                                                     FTS5)                    Ollama)
```

Modules:

- `ingest/export.py` — parse Telegram Desktop JSON into normalized rows
- `ingest/live.py` — append / upsert Bot API messages; flush or refresh the tail
- `index.py` — conversation windows, embed (plan/apply so encode can run off the DB lock)
- `retrieve.py` — hybrid BM25 + cosine, RRF, optional time-range and speaker filters
- `timerange.py` / `people.py` / `followup.py` — question parsing helpers
- `answer.py` — provider-agnostic LLM call, grounded prompt + citations
- `bot.py` — aiogram: routing, DM allow-list, live ingest, cooldown, lookback
- `eval.py` — golden-set success@k (`python -m answerbot.eval [--fixture]`)

## Why windows, not messages

Chat messages are short and context-free ("yeah, that one"). Embedding them
individually retrieves noise. Consecutive messages become **conversation
windows**:

- new window when the time gap exceeds ~30 min, or the window passes ~25 messages / ~1500 chars
- 2-message overlap between adjacent windows so answers aren't cut in half
- reply chains stay together regardless of the time gap

Windows are the retrieval unit and the citation unit.

## Schema

```sql
CREATE TABLE messages (
  id         INTEGER PRIMARY KEY,
  chat_id    INTEGER NOT NULL,
  msg_id     INTEGER NOT NULL,       -- telegram message id
  reply_to   INTEGER,
  sender_id  INTEGER,
  sender     TEXT,                   -- export label or live display name
  ts         INTEGER NOT NULL,
  text       TEXT NOT NULL,
  UNIQUE (chat_id, msg_id)
);

CREATE VIRTUAL TABLE messages_fts USING fts5(
  text, content='messages', content_rowid='id', tokenize='unicode61'
);

CREATE TABLE windows (
  id        INTEGER PRIMARY KEY,
  chat_id   INTEGER NOT NULL,
  first_msg INTEGER NOT NULL,
  last_msg  INTEGER NOT NULL,
  ts_start  INTEGER NOT NULL,
  ts_end    INTEGER NOT NULL,
  speakers  TEXT,                    -- comma-separated labels
  text      TEXT NOT NULL            -- rendered "Name: message" transcript
);

CREATE TABLE window_vecs (
  window_id INTEGER PRIMARY KEY REFERENCES windows(id) ON DELETE CASCADE,
  vec       BLOB NOT NULL
);

CREATE TABLE state (chat_id INTEGER PRIMARY KEY, last_indexed_msg_id INTEGER);

CREATE TABLE people (                -- real names keyed by telegram user id
  sender_id    INTEGER PRIMARY KEY,
  display_name TEXT NOT NULL,
  username     TEXT,
  source       TEXT NOT NULL,        -- live < api < manual
  updated_at   INTEGER NOT NULL
);

CREATE TABLE aliases (               -- SPEAKER_LABEL=id → stable "User N"
  sender_id INTEGER PRIMARY KEY,
  ordinal   INTEGER NOT NULL UNIQUE
);

CREATE TABLE dm_prefs (              -- leftover DM focus; unused (one chat)
  user_id INTEGER PRIMARY KEY,
  chat_id INTEGER NOT NULL
);

CREATE TABLE query_log (
  id          INTEGER PRIMARY KEY,
  ts          INTEGER NOT NULL,
  question    TEXT NOT NULL,
  chat_ids    TEXT,
  window_ids  TEXT NOT NULL,
  cited_ids   TEXT NOT NULL,
  latency_ms  INTEGER,
  model       TEXT
);
```

New tables go in `SCHEMA`. New columns on existing tables go through
`db.ensure_column` inside `db.migrate()`, so an older file still opens.

## Retrieval

Hybrid, merged with Reciprocal Rank Fusion (`score = Σ weight/(RRF_K + rank)`):

1. **BM25** over `messages_fts`, mapped up to owning windows — names, dates, exact terms
2. **Vector** cosine over `window_vecs` — paraphrase

An empty chat allow-list is *no chats*, never “all chats”. Time phrases
(“last week”, “yesterday”, “in February”, ISO dates) filter windows by
`ts_start`/`ts_end`. “What did Anna say” over-fetches then keeps windows whose
`speakers` contain that name.

Short follow-ups (“how much was it”, a reply to the bot) stitch the previous
question into the *search* string. The answer model still sees the original
wording.

## Answering

`answer.py` exposes `complete_answer` / `answer` and an `LLM` protocol.
`ClaudeLLM`, `GeminiLLM`, `GroqLLM`, `OpenRouterLLM`, and `OllamaLLM` all honour
`LLM_TIMEOUT`. Groq and OpenRouter speak the OpenAI Chat Completions API over
HTTPS (no extra SDK). Ollama uses `OLLAMA_HOST` and fails with a clear error on
timeout or empty response. Gemini reads `GEMINI_API_KEY` (or `GOOGLE_API_KEY`);
Groq `GROQ_API_KEY`; OpenRouter `OPENROUTER_API_KEY`.

Prompt rule: answers come **only** from the supplied excerpts. “I couldn't find
this in the history” is a valid answer. Context blocks carry `[W3] 2026-03-14,
Anna & Nino:` headers; the bot turns `[W3]` into `t.me/c/<chat>/<msg_id>` links.

## Bot behaviour

- **Group**: replies when @mentioned or when someone replies to its message,
  and only in `TELEGRAM_CHAT_ID`. Privacy mode **off** in BotFather, otherwise
  it receives nothing. Incoming messages and edits are ingested; the tail
  re-windows every `LIVE_REINDEX_EVERY` messages. Every `LIVE_LOOKBACK_HOURS`
  the last `UPDATE_LOOKBACK_DAYS` are rebuilt so recent edits self-heal.
  Telegram does not send group deletes to bots.
- **DM**: any plain message is a question if the sender is a member of
  `TELEGRAM_CHAT_ID` (`getChatMember`, TTL-cached). Non-members are declined.
  Search always uses that chat.
- Commands: `/ask`, `/stats`; admins `/reindex` (lookback)
  and `/reindex full`, `/resolve` (Bot API names).
- Per-user-per-chat cooldown (`ANSWER_COOLDOWN_SECONDS`); admins exempt.
- Admins are DMed `Bot is up` / `Bot is down` on polling start and graceful
  stop, and any `ERROR` log line (message + traceback). Error DMs are
  coalesced (~20s) so a tight loop cannot flood Telegram. They must have
  `/start`'d the bot first.

## Indexing

`index.reindex` rebuilds every window. `index.update` (CLI `--update`, bot
`/reindex`) re-windows the open tail plus `--lookback-days` (default 14) so
recent edits are picked up without a full embed. Encoding is planned under the
DB lock and applied after `embed.encode_passages` returns, so search and ingest
are not blocked on SentenceTransformer.

## Quality

`tests/make_fixture.py` is a synthetic export. Keyword success@k on that set
runs in CI (`python -m pytest tests/`). Real vectors:

```bash
python -m answerbot.eval --fixture
```

## Stack

Python 3.11+, `aiogram` 3.x, `anthropic`, `google-genai`, `sentence-transformers`, `numpy`,
stdlib `sqlite3`. Config via `.env` — see `.env.example`. Logs go to stderr and a
rotating `answerbot.log` next to the DB. `Dockerfile` /
`docker-compose.yml` wrap the same process (CPU torch, SQLite on a volume).

## Known limits

- Export is a point-in-time snapshot; the window between export and bot-join is a gap.
- Deleted messages are not tracked. Edits older than the lookback window need a full reindex.
- Text only. Images, voice notes, and documents are indexed by caption or skipped.
- Retrieved windows go to the LLM as-is (wifi passwords and similar will be repeated if asked).
- `SPEAKER_LABEL=id` anonymizes speaker labels only; names people typed in message text stay.
- No rerank pass. Add one only if eval recall drops.
