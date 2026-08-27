# answer-chat-history-bot — system plan

A Telegram bot that answers questions from a chat's own message history.
The v1 pipeline (ingest → index → retrieve → answer → bot + live ingest) is
shipped. This document is the current design, not a build queue.

## Decisions

| Area | Choice |
|---|---|
| History | One-time Telegram Desktop JSON export to seed, Bot API to stay live |
| Answering | Claude, Gemini, Groq, OpenRouter, Cursor, or local Ollama (`LLM_PROVIDER`) behind the same protocol |
| Embeddings | Local `sentence-transformers` (`intfloat/multilingual-e5-small`) |
| Interface | Group (@mention, reply to a question, or reply to the bot) **and** private DM |
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
                                                     msg vecs,  thread      Groq / OpenRouter /
                                                     FTS5)      expansion)  Cursor / Ollama)
```

Modules:

- `ingest/export.py` — parse Telegram Desktop JSON into normalized rows
- `ingest/live.py` — append / upsert Bot API messages; flush or refresh the tail
- `index.py` — conversation windows, per-message embed (plan/apply so encode can run off the DB lock)
- `retrieve.py` — hybrid BM25 + cosine over messages, query-time thread expansion, optional time/speaker filters; `query_vec` so the bot can encode off the SQLite lock
- `thread.py` — replies / @mentions / cosine neighbours around a seed message
- `timerange.py` / `people.py` / `followup.py` — question parsing helpers
- `answer.py` — provider-agnostic LLM call, grounded prompt + citations
- `i18n.py` — RU/EN UI strings; Russian default, `/settings` to switch
- `bot.py` — aiogram: routing, DM allow-list, live ingest, cooldown, lookback
- `eval.py` — golden-set success@k (`python -m answerbot.eval [--fixture]`)

## Why windows, then message threads

Chat messages are short and context-free ("yeah, that one"). Consecutive
messages are still grouped into **conversation windows** for incremental
indexing and stats:

- new window when the time gap exceeds ~30 min, or the window passes ~25 messages / ~1500 chars
- 2-message overlap between adjacent windows
- reply chains and @mentions of a speaker in the window stay together regardless of the time gap

Retrieval ranks **messages** (BM25 + cosine). Each hit is expanded at query
time into a thread (replies, @mentions, embedding neighbours). That excerpt
is the citation unit — so two topics in the same burst do not share a bag.

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
  vec       BLOB NOT NULL            -- leftover; no longer written
);

CREATE TABLE message_vecs (
  message_id INTEGER PRIMARY KEY REFERENCES messages(id) ON DELETE CASCADE,
  vec        BLOB NOT NULL
);

CREATE TABLE state (chat_id INTEGER PRIMARY KEY, last_indexed_msg_id INTEGER);

CREATE TABLE people (                -- real names keyed by telegram user id
  sender_id    INTEGER PRIMARY KEY,
  display_name TEXT NOT NULL,
  username     TEXT,
  source       TEXT NOT NULL,        -- live < api < manual
  updated_at   INTEGER NOT NULL
);

CREATE TABLE resolve_misses (        -- getChatMember failures for resumable /resolve
  sender_id  INTEGER PRIMARY KEY,
  reason     TEXT NOT NULL,
  updated_at INTEGER NOT NULL
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

1. **BM25** over `messages_fts` — names, dates, exact terms
2. **Vector** cosine over `message_vecs` — paraphrase

Fusion ranks messages. Unless the question already has a time range,
fused scores are multiplied by a recency decay (`0.5^(age / RECENCY_HALF_LIFE_DAYS)`
on the message timestamp) so newer excerpts outrank equally relevant older ones.
Each ranked message is expanded into a thread (`thread.expand_thread`); near-
duplicate threads are merged. Then `cap_hits` keeps `MIN_K`–`MAX_K` excerpts
(defaults match `TOP_K`, so the model sees 10): always the first `MIN_K`, then
stop when cosine falls below `COSINE_MIN`. The bot does not re-window on each
question; live ingest batches (`LIVE_REINDEX_EVERY`) and the periodic lookback do.

An empty chat allow-list is *no chats*, never “all chats”. Time phrases
(“last week”, “yesterday”, “in February”, ISO dates) filter messages by `ts`.
“What did Anna say” over-fetches then keeps threads whose `speakers` contain
that name.

Short follow-ups (“how much was it”, a reply to the bot) stitch the previous
question into the *search* string. The answer model still sees the original
wording.

## Answering

`answer.py` exposes `complete_answer` / `answer` and an `LLM` protocol.
`ClaudeLLM`, `GeminiLLM`, `GroqLLM`, `OpenRouterLLM`, `CursorLLM`, and
`OllamaLLM` all honour `LLM_TIMEOUT` except Cursor, whose local agent run is
not an HTTP completion and is not hard-capped by that value. Groq and
OpenRouter speak the OpenAI Chat Completions API over HTTPS (no extra SDK).
Ollama uses `OLLAMA_HOST` and fails with a clear error on timeout or empty
response. Gemini reads `GEMINI_API_KEY` (or `GOOGLE_API_KEY`); Groq
`GROQ_API_KEY`; OpenRouter `OPENROUTER_API_KEY`; Cursor `CURSOR_API_KEY`
(and the optional `cursor-sdk` extra). Cursor concatenates the system and
user strings into one agent prompt and disables tools so the run can only
return text.

Prompt rule: answers come **only** from the supplied excerpts. “I couldn't find
this in the history” is a valid answer. The model writes **Markdown** (`**bold**`,
lists, `` `code` ``); the bot renders that subset as Telegram HTML. Context
blocks carry `[W3] 2026-03-14, Anna & Nino:` headers; the bot turns `[W3]` into
`t.me/c/<chat>/<msg_id>` links.

## Bot behaviour

- **Group**: replies when @mentioned (a bare @mention as a reply to someone
  else's message uses that text as the question) or when someone replies to
  its message, and only in `TELEGRAM_CHAT_ID`. Privacy mode **off** in BotFather, otherwise
  it receives nothing. Incoming messages and edits are ingested; the tail
  re-windows every `LIVE_REINDEX_EVERY` messages. Every `LIVE_LOOKBACK_HOURS`
  the last `UPDATE_LOOKBACK_DAYS` are rebuilt so recent edits self-heal.
  Telegram does not send group deletes to bots.
- **DM**: any plain message is a question if the sender is a member of
  `TELEGRAM_CHAT_ID` (`getChatMember`, TTL-cached). Non-members are declined.
  Search always uses that chat.
- Commands: `/ask` (prompt, then the next message) or `/ask <question>`;
  `/cancel` (stop a running search or a pending `/ask`); `/info` (includes
  index stats), `/settings` (UI language: Russian default, or English);
  admins also `/stats` (index, question counts, ask-time median ± std
  and min/max over the last day / week / month; `/stats a b` lists terms
  in a–b% of messages), `/reindex` (lookback)
  and `/reindex full`, `/resolve` (Bot API names, background, resumable;
  `/resolve retry` / `/resolve stop`). Non-admins see only
  `/ask`, `/cancel`, `/settings`, `/info`, `/help` in the command menu.
- Several members can ask at once: each ask is its own task. Query encode
  is off the SQLite lock (one SentenceTransformer call at a time); LLM
  generation overlaps. Cooldown is per-user-per-chat, so different people
  are not throttled by each other.
- Per-user-per-chat cooldown (`ANSWER_COOLDOWN_SECONDS`); admins exempt.
- Sliding-hour LLM caps (`ANSWER_MAX_PER_USER_PER_HOUR`, `ANSWER_MAX_PER_HOUR`);
  admins exempt. In-memory, counted only when retrieval returned windows.
- Admins are DMed `Bot is starting` as soon as polling begins, then
  `Bot started, stats: …` after setup (chat lookup, commands, embedding
  warmup), `Bot is down` on graceful stop, and any `ERROR` log line
  (message + traceback). Error DMs are
  coalesced (~20s) so a tight loop cannot flood Telegram. They must have
  `/start`'d the bot first.

## Indexing

`index.reindex` rebuilds every window. `index.update` (CLI `--update`, bot
`/reindex`) re-windows the open tail plus `--lookback-days` (default 14) so
recent edits are picked up without a full embed. Encoding is planned under the
DB lock and applied after `embed.encode_passages` returns. The bot encodes
search queries the same way — `embed.encode_query` off the lock, then a short
SQLite critical section — so ingest and another member's FTS are not stuck
behind SentenceTransformer. Query and passage encode still take turns on the
model.

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
