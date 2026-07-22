# answer-bot — system plan (v1)

A Telegram bot that answers questions from a chat's own message history.

## Decisions

| Area | Choice |
|---|---|
| History | One-time Telegram Desktop JSON export to seed, Bot API to stay live |
| Answering | Claude API, behind a provider interface so Ollama can swap in later |
| Embeddings | Local `sentence-transformers` from day 1 (Anthropic has no embeddings endpoint) |
| Interface | Group (@mention or reply) **and** private DM |
| Storage | Single SQLite file — FTS5 for keyword, float32 blobs + numpy for vectors |

No vector DB, no Postgres, no queue, no Docker for v1. At ~100k messages a brute-force
numpy dot product over 384-dim vectors is ~150 MB RAM and ~10 ms per query. Introducing
Qdrant/pgvector here would be pure ceremony.

## Architecture

```
Telegram export (.json) ──┐
                          ├──> ingest ──> SQLite ──> index ──> retrieve ──> answer ──> Telegram
Bot API live updates ─────┘             (messages)  (windows,  (hybrid)    (Claude)
                                                     vectors,
                                                     FTS5)
```

Six modules, each independently testable:

- `ingest/export.py` — parse Telegram Desktop JSON into normalized rows
- `ingest/live.py` — aiogram handler that appends new messages to the same table
- `index.py` — group messages into conversation windows, embed, populate FTS
- `retrieve.py` — hybrid search, returns ranked windows
- `answer.py` — provider-agnostic LLM call, builds prompt + citations
- `bot.py` — aiogram app, routing, access control

## Why windows, not messages

Chat messages are short and context-free ("yeah, that one"). Embedding them individually
retrieves noise. Instead group consecutive messages into **conversation windows**:

- new window when the time gap exceeds ~30 min, or the window passes ~25 messages / ~1500 chars
- 2-message overlap between adjacent windows so answers aren't cut in half
- reply chains stay together regardless of the time gap

Windows are the retrieval unit and the citation unit. This is the single highest-leverage
design choice in the whole system.

## Schema

```sql
CREATE TABLE messages (
  id         INTEGER PRIMARY KEY,
  chat_id    INTEGER NOT NULL,
  msg_id     INTEGER NOT NULL,       -- telegram message id
  reply_to   INTEGER,
  sender_id  INTEGER,
  sender     TEXT,
  ts         INTEGER NOT NULL,       -- unix seconds
  text       TEXT NOT NULL,
  UNIQUE (chat_id, msg_id)
);

CREATE VIRTUAL TABLE messages_fts USING fts5(
  text, content='messages', content_rowid='id', tokenize='unicode61'
);

CREATE TABLE windows (
  id        INTEGER PRIMARY KEY,
  chat_id   INTEGER NOT NULL,
  first_msg INTEGER NOT NULL,        -- telegram msg_id, for deep links
  last_msg  INTEGER NOT NULL,
  ts_start  INTEGER NOT NULL,
  ts_end    INTEGER NOT NULL,
  speakers  TEXT,                    -- comma-separated, for "who said X"
  text      TEXT NOT NULL            -- rendered "Name: message" transcript
);

CREATE TABLE window_vecs (
  window_id INTEGER PRIMARY KEY REFERENCES windows(id) ON DELETE CASCADE,
  vec       BLOB NOT NULL            -- float32, normalized
);

CREATE TABLE state (chat_id INTEGER PRIMARY KEY, last_indexed_msg_id INTEGER);
```

## Retrieval

Hybrid, merged with Reciprocal Rank Fusion (`score = Σ 1/(60 + rank)`):

1. **BM25** over `messages_fts`, mapped up to owning windows — catches names, dates, exact terms
2. **Vector** cosine over `window_vecs` — catches paraphrase

RRF needs no score normalization and is about 15 lines. Take top 8 windows into the prompt.

Optional later, only if quality demands it: a rerank pass, or a cheap Haiku call that rewrites
the question into a standalone query when it's a follow-up.

## Answering

`answer.py` exposes one function and one protocol:

```python
class LLM(Protocol):
    def complete(self, system: str, messages: list[dict]) -> str: ...
```

`ClaudeLLM` for now, `OllamaLLM` later — the swap touches one file plus a config key.
Embeddings are already local, so nothing else moves.

Prompt shape: system prompt sets the rule that answers come **only** from the supplied
excerpts and that "I couldn't find this in the history" is a valid, expected answer. Context
blocks carry `[W3] 2026-03-14, Anna & Nino:` headers so the model can cite `[W3]`, which the
bot rewrites into `t.me/c/<chat>/<msg_id>` deep links.

Refusing to guess is the difference between a bot people trust and one they mute after a week.

## Bot behaviour

- **Group**: replies when @mentioned or when someone replies to its message. Requires privacy
  mode **off** in BotFather, otherwise it receives nothing.
- **DM**: any plain message is a question; the user must be a member of an indexed chat
  (checked via `getChatMember`) — this is the whole access-control story for v1.
- Commands: `/ask <q>`, `/stats`, `/reindex` (admin only).
- Answers are one paragraph plus up to 3 source links.

## Build order

Each milestone is independently useful — don't build the bot first.

- **M1 — Ingest.** Export parser → `messages`. Verify counts against the export.
- **M2 — Index.** Windowing + embeddings + FTS. Eyeball 10 windows for sane boundaries.
- **M3 — Search CLI.** `python -m answerbot.search "question"` prints ranked windows.
  *This is the real checkpoint:* if retrieval is bad here, no prompt will save it, and
  debugging it through Telegram is miserable.
- **M4 — Answer CLI.** Same command, now with a Claude-generated answer and citations.
- **M5 — Bot.** aiogram wrapper around M4. Mention + DM routing.
- **M6 — Live ingest.** Append incoming messages; re-window the tail on a timer.

M1–M4 need no bot token and no Telegram connection at all.

## Stack

Python 3.11+, `aiogram` 3.x, `anthropic`, `sentence-transformers`, `numpy`, stdlib `sqlite3`.
Embedding model: `intfloat/multilingual-e5-small` (384-dim, strong on non-English chat —
worth checking against your actual chat language mix before committing).

Config via `.env`: `TELEGRAM_BOT_TOKEN`, `ANTHROPIC_API_KEY`, `LLM_PROVIDER`, `DB_PATH`,
`ADMIN_USER_IDS`.

## Known limits of v1

- Export is a point-in-time snapshot; the window between export and bot-join is a gap.
- Edited and deleted messages are not tracked — the index drifts slowly from reality.
- Text only. Images, voice notes, and documents are indexed by caption or skipped.
- Single chat assumed throughout; the schema carries `chat_id` so multi-chat is additive.
