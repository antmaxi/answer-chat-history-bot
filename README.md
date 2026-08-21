# answer-chat-history-bot

![answer-chat-history-bot](docs/answer-chat-history-bot.png)

Bot to answer questions on the Telegram chat history.

Full design in [docs/PLAN.md](docs/PLAN.md). The whole pipeline — ingest, index,
search, answer, and the Telegram bot with live ingest — is implemented.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu  # CPU-only, much smaller
.venv/bin/pip install -e .
```

## Docker

Optional. Same single-process app, with CPU torch and the default embed model
already in the image — useful for running the bot without a venv. Copy
`.env.example` to `.env` and fill in the keys you need first.

```bash
mkdir -p data
# put the Telegram export at data/result.json, then:
docker compose run --rm bot python -m answerbot.ingest.export /data/result.json
docker compose run --rm bot python -m answerbot.index
docker compose up -d
```

SQLite lives at `data/answerbot.db`. If you already indexed locally, copy
`answerbot.db` into `data/` before the first `up`. The entrypoint chowns
`data/` to the container user so a root-created bind mount still works.

Stop the bot (`docker compose stop`) before a bulk export or `index --update`
so two writers don't share the file. Other commands (`search`, `answer`,
`people`, `index --update`) are the same `docker compose run --rm bot python -m
answerbot…` form, with paths under `/data`. Ollama on the host:
`OLLAMA_HOST=http://host.docker.internal:11434`.

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
windows instead of excerpts. `answer` needs an LLM key set for the chosen
`LLM_PROVIDER` (Claude, Gemini, Groq, or OpenRouter).

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

### Fixing people's names

A Telegram export records each sender under the name the **exporting account**
had saved for them — so contacts show up under that account's private labels,
not their real public names, and those labels then end up in embeddings,
prompts, and answers. The export itself has no public names, only a stable user
id, so real names have to come from elsewhere. Three ways to supply them, keyed
by that id:

- **Automatic (live):** while the bot runs, every message carries the sender's
  real public name, which is recorded and overrides the label on the next
  reindex. Active members self-heal over time, for free.
- **API backfill:** an admin runs `/resolve` inside the group; the bot looks up
  each member via the Bot API. Only people *still in the group* can be resolved.
- **Manual (fully local, no API):**

  ```bash
  python -m answerbot.people --template names.json   # dump everyone, busiest first
  # edit the "name" fields in names.json
  python -m answerbot.people --import names.json      # apply them
  python -m answerbot.index                           # rewrite history with the names
  ```

`python -m answerbot.people --stats` shows how many are resolved. A name you set
by hand is never overwritten by a live sighting. Names live in the window text,
so changes take effect on the next (re)index — a full `index` to apply them
everywhere, or `--update` for just the recent tail.

#### Or drop names entirely

To show speakers as anonymous ids instead of any name — resolved names *and*
export labels suppressed everywhere — set `SPEAKER_LABEL=id`:

```bash
SPEAKER_LABEL=id python -m answerbot.index          # or: python -m answerbot.index --speaker-label id
```

Speakers then render as `User N`, a stable sequential pseudonym (stored in the
`aliases` table, assigned once) — so threads and "who said what" still hold
together, without exposing anyone's real telegram id. Set it in `.env` so the
bot uses it too. Caveat: this anonymizes the *speaker label* only — message
**text** can still contain names or `@mentions` that people typed, left as-is.

## Running the bot

1. Create a bot with @BotFather and copy the token.
2. **Turn privacy mode OFF** (BotFather → Bot Settings → Group Privacy), or the
   bot receives no group messages to read or index.
3. Set `TELEGRAM_BOT_TOKEN`, `ADMIN_USER_IDS` (your numeric Telegram user id),
   and an LLM key (`ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `GROQ_API_KEY`, or
   `OPENROUTER_API_KEY`) in `.env`. Open a DM with the bot so it can send you
   `Bot is up` / `Bot is down` when polling starts or stops, and any logged
   error with its traceback.
4. Add the bot to your group, then run:

```bash
python -m answerbot.bot
```

Or `docker compose up -d` if you followed [Docker](#docker) above.

In a group it answers when @mentioned or replied to. In a DM it answers if you
belong to a chat it has indexed — `/chats` lists those chats, `/chat N` focuses
one, `/chat all` searches every chat you are in. New group messages are appended
live; the tail is re-windowed every `LIVE_REINDEX_EVERY` messages, and every
`LIVE_LOOKBACK_HOURS` the last couple of weeks are rebuilt so recent edits
self-heal. `/stats` and (for admins) `/reindex` / `/reindex full` are available.

## Switching the answer model

Answer generation is the only cloud LLM call; embeddings are already local.
Set `LLM_PROVIDER` and, if needed, `ANSWER_MODEL`:

- **Claude** (default): `LLM_PROVIDER=claude` and `ANTHROPIC_API_KEY`
- **Gemini**: `LLM_PROVIDER=gemini` and `GEMINI_API_KEY` (defaults to
  `gemini-2.5-flash`; override with `ANSWER_MODEL`)
- **Groq**: `LLM_PROVIDER=groq` and `GROQ_API_KEY` (defaults to
  `openai/gpt-oss-20b`)
- **OpenRouter**: `LLM_PROVIDER=openrouter` and `OPENROUTER_API_KEY`
  (defaults to `openai/gpt-oss-20b:free`; any OpenRouter model id works,
  including paid ones without the `:free` suffix)
- **Ollama**: `LLM_PROVIDER=ollama`, `ANSWER_MODEL` to a pulled model, and
  optionally `OLLAMA_HOST` / `LLM_TIMEOUT`

See [answerbot/llm.py](answerbot/llm.py).

## Trying it without real data

```bash
python tests/make_fixture.py > /tmp/result.json
DB_PATH=/tmp/demo.db python -m answerbot.ingest.export /tmp/result.json
DB_PATH=/tmp/demo.db python -m answerbot.index
DB_PATH=/tmp/demo.db python -m answerbot.search "why was the morning meeting moved"
```

## Configuration

All optional, via environment or a `.env` file. Copy [`.env.example`](.env.example)
and fill in the keys you need. Values are read once at startup from
[answerbot/config.py](answerbot/config.py).

### Paths and logs

- **`DB_PATH`** (`answerbot.db`) — SQLite file for messages, windows, and
  embeddings. In Docker this is `/data/answerbot.db`.
- **`LOG_LEVEL`** (`INFO`) — logging threshold (`DEBUG`, `INFO`, `WARNING`,
  `ERROR`).
- **`LOG_PATH`** — rotating file next to the database (`answerbot.log` by
  default). Logs also go to stderr. Set to `off` (or `0` / `false` / `none`)
  for stderr only; any other path writes there instead.

### Embeddings

Local, no API key. A full `index` is required after changing these.

- **`EMBED_MODEL`** (`intfloat/multilingual-e5-small`) — sentence-transformers
  model used to embed windows and queries.
- **`EMBED_DIM`** (`384`) — vector width stored in SQLite. Must match the
  model; wrong values make search silently useless.

### Windowing

Consecutive messages are grouped into conversation windows (the retrieval
unit). A new window starts when the time gap is too large, or the current
window hits a size cap. Reply chains stay together regardless of the gap.
Change these, then run a full `index`.

- **`WINDOW_GAP_SECONDS`** (`1800`) — idle gap that starts a new window
  (30 minutes).
- **`WINDOW_MAX_MSGS`** (`25`) — maximum messages in one window.
- **`WINDOW_MAX_CHARS`** (`1500`) — maximum characters in one window.
- **`WINDOW_OVERLAP`** (`2`) — messages copied onto the next window so an
  answer is not cut in half at the boundary.
- **`UPDATE_LOOKBACK_DAYS`** (`14`) — on `index --update` (and the bot's
  periodic live lookback), how far back to re-window on top of the open tail,
  so recent *edits* are picked up. `0` is tail-only.

### Speaker names

- **`SPEAKER_LABEL`** (`name`) — `name` uses the resolved real name, falling
  back to the export label. `id` renders stable anonymous `User N` aliases
  instead (see [Or drop names entirely](#or-drop-names-entirely)). Names live
  in window text, so a reindex is required after changing this.

### Retrieval

Hybrid keyword (FTS5) + vector search, merged with reciprocal rank fusion.

- **`TOP_K`** (`8`) — windows passed to the LLM as excerpts.
- **`RRF_K`** (`20`) — RRF smoothing; lower keeps the top ranks more
  separated (the usual paper default is 60, which is meant for much longer
  result lists).
- **`WEIGHT_VECTOR`** (`1.0`) / **`WEIGHT_KEYWORD`** (`0.7`) — relative
  weight of each list in the fusion. Keyword is slightly down-weighted so
  embedding similarity leads.
- **`STOPWORD_DF_RATIO`** (`0.25`) — query terms that appear in more than
  this fraction of messages are dropped from the keyword query (chat filler
  like "yeah" / "ok").

### Answering

Answer generation is the only cloud LLM call. See
[Switching the answer model](#switching-the-answer-model) for provider
setup.

- **`LLM_PROVIDER`** (`claude`) — `claude`, `gemini`, `groq`, `openrouter`,
  or `ollama`.
- **`ANTHROPIC_API_KEY`**, **`GEMINI_API_KEY`** (or **`GOOGLE_API_KEY`**),
  **`GROQ_API_KEY`**, **`OPENROUTER_API_KEY`** — key for the chosen
  provider. Unused keys can be left empty.
- **`OPENROUTER_HTTP_REFERER`** / **`OPENROUTER_APP_TITLE`** (`answer-chat-history-bot`)
  — optional OpenRouter attribution headers. Referer is omitted unless set.
- **`ANSWER_MODEL`** — model id. Defaults: Claude `claude-sonnet-5`, Gemini
  `gemini-2.5-flash`, Groq `openai/gpt-oss-20b`, OpenRouter
  `openai/gpt-oss-20b:free`. Required for Ollama (whatever you have pulled).
- **`ANSWER_MAX_TOKENS`** (`8192`) — Groq/OpenRouter completion budget. On
  reasoning models this covers *thinking plus* the visible answer; 1024 is
  often not enough and comes back as an empty reply.
- **`ANSWER_MAX_REQUEST_TOKENS`** (`0`) — cap on *prompt plus* completion
  for one request. `0` means no extra cap, except Groq which then uses
  `8000` to match on_demand TPM (a request that reserves 8192 completion
  tokens 413s even with a short prompt). Raise this if you are on Dev Tier.
- **`ANSWER_COOLDOWN_SECONDS`** (`8`) — per-user wait between answers in
  the same chat. `0` disables. Admins skip the cooldown.
- **`QUERY_LOG`** (`1`) — persist each question and the retrieved window ids
  locally. `0` / `false` / `off` turns it off.
- **`LLM_TIMEOUT`** (`60`) — seconds to wait for the provider.
- **`OLLAMA_HOST`** (`http://localhost:11434`) — Ollama base URL. From
  Docker, talking to Ollama on the host is
  `http://host.docker.internal:11434`.

### Telegram bot

- **`TELEGRAM_BOT_TOKEN`** — from @BotFather. Required to run the bot.
  Privacy mode must be **off** or the bot sees no group messages.
- **`ADMIN_USER_IDS`** — numeric Telegram user ids (space- or
  comma-separated). Those accounts get `Bot is up` / `Bot is down` DMs and
  ERROR logs with traceback; they can run `/reindex` and skip the answer
  cooldown. Open a DM with the bot first so it can write to you.
- **`LIVE_REINDEX_EVERY`** (`20`) — new messages that may accumulate before
  the open tail is re-windowed.
- **`LIVE_LOOKBACK_HOURS`** (`6`) — how often to rebuild the last
  `UPDATE_LOOKBACK_DAYS` of history so recent edits self-heal. `0` disables
  the periodic pass (tail reindex still runs).
- **`MEMBERSHIP_CACHE_SECONDS`** (`300`) — how long a "is this user in this
  chat?" Bot API lookup is remembered, used to decide who may ask in DM.

## Tests

```bash
.venv/bin/python -m pytest tests/ -q
```

Covers parsing, windowing, query construction, and a golden-set retrieval
check (keyword success@k on the synthetic fixture). For real embeddings:

```bash
python -m answerbot.eval --fixture
```

against an already-indexed DB, drop `--fixture`.
