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
`.env.example` to `.env` and fill in the keys you need first. Set `HF_TOKEN`
there before `docker compose build` if Hub rate-limits anonymous downloads or
the embed model is gated.

```bash
mkdir -p data
# put the Telegram export at data/result.json, then:
docker compose run --rm bot python -m answerbot.ingest.export /data/result.json
docker compose run --rm bot python -m answerbot.index
docker compose up -d
```

SQLite lives at `data/answerbot.db` (the same default as a local run). The
entrypoint chowns `data/` to the container user so a root-created bind mount
still works.

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
`LLM_PROVIDER` (Claude, Gemini, Groq, OpenRouter, or Cursor).

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
not their real public names. Those labels are **not** used in embeddings,
prompts, or answers unless you set `SPEAKER_LABEL=export`. Unresolved people
render as `User N`. The export itself has no public names, only a stable user
id, so real names have to come from elsewhere. Three ways to supply them, keyed
by that id:

- **Automatic (live):** while the bot runs, every message carries the sender's
  real public name, which is recorded and overrides the label on the next
  reindex. Active members self-heal over time, for free.
- **API backfill:** an admin runs `/resolve` in the group or in DM. The bot
  looks up unresolved people via `getChatMember` in the background, waits out
  Telegram flood limits, and remembers who left so the next run continues
  instead of retrying all ~2k ids. `/resolve` while running shows progress;
  `/resolve stop` pauses; `/resolve retry` tries people previously skipped.
  Names keep emoji and styled unicode (custom premium emoji arrive as the API's
  fallback character). Only people the API can still see can be resolved — the
  bot should be a group **admin** (no extra rights needed) for that. Then
  `/reindex` to rewrite window text.
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
everywhere, or `--update` for just the recent tail. Anyone not yet resolved is
shown as `User N`.

#### Or drop names entirely

To show speakers as anonymous ids instead of any name — resolved names
suppressed everywhere — set `SPEAKER_LABEL=id`:

```bash
SPEAKER_LABEL=id python -m answerbot.index          # or: python -m answerbot.index --speaker-label id
```

Speakers then render as `User N`, a stable sequential pseudonym (stored in the
`aliases` table, assigned once) — so threads and "who said what" still hold
together, without exposing anyone's real telegram id. Set it in `.env` so the
bot uses it too. Caveat: this anonymizes the *speaker label* only — message
**text** can still contain names or `@mentions` that people typed, left as-is.

To use the exporter's contact labels after all (the old fallback), set
`SPEAKER_LABEL=export` and reindex.

## Running the bot

1. Create a bot with @BotFather and copy the token.
2. **Turn privacy mode OFF** (BotFather → Bot Settings → Group Privacy), or the
   bot receives no group messages to read or index.
3. Set `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` (the supergroup's Bot API id,
   e.g. `-1001234567890`; a positive id is treated as `-100<id>`),
   `ADMIN_USER_IDS` (your numeric Telegram user id),
   and an LLM key (`ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `GROQ_API_KEY`,
   `OPENROUTER_API_KEY`, or `CURSOR_API_KEY`) in `.env`. Open a DM with the bot
   so it can send you
   `Bot is starting`, then `Bot started, stats: …` after setup, `Bot is down`
   on stop, and any logged error with its traceback.
4. Add the bot to that group, then run:

```bash
python -m answerbot.bot
```

Or `docker compose up -d` if you followed [Docker](#docker) above.

In the group it answers when @mentioned or replied to, or after `/ask` (then
the next message is the question). `/ask <question>` still works in one step.
In a DM it answers only
if you are currently a member of `TELEGRAM_CHAT_ID`; otherwise it declines.
New group messages are appended live; the tail is re-windowed every
`LIVE_REINDEX_EVERY` messages, and every `LIVE_LOOKBACK_HOURS` the last couple
of weeks are rebuilt so recent edits self-heal. Asking a question does **not**
reindex — only those schedules (and `/reindex`) do. `/info` (includes index
size), `/settings` (language: Russian by default, or English), `/cancel`
(stop a running search), and (for admins) `/stats`, `/reindex` /
`/reindex full`, `/resolve` (background name lookup; `retry` / `stop`) are
available. Non-admins see only `/ask`, `/cancel`,
`/settings`, `/info`, and `/help` in the command menu.

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
- **Cursor**: `LLM_PROVIDER=cursor` and `CURSOR_API_KEY` from
  [cursor.com/dashboard/api](https://cursor.com/dashboard/api) (Pro or
  higher). A venv install needs the extra (`pip install 'answerbot[cursor]'`);
  the Docker image already includes it. Defaults to `composer-2.5`, which
  bills the included Cursor Models pool rather than a third-party API.
  This is a local Cursor agent (`tools=[]`, text only) — not an
  OpenAI-compatible `/chat/completions` endpoint.
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

- **`DB_PATH`** (`data/answerbot.db`) — SQLite file for messages, windows, and
  embeddings. In Docker this is `/data/answerbot.db` (the `./data` bind mount).
- **`LOG_LEVEL`** (`INFO`) — logging threshold (`DEBUG`, `INFO`, `WARNING`,
  `ERROR`).
- **`LOG_PATH`** — rotating file next to the database (`answerbot.log` by
  default). Logs also go to stderr. Set to `off` (or `0` / `false` / `none`)
  for stderr only; any other path writes there instead.

### Embeddings

Local. A full `index` is required after changing the model or dimension.

- **`EMBED_MODEL`** (`intfloat/multilingual-e5-small`) — sentence-transformers
  model used to embed windows and queries.
- **`EMBED_DIM`** (`384`) — vector width stored in SQLite. Must match the
  model; wrong values make search silently useless.
- **`EMBED_THREADS`** (`1`) — CPU threads for the local embedding model.
  The bot also warms the model at startup so the first question is not the stall.
- **`HF_TOKEN`** — Hugging Face access token for pulling the embed model
  (gated repos, or Hub rate limits). Also accepted as `HUGGING_FACE_HUB_TOKEN`.
  Used at image build (`docker compose build`) and whenever a model is loaded
  that is not already in the cache.

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

- **`SPEAKER_LABEL`** (`name`) — `name` uses the resolved public name, falling
  back to a stable `User N` alias (never the exporter's contact label). `id`
  renders `User N` even for resolved people. `export` uses the public name,
  falling back to the export/contact label (opt-in; see
  [Or drop names entirely](#or-drop-names-entirely)). Names live in window
  text, so a reindex is required after changing this.

### Retrieval

Hybrid keyword (FTS5) + vector search, merged with reciprocal rank fusion.

- **`TOP_K`** (`10`) — windows ranked by fusion before the excerpt cap.
- **`MIN_K`** (`3`) / **`MAX_K`** (`5`) — excerpts sent to the LLM. Always
  keep at least `MIN_K`; stop early once cosine falls below `COSINE_MIN`;
  never send more than `MAX_K`.
- **`COSINE_MIN`** (`0.7`) — cosine floor for extra excerpts after `MIN_K`.
  `0` disables the cutoff (always send `MAX_K`).
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
  `ollama`, or `cursor`.
- **`ANTHROPIC_API_KEY`**, **`GEMINI_API_KEY`** (or **`GOOGLE_API_KEY`**),
  **`GROQ_API_KEY`**, **`OPENROUTER_API_KEY`**, **`CURSOR_API_KEY`** — key
  for the chosen provider. Unused keys can be left empty.
- **`OPENROUTER_HTTP_REFERER`** / **`OPENROUTER_APP_TITLE`** (`answer-chat-history-bot`)
  — optional OpenRouter attribution headers. Referer is omitted unless set.
- **`ANSWER_MODEL`** — model id. Defaults: Claude `claude-sonnet-5`, Gemini
  `gemini-2.5-flash`, Groq `openai/gpt-oss-20b`, OpenRouter
  `openai/gpt-oss-20b:free`, Cursor `composer-2.5`. Required for Ollama
  (whatever you have pulled). Cursor ids other than Composer/Grok draw
  from the Other Models pool at that model's API price.
- **`ANSWER_MAX_TOKENS`** (`8192`) — Groq/OpenRouter completion budget. On
  reasoning models this covers *thinking plus* the visible answer; 1024 is
  often not enough and comes back as an empty reply.
- **`ANSWER_MAX_REQUEST_TOKENS`** (`0`) — cap on *prompt plus* completion
  for one request. `0` means no extra cap, except Groq which then uses
  `8000` to match on_demand TPM and reserves at most 2048 completion
  tokens so a second question in the same minute is not a 429. Groq
  retries a TPM 429 once after the wait it suggests (up to 20s). Raise
  `ANSWER_MAX_REQUEST_TOKENS` if you are on Dev Tier.
- **`ANSWER_COOLDOWN_SECONDS`** (`8`) — per-user wait between answers in
  the same chat. `0` disables. Admins skip the cooldown.
- **`ANSWER_MAX_PER_USER_PER_HOUR`** (`0`) — sliding-hour cap on LLM
  answers per Telegram user (group and DM share the count). `0` disables.
  Counts only when retrieval found windows, so empty "not in history"
  replies are free. Admins skip it. In-memory; resets when the process
  restarts.
- **`ANSWER_MAX_PER_HOUR`** (`0`) — same window, for the whole bot, so
  many members cannot add up to a drain. `0` disables. Admins skip it.
- **`QUERY_LOG`** (`1`) — persist each question and the retrieved window ids
  locally. `0` / `false` / `off` turns it off.
- **`LLM_TIMEOUT`** (`60`) — seconds to wait for the provider.
- **`OLLAMA_HOST`** (`http://localhost:11434`) — Ollama base URL. From
  Docker, talking to Ollama on the host is
  `http://host.docker.internal:11434`.

### Telegram bot

- **`TELEGRAM_BOT_TOKEN`** — from @BotFather. Required to run the bot.
  Privacy mode must be **off** or the bot sees no group messages.
- **`TELEGRAM_CHAT_ID`** — the one supergroup the bot serves. Use the Bot API
  id (`-100…`). A positive id is stored as `-100<id>`. Search, live ingest,
  and DM access are all pinned to this chat. `/ask` with no question uses
  this chat’s Telegram title in the prompt.
- **`ADMIN_USER_IDS`** — numeric Telegram user ids (space- or
  comma-separated). Those accounts get `Bot is starting`, then
  `Bot started, stats: …` after setup, `Bot is down` DMs and
  ERROR logs with traceback; they can run `/reindex` and skip the answer
  cooldown and hourly quotas. Open a DM with the bot first so it can write
  to you.
- **`GITHUB_REPO`** (`https://github.com/antmaxi/answer-chat-history-bot`)
  — source-code URL shown by `/info`.
- **`DISPLAY_UTC_OFFSET_HOURS`** (`2`) — UTC offset for wall-clock times
  in `/info` and `/stats` (default UTC+2). Valid range is −12 to 14.
- **`LIVE_REINDEX_EVERY`** (`20`) — new messages that may accumulate before
  the open tail is re-windowed.
- **`LIVE_LOOKBACK_HOURS`** (`6`) — how often to rebuild the last
  `UPDATE_LOOKBACK_DAYS` of history so recent edits self-heal. `0` disables
  the periodic pass (tail reindex still runs).
- **`MEMBERSHIP_CACHE_SECONDS`** (`300`) — how long a "is this user in this
  chat?" Bot API lookup is remembered. DMs (`/start`, `/ask`, `/info`,
  `/settings`, and questions) are declined unless `getChatMember`
  says the user is in `TELEGRAM_CHAT_ID`. The UI is Russian by default;
  `/settings` switches between Russian and English (saved per user).

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
