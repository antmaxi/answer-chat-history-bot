#!/bin/sh
# Docker creates missing bind-mount dirs as root. The bot runs as uid 1000
# and must be able to write SQLite + logs under /data.
set -e
mkdir -p /data
if [ "$(id -u)" = "0" ]; then
  chown -R answerbot:answerbot /data
  exec gosu answerbot "$@"
fi
exec "$@"
