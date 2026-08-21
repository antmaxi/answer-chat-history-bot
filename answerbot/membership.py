"""TTL cache for Telegram getChatMember results."""

from __future__ import annotations

import time

# Bot API ChatMember.status values that mean the user is currently in the chat.
_IN_CHAT = {"creator", "administrator", "member"}


def is_chat_member(member) -> bool:
    """True if a getChatMember result is someone still in the chat."""
    status = getattr(member, "status", None)
    if status in _IN_CHAT:
        return True
    if status == "restricted":
        return bool(getattr(member, "is_member", False))
    return False


class MembershipCache:
    """Remember whether a user is in a chat, so DM ACL does not hammer the API."""

    def __init__(self, ttl_seconds: float = 300):
        self.ttl = ttl_seconds
        self._data: dict[tuple[int, int], tuple[float, bool]] = {}

    def get(self, user_id: int, chat_id: int, *, now: float | None = None) -> bool | None:
        now = time.monotonic() if now is None else now
        row = self._data.get((user_id, chat_id))
        if row is None:
            return None
        expires, member = row
        if now >= expires:
            del self._data[(user_id, chat_id)]
            return None
        return member

    def remember(
        self,
        user_id: int,
        chat_id: int,
        is_member: bool,
        *,
        now: float | None = None,
        ttl: float | None = None,
    ) -> None:
        now = time.monotonic() if now is None else now
        self._data[(user_id, chat_id)] = (now + (self.ttl if ttl is None else ttl), is_member)

    def invalidate(self, user_id: int | None = None, chat_id: int | None = None) -> None:
        if user_id is None and chat_id is None:
            self._data.clear()
            return
        drop = [
            k
            for k in self._data
            if (user_id is None or k[0] == user_id) and (chat_id is None or k[1] == chat_id)
        ]
        for k in drop:
            del self._data[k]
