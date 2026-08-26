"""Which chats a question may search, given operator policy and memberships.

`related` routing is intentionally not implemented here: a later pass can take
the membership-allowed list, retrieve per chat, and keep strong-scoring chats
before the final merged search.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence

MemberCheck = Callable[[int, int], Awaitable[bool]]


def is_main_chat(chat_id: int, main_id: int | None) -> bool:
    return main_id is not None and chat_id == main_id


def is_source_chat(chat_id: int, configured: Sequence[int]) -> bool:
    return chat_id in configured


def candidate_chats(*, scope: str, main_id: int, configured: Sequence[int]) -> list[int]:
    """Chats the operator selected before membership filtering.

    `main` is only the control chat. `all` is the configured list, with `main_id`
    first. Never returns None; an empty configured list stays empty.
    """
    if scope == "main":
        return [int(main_id)]
    ids: list[int] = []
    seen: set[int] = set()
    for cid in (main_id, *configured):
        cid = int(cid)
        if cid in seen:
            continue
        ids.append(cid)
        seen.add(cid)
    return ids


def apply_access(
    candidates: Sequence[int],
    *,
    main_id: int,
    access: str,
    memberships: Mapping[int, bool],
) -> list[int]:
    """Keep the main chat; filter secondaries when access is `members`.

    An empty result stays empty (never None / "all chats").
    """
    if access == "all":
        return [int(c) for c in candidates]
    result: list[int] = []
    for cid in candidates:
        cid = int(cid)
        if cid == main_id or memberships.get(cid, False):
            result.append(cid)
    return result


async def resolve_search_chats(
    *,
    user_id: int | None,
    main_id: int,
    configured: Sequence[int],
    scope: str,
    access: str,
    is_member: MemberCheck | None = None,
) -> list[int]:
    """Explicit allow-list for one question. Empty means no chats, not all chats."""
    candidates = candidate_chats(scope=scope, main_id=main_id, configured=configured)
    if scope != "all" or access == "all" or not candidates:
        return candidates
    memberships: dict[int, bool] = {}
    for cid in candidates:
        if cid == main_id:
            continue
        if user_id is None or is_member is None:
            memberships[cid] = False
            continue
        try:
            memberships[cid] = bool(await is_member(user_id, cid))
        except Exception:
            memberships[cid] = False
    return apply_access(
        candidates, main_id=main_id, access=access, memberships=memberships
    )
