"""Operator search-scope policy and membership filtering."""

import asyncio

from answerbot import chat_scope


MAIN = -1001
OTHER = -1002
THIRD = -1003
CONFIGURED = [MAIN, OTHER, THIRD]


def test_secondary_chats_are_ingest_only():
    """Live ingest covers every source; commands stay on the main chat."""
    assert chat_scope.is_main_chat(MAIN, MAIN)
    assert not chat_scope.is_main_chat(OTHER, MAIN)
    assert chat_scope.is_source_chat(MAIN, CONFIGURED)
    assert chat_scope.is_source_chat(OTHER, CONFIGURED)
    assert not chat_scope.is_source_chat(99, CONFIGURED)


def test_candidate_chats_main_is_only_control_chat():
    assert chat_scope.candidate_chats(scope="main", main_id=MAIN, configured=CONFIGURED) == [MAIN]


def test_candidate_chats_all_puts_main_first_and_dedupes():
    assert chat_scope.candidate_chats(
        scope="all", main_id=MAIN, configured=[OTHER, MAIN, THIRD]
    ) == [MAIN, OTHER, THIRD]


def test_candidate_chats_empty_configured_stays_empty_aside_from_main():
    assert chat_scope.candidate_chats(scope="all", main_id=MAIN, configured=[]) == [MAIN]


def test_apply_access_all_keeps_every_candidate():
    assert chat_scope.apply_access(
        CONFIGURED, main_id=MAIN, access="all", memberships={}
    ) == CONFIGURED


def test_apply_access_members_keeps_main_and_memberships():
    assert chat_scope.apply_access(
        CONFIGURED,
        main_id=MAIN,
        access="members",
        memberships={OTHER: True, THIRD: False},
    ) == [MAIN, OTHER]


def test_apply_access_members_with_no_secondaries_is_just_main():
    assert chat_scope.apply_access(
        CONFIGURED, main_id=MAIN, access="members", memberships={}
    ) == [MAIN]


def test_apply_access_empty_candidates_stay_empty():
    assert chat_scope.apply_access([], main_id=MAIN, access="all", memberships={}) == []
    assert chat_scope.apply_access([], main_id=MAIN, access="members", memberships={}) == []


def _resolve(**kwargs):
    return asyncio.run(chat_scope.resolve_search_chats(**kwargs))


def test_resolve_main_scope_does_not_call_membership():
    calls: list[tuple[int, int]] = []

    async def is_member(uid, cid):
        calls.append((uid, cid))
        return True

    got = _resolve(
        user_id=7,
        main_id=MAIN,
        configured=CONFIGURED,
        scope="main",
        access="members",
        is_member=is_member,
    )
    assert got == [MAIN]
    assert calls == []


def test_resolve_all_access_skips_membership_after_main_gate():
    calls: list[tuple[int, int]] = []

    async def is_member(uid, cid):
        calls.append((uid, cid))
        return False

    got = _resolve(
        user_id=7,
        main_id=MAIN,
        configured=CONFIGURED,
        scope="all",
        access="all",
        is_member=is_member,
    )
    assert got == CONFIGURED
    assert calls == []


def test_resolve_all_members_filters_secondaries_and_keeps_stable_order():
    async def is_member(uid, cid):
        assert uid == 7
        return cid == THIRD

    got = _resolve(
        user_id=7,
        main_id=MAIN,
        configured=CONFIGURED,
        scope="all",
        access="members",
        is_member=is_member,
    )
    assert got == [MAIN, THIRD]


def test_resolve_membership_api_error_is_fail_closed():
    async def is_member(uid, cid):
        if cid == OTHER:
            raise RuntimeError("telegram down")
        return True

    got = _resolve(
        user_id=7,
        main_id=MAIN,
        configured=CONFIGURED,
        scope="all",
        access="members",
        is_member=is_member,
    )
    assert got == [MAIN, THIRD]


def test_resolve_without_user_excludes_secondaries_when_members_only():
    got = _resolve(
        user_id=None,
        main_id=MAIN,
        configured=CONFIGURED,
        scope="all",
        access="members",
        is_member=None,
    )
    assert got == [MAIN]


def test_resolve_never_returns_none_for_empty_configured_without_main_in_list():
    got = _resolve(
        user_id=7,
        main_id=MAIN,
        configured=[],
        scope="all",
        access="all",
        is_member=None,
    )
    assert got == [MAIN]
    assert got is not None
