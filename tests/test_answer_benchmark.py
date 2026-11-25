from contextlib import closing
from unittest.mock import Mock

import pytest

from answerbot import answer, answer_benchmark as benchmark, db, retrieve


@pytest.fixture
def conn():
    with closing(db.connect(":memory:")) as conn:
        benchmark.seed(conn)
        yield conn


@pytest.mark.parametrize(("case", "reply"), tuple(zip(benchmark.CASES, (
    "The design review starts at 09:30 [W1].",
    "Экскурсия начинается в 14:20 [W1].",
    None,
    answer.NOT_FOUND,
    "The launch meeting moved to 11:00 [W1].",
))))
def test_answer_cases(conn, case, reply):
    hits = retrieve.search(conn, case.question, chat_id=case.chat_id)
    assert benchmark.message_ids(hits, case.chat_id) == set(case.context_messages)

    llm = Mock()
    llm.complete.return_value = reply
    result = answer.complete_answer(case.question, hits, llm)
    assert result.abstain == (case.expected_time is None)
    assert benchmark.score(case, result) == ()

    if not hits:
        llm.complete.assert_not_called()
    else:
        llm.complete.assert_called_once()
        if case.name == "later-correction":
            prompt = llm.complete.call_args.args[1]
            assert "The launch meeting starts at 10:00." in prompt
            assert "moved from 10:00 to 11:00" in prompt


def test_scorer_rejects_stale_answer_with_correct_citation(conn):
    case = benchmark.CASES[-1]
    hits = retrieve.search(conn, case.question, chat_id=case.chat_id)
    assert benchmark.score(case, answer.Answer("The meeting is at 10:00 [W1].", hits)) == (
        "incorrect fact",
    )
    assert benchmark.score(case, answer.Answer("Either 10:00 or 11:00 [W1].", hits)) == (
        "incorrect fact",
    )
    assert benchmark.score(case, answer.Answer("The meeting is at 11:00.", hits)) == (
        "missing or incorrect citation",
    )
    assert benchmark.score(case, answer.Answer(answer.NOT_FOUND, hits, abstain=True)) == (
        "missed supported answer",
    )


def test_report_distinguishes_wrong_answers_and_model_errors(conn, capsys):
    llm = Mock()
    llm.complete.side_effect = [
        "09:30 [W1].", "14:20 [W1].", "It's at the usual time.", TimeoutError()
    ]
    assert benchmark.run(conn, llm) == 2
    output = capsys.readouterr().out
    assert "FAIL: unsupported answer" in output
    assert "ERROR: answer model call failed (TimeoutError)" in output
    assert "3/5 passed, 1 failed, 1 errors" in output
