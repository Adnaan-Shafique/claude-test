"""Integration tests against a real PostgreSQL database.

Covers the parts that only a database can prove: the approval workflow, login
throttling, conversation ownership, and the conversation-scoped memory window.

The tests are skipped unless FORGE_TEST_DSN-style variables point at a scratch database
(see tests/README.md). pgvector is not required: the similarity-search helpers are the
only callers of the vector operators and they are stubbed out here, since what is being
tested is ownership and scoping, not nearest-neighbour search.
"""

import os

import pytest

import backend
from backend import AppError

pytestmark = pytest.mark.skipif(
    not os.environ.get("FORGE_TEST_DB"),
    reason="set FORGE_TEST_DB=1 plus POSTGRES_* to run integration tests",
)


@pytest.fixture(autouse=True)
def clean_db(monkeypatch):
    """Truncate between tests and stub the pgvector-backed example lookups."""
    with backend.db_cursor(commit=True) as cur:
        cur.execute(
            "TRUNCATE coding_agent_schema.users, coding_agent_schema.conversations, "
            "coding_agent_schema.messages, coding_agent_schema.short_term_memory, "
            "coding_agent_schema.long_term_memory, coding_agent_schema.feedback, "
            "coding_agent_schema.golden_examples, coding_agent_schema.flagged_answers "
            "RESTART IDENTITY CASCADE"
        )

    monkeypatch.setattr(backend, "find_golden_example", lambda user_id, question: None)
    monkeypatch.setattr(backend, "find_flagged_answer", lambda user_id, question: None)
    monkeypatch.setattr(backend, "store_golden_example", lambda *a, **k: None)
    monkeypatch.setattr(backend, "store_flagged_answer", lambda *a, **k: None)

    # Reset the in-process throttles so one test's failed logins cannot lock out another.
    backend._login_limiter._hits.clear()
    backend._register_limiter._hits.clear()
    yield


PASSWORD = "correct-horse-battery"


def make_user(name, *, active=True, admin=False):
    return backend.register_user(
        name, PASSWORD, f"{name}@example.com", client_ip=f"ip-{name}", is_active=active, is_admin=admin
    )


def stub_model(monkeypatch, answer="```python\nprint(1)\n```"):
    """Answer every generation with fixed text, so no GPU proxy is needed."""
    monkeypatch.setattr(backend, "generate_code", lambda *a, **k: answer)
    monkeypatch.setattr(backend, "summarize_qa", lambda existing, q, a: f"summary after: {q}")


# --- registration & approval -------------------------------------------

def test_new_registration_is_inactive_and_cannot_log_in():
    backend.register_user("newbie", PASSWORD, None, client_ip="1.1.1.1")
    with pytest.raises(AppError, match="awaiting administrator approval"):
        backend.login_user("newbie", PASSWORD, client_ip="1.1.1.1")


def test_admin_approval_lets_the_user_log_in():
    admin = make_user("root", admin=True)
    backend.register_user("newbie", PASSWORD, None, client_ip="1.1.1.1")

    pending = [u for u in backend.list_users(admin["user_id"]) if not u["is_active"]]
    assert [u["username"] for u in pending] == ["newbie"]

    backend.set_user_active(admin["user_id"], pending[0]["id"], True)
    result = backend.login_user("newbie", PASSWORD, client_ip="1.1.1.1")
    assert result["username"] == "newbie"
    assert result["is_admin"] is False


def test_duplicate_username_is_rejected_case_insensitively():
    make_user("alice")
    with pytest.raises(AppError, match="already taken"):
        backend.register_user("ALICE", PASSWORD, None, client_ip="2.2.2.2")


def test_non_admin_cannot_list_or_approve_users():
    plain = make_user("plain")
    target = backend.register_user("pending", PASSWORD, None, client_ip="3.3.3.3")
    with pytest.raises(AppError, match="Administrator"):
        backend.list_users(plain["user_id"])
    with pytest.raises(AppError, match="Administrator"):
        backend.set_user_active(plain["user_id"], target["user_id"], True)


def test_admin_cannot_lock_themselves_out():
    admin = make_user("root", admin=True)
    with pytest.raises(AppError, match="your own account"):
        backend.set_user_active(admin["user_id"], admin["user_id"], False)
    with pytest.raises(AppError, match="your own administrator role"):
        backend.set_user_admin(admin["user_id"], admin["user_id"], False)


def test_suspending_a_user_invalidates_their_live_session():
    admin = make_user("root", admin=True)
    victim = make_user("victim")
    token = backend.login_user("victim", PASSWORD, client_ip="4.4.4.4")["token"]
    assert backend.resolve_user(token)["username"] == "victim"

    backend.set_user_active(admin["user_id"], victim["user_id"], False)
    # The token is still cryptographically valid; the live row is what revokes access.
    assert backend.decode_token(token) is not None
    assert backend.resolve_user(token) is None


def test_wrong_password_is_rejected():
    make_user("alice")
    with pytest.raises(AppError, match="Invalid username or password"):
        backend.login_user("alice", "wrong-password-entirely", client_ip="5.5.5.5")


def test_repeated_failures_are_throttled():
    make_user("alice")
    for _ in range(backend.LOGIN_MAX_ATTEMPTS):
        with pytest.raises(AppError):
            backend.login_user("alice", "wrong-password-entirely", client_ip="6.6.6.6")
    with pytest.raises(AppError, match="Too many failed login attempts"):
        backend.login_user("alice", PASSWORD, client_ip="6.6.6.6")


# --- conversation ownership --------------------------------------------

def test_a_user_cannot_read_another_users_conversation(monkeypatch):
    stub_model(monkeypatch)
    alice, bob = make_user("alice"), make_user("bob")

    result = backend.ask_logic(alice["user_id"], "how do I join two tables?")
    conv_id = result["conversation_id"]

    assert len(backend.get_conversation_messages(alice["user_id"], conv_id)) == 1
    with pytest.raises(AppError, match="not available"):
        backend.get_conversation_messages(bob["user_id"], conv_id)


def test_a_user_cannot_append_to_another_users_conversation(monkeypatch):
    stub_model(monkeypatch)
    alice, bob = make_user("alice"), make_user("bob")
    conv_id = backend.ask_logic(alice["user_id"], "first question")["conversation_id"]

    with pytest.raises(AppError, match="not available"):
        backend.ask_logic(bob["user_id"], "sneaking in", conversation_id=conv_id)

    # Nothing was written, and no stray memory row was left behind either.
    assert len(backend.get_conversation_messages(alice["user_id"], conv_id)) == 1
    assert backend.get_short_term_history(bob["user_id"], conv_id) == []


def test_sidebar_lists_only_your_own_conversations(monkeypatch):
    stub_model(monkeypatch)
    alice, bob = make_user("alice"), make_user("bob")
    backend.ask_logic(alice["user_id"], "alice question")
    backend.ask_logic(bob["user_id"], "bob question")

    assert [c["title"] for c in backend.list_conversations_for_user(alice["user_id"])] == ["alice question"]
    assert [c["title"] for c in backend.list_conversations_for_user(bob["user_id"])] == ["bob question"]


def test_retention_prunes_old_conversations(monkeypatch):
    stub_model(monkeypatch)
    alice = make_user("alice")
    old_id = backend.ask_logic(alice["user_id"], "ancient question")["conversation_id"]
    backend.ask_logic(alice["user_id"], "recent question")

    with backend.db_cursor(commit=True) as cur:
        cur.execute(
            "UPDATE coding_agent_schema.conversations SET created_at = NOW() - make_interval(days => %s) WHERE id = %s",
            (backend.CONVERSATION_RETENTION_DAYS + 1, old_id),
        )

    titles = [c["title"] for c in backend.list_conversations_for_user(alice["user_id"])]
    assert titles == ["recent question"]


# --- memory scoping ----------------------------------------------------

def test_short_term_memory_does_not_leak_between_conversations(monkeypatch):
    stub_model(monkeypatch)
    alice = make_user("alice")

    first = backend.ask_logic(alice["user_id"], "question in chat one")["conversation_id"]
    second = backend.ask_logic(alice["user_id"], "question in chat two")["conversation_id"]
    assert first != second

    assert [r["question"] for r in backend.get_short_term_history(alice["user_id"], first)] == ["question in chat one"]
    assert [r["question"] for r in backend.get_short_term_history(alice["user_id"], second)] == ["question in chat two"]


def test_overflowing_the_window_folds_the_oldest_into_the_summary(monkeypatch):
    stub_model(monkeypatch)
    alice = make_user("alice")

    conv_id = backend.ask_logic(alice["user_id"], "question 0")["conversation_id"]
    for i in range(1, backend.SHORT_TERM_LIMIT + 2):
        backend.ask_logic(alice["user_id"], f"question {i}", conversation_id=conv_id)

    window = backend.get_short_term_history(alice["user_id"], conv_id)
    assert len(window) <= backend.SHORT_TERM_LIMIT
    assert "question 0" not in [r["question"] for r in window]
    assert backend.get_long_term_summary(alice["user_id"]) is not None


def test_a_failing_summarizer_does_not_fail_the_answer(monkeypatch):
    import requests

    stub_model(monkeypatch)
    monkeypatch.setattr(
        backend, "summarize_qa",
        lambda *a, **k: (_ for _ in ()).throw(requests.ConnectionError("proxy down")),
    )
    alice = make_user("alice")

    conv_id = backend.ask_logic(alice["user_id"], "question 0")["conversation_id"]
    for i in range(1, backend.SHORT_TERM_LIMIT + 2):
        result = backend.ask_logic(alice["user_id"], f"question {i}", conversation_id=conv_id)
        assert result["code"] == "print(1)"

    # The summary never got written, but every answer still came back.
    assert backend.get_long_term_summary(alice["user_id"]) is None


# --- ask_logic behaviour ------------------------------------------------

def test_upload_is_validated_before_anything_is_written(monkeypatch):
    stub_model(monkeypatch)
    alice = make_user("alice")

    with pytest.raises(AppError, match="Only"):
        backend.ask_logic(alice["user_id"], "explain this", file_name="app.js", file_content="let x = 1;")

    assert backend.list_conversations_for_user(alice["user_id"]) == []


def test_a_failed_generation_leaves_no_empty_conversation(monkeypatch):
    import requests

    monkeypatch.setattr(
        backend, "generate_code",
        lambda *a, **k: (_ for _ in ()).throw(requests.ConnectionError("refused")),
    )
    alice = make_user("alice")

    with pytest.raises(AppError):
        backend.ask_logic(alice["user_id"], "a question that will fail")

    # No half-written conversation in the sidebar, and no answerless memory row.
    assert backend.list_conversations_for_user(alice["user_id"]) == []
    with backend.db_cursor() as cur:
        cur.execute("SELECT count(*) AS n FROM coding_agent_schema.short_term_memory")
        assert cur.fetchone()["n"] == 0


def test_a_failure_does_not_disturb_an_existing_conversation(monkeypatch):
    import requests

    stub_model(monkeypatch)
    alice = make_user("alice")
    conv_id = backend.ask_logic(alice["user_id"], "the good question")["conversation_id"]

    monkeypatch.setattr(
        backend, "generate_code",
        lambda *a, **k: (_ for _ in ()).throw(requests.ConnectionError("refused")),
    )
    with pytest.raises(AppError):
        backend.ask_logic(alice["user_id"], "the failing question", conversation_id=conv_id)

    messages = backend.get_conversation_messages(alice["user_id"], conv_id)
    assert [m["question"] for m in messages] == ["the good question"]


def test_empty_model_response_is_reported_not_stored(monkeypatch):
    monkeypatch.setattr(backend, "generate_code", lambda *a, **k: "   ")
    alice = make_user("alice")
    with pytest.raises(AppError, match="empty response"):
        backend.ask_logic(alice["user_id"], "say nothing")


def test_unreachable_proxy_surfaces_a_readable_message(monkeypatch):
    import requests

    monkeypatch.setattr(
        backend, "generate_code",
        lambda *a, **k: (_ for _ in ()).throw(requests.ConnectionError("refused")),
    )
    alice = make_user("alice")
    with pytest.raises(AppError, match="unreachable"):
        backend.ask_logic(alice["user_id"], "anything")


def test_file_name_is_stored_and_replayed_with_the_conversation(monkeypatch):
    stub_model(monkeypatch)
    alice = make_user("alice")

    result = backend.ask_logic(
        alice["user_id"], "what does this do?", file_name="etl.sql", file_content="SELECT 1;"
    )
    replayed = backend.get_conversation_messages(alice["user_id"], result["conversation_id"])
    assert replayed[0]["file_name"] == "etl.sql"
    assert replayed[0]["blocks"][0]["code"] == "print(1)"


def test_feedback_is_recorded_against_the_voting_user(monkeypatch):
    stub_model(monkeypatch)
    alice = make_user("alice")
    result = backend.ask_logic(alice["user_id"], "a question")

    backend.submit_feedback(alice["user_id"], result["question"], result["answer"], "up", None)
    with backend.db_cursor() as cur:
        cur.execute("SELECT user_id, vote FROM coding_agent_schema.feedback")
        rows = cur.fetchall()
    assert rows == [{"user_id": alice["user_id"], "vote": "up"}]

    with pytest.raises(AppError, match="Invalid vote"):
        backend.submit_feedback(alice["user_id"], "q", "a", "sideways", None)
