import pytest

import backend
from backend import AppError


# --- parse_response -----------------------------------------------------

def test_plain_message_has_no_code():
    parsed = backend.parse_response("Which table holds the order rows?")
    assert parsed["response_type"] == "message"
    assert parsed["blocks"] == []
    assert parsed["code"] is None


def test_single_block_keeps_prose_as_instructions():
    raw = "Drop this in as a helper.\n```python\ndef add(a, b):\n    return a + b\n```\n"
    parsed = backend.parse_response(raw)
    assert parsed["response_type"] == "code"
    assert parsed["blocks"] == [{"language": "Python", "code": "def add(a, b):\n    return a + b", "truncated": False}]
    assert parsed["instructions"] == "Drop this in as a helper."


def test_every_block_is_returned_not_just_the_first():
    raw = (
        "First create the table:\n```sql\nCREATE TABLE t (id int);\n```\n"
        "Then query it:\n```sql\nSELECT * FROM t;\n```\n"
    )
    parsed = backend.parse_response(raw)
    assert [b["code"] for b in parsed["blocks"]] == ["CREATE TABLE t (id int);", "SELECT * FROM t;"]
    assert parsed["instructions"] == "First create the table:\nThen query it:"
    assert parsed["code"] == "CREATE TABLE t (id int);"


def test_language_comes_from_the_fence_tag():
    parsed = backend.parse_response("```py\nimport os\n```")
    assert parsed["blocks"][0]["language"] == "Python"
    parsed = backend.parse_response("```postgresql\nSELECT 1;\n```")
    assert parsed["blocks"][0]["language"] == "SQL"


def test_untagged_fence_falls_back_to_sniffing():
    parsed = backend.parse_response("```\nSELECT name FROM users WHERE id = 1;\n```")
    assert parsed["blocks"][0]["language"] == "SQL"
    parsed = backend.parse_response("```\ndef f():\n    pass\n```")
    assert parsed["blocks"][0]["language"] == "Python"


def test_truncated_block_is_recovered_and_flagged():
    parsed = backend.parse_response("Here you go:\n```python\ndef long_function():\n    x = 1")
    assert parsed["response_type"] == "code"
    assert parsed["blocks"][0]["truncated"] is True
    assert "def long_function" in parsed["blocks"][0]["code"]
    assert parsed["instructions"] == "Here you go:"


def test_empty_fence_is_not_a_code_answer():
    parsed = backend.parse_response("Nothing to add.\n```\n```")
    assert parsed["response_type"] == "message"


# --- make_title ---------------------------------------------------------

def test_title_collapses_whitespace_and_truncates():
    assert backend.make_title("  how   do\nI join?  ") == "how do I join?"
    long = "x" * 100
    title = backend.make_title(long)
    assert len(title) == backend.TITLE_MAX_LEN
    assert title.endswith("…")


def test_blank_title_gets_a_placeholder():
    assert backend.make_title("   ") == "New conversation"


# --- validate_credentials -----------------------------------------------

@pytest.mark.parametrize("username", ["ab", "a" * 33, "has space", "bad!char", ""])
def test_bad_usernames_are_rejected(username):
    with pytest.raises(AppError):
        backend.validate_credentials(username, "a-long-enough-password")


def test_short_password_is_rejected():
    with pytest.raises(AppError, match="at least"):
        backend.validate_credentials("alice", "short")


def test_password_equal_to_username_is_rejected():
    name = "developer01"
    with pytest.raises(AppError, match="same as the username"):
        backend.validate_credentials(name, name)


def test_overlong_password_is_rejected_rather_than_silently_truncated():
    # bcrypt only reads the first 72 bytes, so a longer password must not be accepted.
    with pytest.raises(AppError, match="at most"):
        backend.validate_credentials("alice", "p" * 200)


def test_bad_email_is_rejected_but_none_is_fine():
    with pytest.raises(AppError, match="email"):
        backend.validate_credentials("alice", "a-long-enough-password", "not-an-email")
    backend.validate_credentials("alice", "a-long-enough-password", None)
    backend.validate_credentials("alice", "a-long-enough-password", "alice@example.com")


# --- validate_upload ----------------------------------------------------

def test_supported_extensions_pass_and_path_is_stripped():
    result = backend.validate_upload("/etc/../tmp/query.sql", "SELECT 1;")
    assert result == {"name": "query.sql", "content": "SELECT 1;"}
    assert backend.validate_upload("script.py", "print(1)")["name"] == "script.py"


@pytest.mark.parametrize("name", ["notes.txt", "main.js", "Main.java", "archive.tar.gz", "noextension"])
def test_other_languages_are_rejected(name):
    with pytest.raises(AppError, match="Only"):
        backend.validate_upload(name, "content")


def test_oversized_upload_is_rejected():
    too_big = "x" * (backend.MAX_UPLOAD_BYTES + 1)
    with pytest.raises(AppError, match="limit"):
        backend.validate_upload("big.py", too_big)


def test_empty_upload_is_rejected():
    with pytest.raises(AppError, match="empty"):
        backend.validate_upload("blank.py", "   \n")


def test_no_attachment_is_not_an_error():
    assert backend.validate_upload(None, None) is None
    assert backend.validate_upload("x.py", None) is None


# --- tokens -------------------------------------------------------------

def test_token_round_trip_carries_the_admin_flag():
    token = backend.create_token(7, "alice", True)
    claims = backend.decode_token(token)
    assert claims["user_id"] == 7
    assert claims["username"] == "alice"
    assert claims["is_admin"] is True


def test_tampered_and_empty_tokens_are_rejected():
    token = backend.create_token(7, "alice", False)
    assert backend.decode_token(token + "x") is None
    assert backend.decode_token("") is None
    assert backend.decode_token("not.a.token") is None


def test_token_signed_with_another_key_is_rejected():
    import jwt

    forged = jwt.encode({"user_id": 1, "username": "root", "is_admin": True}, "another-key" * 4, algorithm="HS256")
    assert backend.decode_token(forged) is None


# --- prompt building ----------------------------------------------------

def test_file_prompt_is_used_only_when_a_file_is_attached():
    without = backend.build_prompt("write a parser", None, [])
    assert without[0]["content"].startswith(backend.CODE_SYSTEM_PROMPT[:40])

    with_file = backend.build_prompt("explain this", None, [], uploaded_file={"name": "a.py", "content": "x = 1"})
    assert "UPLOADED FILE" in with_file[1]["content"]
    assert with_file[0]["content"].startswith(backend.FILE_EDIT_SYSTEM_PROMPT[:40])


def test_prompt_states_the_python_sql_scope():
    system = backend.build_prompt("anything", None, [])[0]["content"]
    assert "only support Python and SQL" in system


def test_memory_and_examples_appear_in_the_user_message():
    user_message = backend.build_prompt(
        "now add an index",
        long_term_summary="Working on an orders table.",
        short_term=[{"question": "create a table", "answer": "CREATE TABLE ..."}],
        golden_example={"question": "g-q", "answer": "g-a"},
        flagged_answer={"question": "f-q", "answer": "f-a", "reason": "wrong dialect"},
    )[1]["content"]

    assert "Working on an orders table." in user_message
    assert "create a table" in user_message
    assert "GOOD EXAMPLE" in user_message and "g-a" in user_message
    assert "BAD EXAMPLE" in user_message and "wrong dialect" in user_message
    # The live question must come last so the model answers it, not the examples.
    assert user_message.rindex("[CURRENT QUESTION]") > user_message.rindex("BAD EXAMPLE")


# --- GPU client ---------------------------------------------------------

def test_chat_flattens_messages_into_the_proxy_prompt_format():
    sent = {}

    class FakeClient(backend.GPUApiClient):
        def infer(self, prompt, **kwargs):
            sent["prompt"] = prompt
            sent["kwargs"] = kwargs
            return "done"

    result = FakeClient(api_key="k", proxy_url="http://x/v1/infer").chat(
        [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hello"}],
        max_new_tokens=16,
    )
    assert result == "done"
    assert sent["prompt"] == "SYSTEM:\nbe brief\n\nUSER:\nhello\n\nASSISTANT:\n"
    assert sent["kwargs"] == {"max_new_tokens": 16}


def test_api_key_header_is_omitted_when_no_key_is_configured():
    with_key = backend.GPUApiClient(api_key="abc", proxy_url="http://x")
    without = backend.GPUApiClient(api_key="", proxy_url="http://x")
    assert with_key._headers["X-API-Key"] == "abc"
    assert "X-API-Key" not in without._headers


def test_pgvector_literal_round_trips_as_floats():
    assert backend._to_pgvector([0.5, -1.0, 0.0]) == "[0.5,-1.0,0.0]"
