"""The OpenAI-compatible adapter, tested against an in-memory fake server (no network, no key)."""

import json
import re
import time

import httpx
import pytest
from harness import audit_events, make_policy, tool_results
from langgraph.checkpoint.memory import InMemorySaver

from portcullis.agent import build_agent, initial_state
from portcullis.llm_http import LLMError, OpenAICompatLLM

TOOLS = [
    {
        "name": "check_vendor_availability",
        "description": "Look up vendors by trade.",
        "parameters": {"type": "object", "properties": {"trade": {"type": "string"}}, "required": ["trade"]},
    }
]
SECRET = "sk-test-secret-value"


class FakeServer:
    """Records requests and answers with queued JSON bodies (or a status)."""

    def __init__(self, *answers, status=200):
        self.answers, self.status, self.requests = list(answers), status, []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.status != 200:
            return httpx.Response(self.status, text=f"upstream said: key={SECRET}")
        return httpx.Response(200, json=self.answers.pop(0))


def completion(content=None, tool_calls=None, total=42, usage=True):
    body = {"choices": [{"message": {"role": "assistant", "content": content, "tool_calls": tool_calls}}]}
    if usage:
        body["usage"] = {"prompt_tokens": total - 2, "completion_tokens": 2, "total_tokens": total}
    return body


def tool_call(call_id="call_1", name="check_vendor_availability", arguments='{"trade": "plumber"}'):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def llm(server, **kwargs) -> OpenAICompatLLM:
    options = {"base_url": "https://api.example.test/v1", "model": "m1", "tools": TOOLS, "api_key_env": "TEST_KEY"}
    client = httpx.Client(transport=httpx.MockTransport(server))
    return OpenAICompatLLM(**(options | kwargs), client=client)


@pytest.fixture(autouse=True)
def key(monkeypatch):
    monkeypatch.setenv("TEST_KEY", SECRET)


# --- requests ---


def test_the_request_has_the_model_tools_system_prompt_and_bearer_key():
    server = FakeServer(completion("hi"))
    llm(server, system_prompt="You triage repairs.")([{"role": "user", "content": "leak"}])

    [request] = server.requests
    body = json.loads(request.content)
    assert (request.method, str(request.url)) == ("POST", "https://api.example.test/v1/chat/completions")
    assert request.headers["authorization"] == f"Bearer {SECRET}"
    assert body["model"] == "m1" and body["temperature"] == 0 and body["tool_choice"] == "auto"
    assert body["messages"][:2] == [{"role": "system", "content": "You triage repairs."}, {"role": "user", "content": "leak"}]
    assert body["tools"] == [{"type": "function", "function": TOOLS[0]}]


def test_the_history_is_translated_including_a_call_the_agent_refused_as_malformed():
    server = FakeServer(completion("ok"))
    history = [
        {"role": "user", "content": "leak"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "a", "name": "check_vendor_availability", "args": {"trade": "plumber"}},
                {"id": "b", "invalid": True},  # a stub the agent kept for a malformed call
            ],
        },
        {"role": "tool", "tool_call_id": "a", "content": '{"result": {}}'},
        {"role": "tool", "tool_call_id": "b", "content": '{"error": "invalid_tool_call"}'},
        {"role": "assistant", "content": "done", "tool_calls": []},
    ]
    llm(server)(history)

    sent = json.loads(server.requests[0].content)["messages"]
    assert [m["role"] for m in sent] == ["user", "assistant", "tool", "tool", "assistant"]
    assert sent[1]["tool_calls"][0] == tool_call("a", arguments='{"trade": "plumber"}')
    assert sent[1]["tool_calls"][1]["id"] == "b"  # every tool message still has its call
    assert "tool_calls" not in sent[-1]  # an assistant turn with no calls carries none


def test_no_tools_means_no_tools_field_in_the_request():
    server = FakeServer(completion("hi"))
    llm(server, tools=[])([{"role": "user", "content": "x"}])
    assert "tools" not in json.loads(server.requests[0].content)


# --- responses ---


def test_a_tool_call_answer_becomes_the_agents_format_with_tokens_from_usage():
    server = FakeServer(completion(None, [tool_call()], total=77))
    reply = llm(server)([{"role": "user", "content": "x"}])
    assert reply == {
        "content": "",
        "tool_calls": [{"id": "call_1", "name": "check_vendor_availability", "args": {"trade": "plumber"}}],
        "tokens": 77,
    }


def test_a_plain_answer_has_no_tool_calls():
    reply = llm(FakeServer(completion("All booked.")))([{"role": "user", "content": "x"}])
    assert (reply["content"], reply["tool_calls"]) == ("All booked.", [])


@pytest.mark.parametrize("arguments", ["not json", "[1, 2]", '"string"', ""])
def test_arguments_that_are_not_a_json_object_are_passed_on_unparsed_so_the_agent_refuses_them(arguments):
    reply = llm(FakeServer(completion(None, [tool_call(arguments=arguments)])))([{"role": "user", "content": "x"}])
    assert not isinstance(reply["tool_calls"][0]["args"], dict)


def test_a_call_without_an_id_gets_one():
    bare = {"type": "function", "function": {"name": "check_vendor_availability", "arguments": "{}"}}
    reply = llm(FakeServer(completion(None, [bare, bare])))([{"role": "user", "content": "x"}])
    ids = [c["id"] for c in reply["tool_calls"]]
    assert all(re.fullmatch(r"call-[0-9a-f]{12}", i) for i in ids) and len(set(ids)) == 2
    again = llm(FakeServer(completion(None, [bare])))([{"role": "user", "content": "x"}])
    assert again["tool_calls"][0]["id"] not in ids  # not reused on a later turn either


def test_arguments_sent_as_an_object_instead_of_a_string_are_accepted():
    call = tool_call() | {"function": {"name": "check_vendor_availability", "arguments": {"trade": "plumber"}}}
    reply = llm(FakeServer(completion(None, [call])))([{"role": "user", "content": "x"}])
    assert reply["tool_calls"][0]["args"] == {"trade": "plumber"}


def test_missing_arguments_are_passed_on_as_they_are_so_the_agent_refuses_the_call():
    call = tool_call() | {"function": {"name": "check_vendor_availability"}}
    reply = llm(FakeServer(completion(None, [call])))([{"role": "user", "content": "x"}])
    assert reply["tool_calls"][0]["args"] is None


@pytest.mark.parametrize("raw", ["just text", {"id": "x"}, 7])
def test_a_malformed_tool_call_object_survives_as_a_call_the_agent_refuses(raw):
    reply = llm(FakeServer(completion(None, [raw])))([{"role": "user", "content": "x"}])
    assert reply["tool_calls"][0]["name"] is None and reply["tool_calls"][0]["args"] is None


def test_tool_calls_that_are_not_a_list_are_an_error():
    body = completion("x")
    body["choices"][0]["message"]["tool_calls"] = {"oops": 1}
    with pytest.raises(LLMError, match="tool_calls"):
        llm(FakeServer(body))([{"role": "user", "content": "x"}])


def test_a_200_answer_that_is_not_json_is_an_error():
    with pytest.raises(LLMError, match="not JSON"):
        llm(lambda request: httpx.Response(200, text="<html>maintenance</html>"))([{"role": "user", "content": "x"}])


@pytest.mark.parametrize(
    "body",
    [
        completion("x", usage=False),
        {"choices": []},
        {},
        {"choices": [{"message": "nope"}], "usage": {"total_tokens": 1}},
        completion("x", total=-5),
        {**completion("x"), "usage": {"total_tokens": "12"}},
        {**completion("x"), "usage": {"total_tokens": True}},
    ],
    ids=["no_usage", "no_choices", "empty", "bad_message", "negative_tokens", "string_tokens", "bool_tokens"],
)
def test_an_answer_that_cannot_be_trusted_is_an_error_not_a_guess(body):
    with pytest.raises(LLMError):
        llm(FakeServer(body))([{"role": "user", "content": "x"}])


# --- failures and secrets ---


def test_an_http_error_is_reported_by_status_and_never_echoes_the_body_or_the_key():
    with pytest.raises(LLMError) as raised:
        llm(FakeServer(status=429))([{"role": "user", "content": "x"}])
    assert "429" in str(raised.value) and SECRET not in str(raised.value)


def test_a_timeout_is_an_llm_error():
    def slow(request):
        raise httpx.ReadTimeout("took too long", request=request)

    with pytest.raises(LLMError, match="timed out"):
        llm(slow)([{"role": "user", "content": "x"}])


def test_a_connection_failure_is_an_llm_error_that_does_not_leak_the_url_query():
    def down(request):
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(LLMError, match="could not reach"):
        llm(down)([{"role": "user", "content": "x"}])


def test_a_named_key_variable_that_is_not_set_is_an_error_not_an_unauthenticated_call(monkeypatch):
    monkeypatch.delenv("TEST_KEY")
    server = FakeServer(completion("hi"))
    with pytest.raises(LLMError, match="TEST_KEY"):
        llm(server)([{"role": "user", "content": "x"}])
    assert server.requests == []


def test_a_local_server_needs_no_key():
    server = FakeServer(completion("hi"))
    llm(server, base_url="http://127.0.0.1:11434/v1", api_key_env=None)([{"role": "user", "content": "x"}])
    assert "authorization" not in server.requests[0].headers


@pytest.mark.parametrize(
    "url, env",
    [
        ("http://api.example.test/v1", "TEST_KEY"),  # a key must not travel over plain http
        ("ftp://api.example.test/v1", None),
        ("api.example.test/v1", None),
        ("https://user:pw@api.example.test/v1", None),  # credentials belong in the key variable
    ],
)
def test_an_unsafe_base_url_is_refused_up_front(url, env):
    with pytest.raises(ValueError):
        llm(FakeServer(), base_url=url, api_key_env=env)


def test_the_repr_never_contains_the_key():
    assert SECRET not in repr(llm(FakeServer())) and "TEST_KEY" in repr(llm(FakeServer()))


# --- with the real agent ---


def test_the_adapter_drives_the_guarded_agent_end_to_end(conn, run_id):
    server = FakeServer(
        completion(None, [tool_call("x1", arguments='{"trade": "plumber"}')], total=30),
        completion("Plumber found.", total=20),
    )
    ran = []
    agent = build_agent(
        llm(server), make_policy(), {"check_vendor_availability": lambda args, key: ran.append(args) or {"ok": True}}, conn, InMemorySaver()
    )

    final = agent.invoke(initial_state("acme", run_id, "leak in unit 4"), {"configurable": {"thread_id": run_id}})

    assert ran == [{"trade": "plumber"}]
    assert tool_results(final) == [{"result": {"ok": True}}]
    assert (final["tokens"], final["messages"][-1]["content"]) == (50, "Plumber found.")
    second = json.loads(server.requests[1].content)["messages"]
    assert second[-1]["role"] == "tool" and second[-2]["tool_calls"][0]["function"]["name"] == "check_vendor_availability"
    assert len(audit_events(conn, run_id)) >= 2


def test_a_model_that_returns_garbage_arguments_is_refused_by_the_agent_not_crashed(conn, run_id):
    server = FakeServer(completion(None, [tool_call(arguments="{oops")]), completion("sorry"))
    agent = build_agent(llm(server), make_policy(), {"check_vendor_availability": lambda a, k: {}}, conn, InMemorySaver())

    final = agent.invoke(initial_state("acme", run_id, "x"), {"configurable": {"thread_id": run_id}})

    assert tool_results(final) == [{"error": "invalid_tool_call"}]


# --- a hostile or broken server ---


class Drip(httpx.SyncByteStream):
    """A body that arrives slowly, a chunk at a time."""

    def __init__(self, chunks, pause):
        self.chunks, self.pause = chunks, pause

    def __iter__(self):
        for chunk in self.chunks:
            time.sleep(self.pause)
            yield chunk


def test_an_answer_that_is_too_large_is_refused_while_it_is_read():
    huge = {"choices": [{"message": {"content": "x" * 5000}}], "usage": {"total_tokens": 1}}
    with pytest.raises(LLMError, match="too large"):
        llm(FakeServer(huge), max_response_bytes=1000)([{"role": "user", "content": "x"}])


def test_an_answer_that_is_a_small_download_but_huge_when_decompressed_is_refused():
    import gzip

    bomb = gzip.compress(json.dumps({"choices": [{"message": {"content": "x" * 2_000_000}}], "usage": {"total_tokens": 1}}).encode())
    assert len(bomb) < 5000

    def serve(request):
        return httpx.Response(200, content=bomb, headers={"content-encoding": "gzip"})

    with pytest.raises(LLMError, match="too large"):
        llm(serve)([{"role": "user", "content": "x"}])


def test_an_answer_that_trickles_in_is_cut_off_by_a_total_deadline():
    body = json.dumps(completion("hi")).encode()
    pieces = [body[i : i + 5] for i in range(0, len(body), 5)]

    def serve(request):
        return httpx.Response(200, stream=Drip(pieces, 0.05))

    with pytest.raises(LLMError, match="timed out"):
        llm(serve, timeout=0.2)([{"role": "user", "content": "x"}])


@pytest.mark.parametrize("body", [b"[" * 200_000, b'{"a":' * 100_000], ids=["nested_lists", "nested_objects"])
def test_absurdly_nested_json_is_an_error_not_a_crash(body):
    with pytest.raises(LLMError, match="not JSON"):
        llm(lambda request: httpx.Response(200, content=body))([{"role": "user", "content": "x"}])


def test_too_many_tool_calls_in_one_answer_is_an_error():
    many = [tool_call(f"c{i}") for i in range(17)]
    with pytest.raises(LLMError, match="tool calls"):
        llm(FakeServer(completion(None, many)))([{"role": "user", "content": "x"}])
    ok = [tool_call(f"c{i}") for i in range(16)]
    assert len(llm(FakeServer(completion(None, ok)))([{"role": "user", "content": "x"}])["tool_calls"]) == 16


def test_the_default_client_ignores_proxy_settings_so_the_key_cannot_be_sent_to_one(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example.test:3128")
    adapter = OpenAICompatLLM(base_url="https://api.example.test/v1", model="m", api_key_env="TEST_KEY")
    assert adapter._client._trust_env is False
