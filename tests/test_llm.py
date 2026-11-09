import pytest

from portcullis.llm import CassetteLLM, CassetteMiss, FakeLLM, ScriptExhausted

HELLO = [{"role": "user", "content": "hello"}]


def test_fake_llm_returns_scripted_responses_in_order_and_records_calls():
    llm = FakeLLM([{"content": "one"}, {"content": "two"}])

    assert llm(HELLO) == {"content": "one"}
    assert llm(HELLO) == {"content": "two"}
    assert llm.calls == [HELLO, HELLO]


def test_fake_llm_raises_when_script_runs_out():
    llm = FakeLLM([{"content": "only"}])
    llm(HELLO)

    with pytest.raises(ScriptExhausted):
        llm(HELLO)


def test_replay_miss_raises_instead_of_calling_the_model(tmp_path):
    cassette = CassetteLLM(FakeLLM([{"content": "x"}]), tmp_path / "c.json")

    with pytest.raises(CassetteMiss):
        cassette(HELLO)


def test_record_saves_response_and_replay_serves_it_without_the_inner_model(tmp_path):
    path = tmp_path / "c.json"
    recorder = CassetteLLM(FakeLLM([{"content": "hi"}]), path, record=True)
    assert recorder(HELLO) == {"content": "hi"}

    inner = FakeLLM([])  # any call would raise ScriptExhausted
    replayer = CassetteLLM(inner, path)

    assert replayer(HELLO) == {"content": "hi"}
    assert inner.calls == []


def test_recording_a_second_request_keeps_the_first(tmp_path):
    path = tmp_path / "c.json"
    other = [{"role": "user", "content": "other"}]
    recorder = CassetteLLM(FakeLLM([{"content": "one"}, {"content": "two"}]), path, record=True)
    recorder(HELLO)
    recorder(other)

    replayer = CassetteLLM(FakeLLM([]), path)

    assert replayer(HELLO) == {"content": "one"}
    assert replayer(other) == {"content": "two"}


def test_record_mode_replays_hits_without_calling_the_model_again(tmp_path):
    inner = FakeLLM([{"content": "hi"}])
    recorder = CassetteLLM(inner, tmp_path / "c.json", record=True)

    recorder(HELLO)
    recorder(HELLO)

    assert len(inner.calls) == 1


def test_different_messages_do_not_share_a_recording(tmp_path):
    path = tmp_path / "c.json"
    CassetteLLM(FakeLLM([{"content": "hi"}]), path, record=True)(HELLO)

    with pytest.raises(CassetteMiss):
        CassetteLLM(FakeLLM([]), path)([{"role": "user", "content": "other"}])


def test_key_ignores_dict_key_order(tmp_path):
    path = tmp_path / "c.json"
    CassetteLLM(FakeLLM([{"content": "hi"}]), path, record=True)(
        [{"role": "user", "content": "hello"}]
    )

    reordered = [{"content": "hello", "role": "user"}]

    assert CassetteLLM(FakeLLM([]), path)(reordered) == {"content": "hi"}


def test_model_name_is_part_of_the_key(tmp_path):
    path = tmp_path / "c.json"
    CassetteLLM(FakeLLM([{"content": "hi"}]), path, record=True, model="a")(HELLO)

    with pytest.raises(CassetteMiss):
        CassetteLLM(FakeLLM([]), path, model="b")(HELLO)
