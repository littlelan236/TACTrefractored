from tact.artifacts import write_json
from tact.research.trajectories import TrajectoryReader


def response(ident, prompt=7, completion=3):
    return {"id": ident, "usage": {"prompt_tokens": prompt, "completion_tokens": completion}}


def test_format_error_usage_and_multiple_tool_observations_are_preserved(tmp_path):
    path = tmp_path / "x.traj.json"
    write_json(path, {"info": {"model_stats": {"api_calls": 2}}, "messages": [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        {"role": "user", "content": "format error", "extra": {"response": response("bad")}},
        {"role": "assistant", "reasoning_content": "reason", "content": "", "tool_calls": [{}, {}],
         "extra": {"response": response("good")}},
        {"role": "tool", "content": "first"}, {"role": "tool", "content": "second"},
    ]})
    reader = TrajectoryReader(path)
    steps = list(reader.steps())
    assert len(steps) == 1
    assert steps[0]["step"] == 0 and steps[0]["message_index"] == 3
    assert len(steps[0]["observations"]) == 2
    assert reader.usage()["total_tokens"] == 20
    assert reader.usage()["response_coverage_complete"]
    assert "extra" not in reader.prepared_prefix(3)[2]


def test_unknown_response_usage_is_never_reported_as_complete(tmp_path):
    path = tmp_path / "x.traj.json"
    write_json(path, {"info": {"model_stats": {"api_calls": 2}}, "messages": [
        {"role": "assistant", "extra": {"response": response("one")}},
        {"role": "user", "extra": {"response": "unserializable response"}},
    ]})
    usage = TrajectoryReader(path).usage()
    assert usage["missing_usage"] == 1
    assert not usage["response_coverage_complete"]
