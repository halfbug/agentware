from agentware.core.shared import get_shared_value, merge_shared_variables, set_shared_value


def test_set_and_get_shared():
    update = set_shared_value("foo", "bar")
    merged = merge_shared_variables([], update)
    assert get_shared_value(merged, "foo") == "bar"
