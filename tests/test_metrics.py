from openeval.metrics import ToolSelectionAccuracy
from openeval.models import AgentTrace, EvalTestCase, TraceStep

def test_tool_selection_accuracy():
    metric = ToolSelectionAccuracy()
    
    test_case = EvalTestCase(
        task_id="test-1",
        input="do something",
        expected_tool_calls=[{"tool": "search", "args": {}}, {"tool": "read_file", "args": {}}],
        expected_final_state={},
        expected_output_contains=[],
        max_steps=10,
        timeout_seconds=10.0
    )
    
    steps = [
        TraceStep(step_id=1, type="tool_call", content="", tool_name="read_file", tool_args={}, tool_result=None, timestamp=0.0),
        TraceStep(step_id=2, type="tool_call", content="", tool_name="search", tool_args={}, tool_result=None, timestamp=1.0),
        TraceStep(step_id=3, type="tool_call", content="", tool_name="summarize", tool_args={}, tool_result=None, timestamp=2.0)
    ]
    trace = AgentTrace(task_id="test-1", input="", steps=steps, final_output="done", actual_state={}, metadata={})
    
    result = metric.score(trace, test_case)
    assert result.score == 1.0
    assert result.passed is True

def test_tool_selection_accuracy_empty_expected():
    metric = ToolSelectionAccuracy()
    test_case = EvalTestCase(
        task_id="test-2", input="", expected_tool_calls=[], expected_final_state={},
        expected_output_contains=[], max_steps=10, timeout_seconds=10.0
    )
    
    trace_empty = AgentTrace(task_id="t", input="", steps=[], final_output="", actual_state={}, metadata={})
    assert metric.score(trace_empty, test_case).score == 1.0
    
    trace_one = AgentTrace(task_id="t", input="", steps=[
        TraceStep(step_id=1, type="tool_call", content="", tool_name="search", tool_args={}, tool_result=None, timestamp=0.0)
    ], final_output="", actual_state={}, metadata={})
    assert metric.score(trace_one, test_case).score == 0.0

def test_tool_selection_accuracy_partial():
    metric = ToolSelectionAccuracy()
    test_case = EvalTestCase(
        task_id="test-3", input="", expected_tool_calls=[{"tool": "search"}, {"tool": "search"}, {"tool": "read_file"}], 
        expected_final_state={}, expected_output_contains=[], max_steps=10, timeout_seconds=10.0
    )
    steps = [
        TraceStep(step_id=1, type="tool_call", content="", tool_name="search", tool_args={}, tool_result=None, timestamp=0.0),
        TraceStep(step_id=2, type="tool_call", content="", tool_name="read_file", tool_args={}, tool_result=None, timestamp=1.0),
    ]
    trace = AgentTrace(task_id="t", input="", steps=steps, final_output="", actual_state={}, metadata={})
    result = metric.score(trace, test_case)
    
    assert abs(result.score - (2 / 3)) < 1e-6
    assert result.passed is False

from openeval.metrics import ArgumentCorrectness
import pytest

def test_argument_correctness():
    metric = ArgumentCorrectness()
    
    test_case = EvalTestCase(
        task_id="test-ac-1", input="", 
        expected_tool_calls=[
            {"tool": "search", "args": {"query": "foo"}},
            {"tool": "read_file", "args": {"path": "bar.txt", "lines": 10}}
        ], 
        expected_final_state={}, expected_output_contains=[], max_steps=10, timeout_seconds=10.0
    )
    
    # search called with correct args + extra arg (ignored)
    # read_file called but missing 'lines' and 'path' is wrong
    steps = [
        TraceStep(step_id=1, type="tool_call", content="", tool_name="search", tool_args={"query": "foo", "extra": "baz"}, tool_result=None, timestamp=0.0),
        TraceStep(step_id=2, type="tool_call", content="", tool_name="read_file", tool_args={"path": "wrong.txt"}, tool_result=None, timestamp=1.0),
    ]
    trace = AgentTrace(task_id="t", input="", steps=steps, final_output="", actual_state={}, metadata={})
    
    result = metric.score(trace, test_case)
    # expected args evaluated = 1 (search) + 2 (read_file) = 3
    # correct args = 1 (search.query)
    # score = 1/3
    assert abs(result.score - (1 / 3)) < 1e-6
    assert result.passed is False

def test_argument_correctness_uncalled_exclusion():
    metric = ArgumentCorrectness()
    test_case = EvalTestCase(
        task_id="test-ac-2", input="", 
        expected_tool_calls=[
            {"tool": "search", "args": {"query": "foo"}},
            {"tool": "read_file", "args": {"path": "bar.txt"}}
        ], 
        expected_final_state={}, expected_output_contains=[], max_steps=10, timeout_seconds=10.0
    )
    
    # Only search is called (correctly). read_file is uncalled.
    steps = [
        TraceStep(step_id=1, type="tool_call", content="", tool_name="search", tool_args={"query": "foo"}, tool_result=None, timestamp=0.0)
    ]
    trace = AgentTrace(task_id="t", input="", steps=steps, final_output="", actual_state={}, metadata={})
    
    result = metric.score(trace, test_case)
    # read_file is excluded. evaluated = 1, correct = 1.
    assert result.score == 1.0

def test_argument_correctness_zero_guards():
    metric = ArgumentCorrectness()
    
    # Case 1: Empty expected tools -> 1.0
    tc1 = EvalTestCase(task_id="1", input="", expected_tool_calls=[], expected_final_state={}, expected_output_contains=[], max_steps=10, timeout_seconds=10.0)
    tr1 = AgentTrace(task_id="t", input="", steps=[], final_output="", actual_state={}, metadata={})
    assert metric.score(tr1, tc1).score == 1.0

    # Case 2: Expected tools, but none called -> 0.0
    tc2 = EvalTestCase(task_id="2", input="", expected_tool_calls=[{"tool": "search", "args": {"q": "foo"}}], expected_final_state={}, expected_output_contains=[], max_steps=10, timeout_seconds=10.0)
    tr2 = AgentTrace(task_id="t", input="", steps=[], final_output="", actual_state={}, metadata={})
    assert metric.score(tr2, tc2).score == 0.0

    # Case 3: Expected tools called, but no expected args -> 1.0
    tc3 = EvalTestCase(task_id="3", input="", expected_tool_calls=[{"tool": "search", "args": {}}], expected_final_state={}, expected_output_contains=[], max_steps=10, timeout_seconds=10.0)
    tr3 = AgentTrace(task_id="t", input="", steps=[
        TraceStep(step_id=1, type="tool_call", content="", tool_name="search", tool_args={"extra": "foo"}, tool_result=None, timestamp=0.0)
    ], final_output="", actual_state={}, metadata={})
    assert metric.score(tr3, tc3).score == 1.0

def test_argument_correctness_duplicate_raises():
    metric = ArgumentCorrectness()
    test_case = EvalTestCase(
        task_id="test-ac-dup", input="", 
        expected_tool_calls=[{"tool": "search"}, {"tool": "search"}], 
        expected_final_state={}, expected_output_contains=[], max_steps=10, timeout_seconds=10.0
    )
    trace = AgentTrace(task_id="t", input="", steps=[], final_output="", actual_state={}, metadata={})
    
    with pytest.raises(ValueError, match="does not support duplicate tool names"):
        metric.score(trace, test_case)

from openeval.metrics import StepEfficiency

def test_step_efficiency():
    metric = StepEfficiency()
    
    test_case = EvalTestCase(task_id="te", input="", expected_tool_calls=[{}, {}], expected_final_state={}, expected_output_contains=[], max_steps=10, timeout_seconds=10.0)
    
    # Perfect efficiency (2 optimal, 2 actual -> 1.0)
    steps_perf = [TraceStep(step_id=1, type="tool_call", content="", tool_name="a", tool_args={}, tool_result=None, timestamp=0.0)] * 2
    tr_perf = AgentTrace(task_id="t", input="", steps=steps_perf, final_output="", actual_state={}, metadata={})
    assert metric.score(tr_perf, test_case).score == 1.0
    
    # Inefficient (2 optimal, 4 actual -> 0.5)
    steps_ineff = [TraceStep(step_id=1, type="tool_call", content="", tool_name="a", tool_args={}, tool_result=None, timestamp=0.0)] * 4
    tr_ineff = AgentTrace(task_id="t", input="", steps=steps_ineff, final_output="", actual_state={}, metadata={})
    assert metric.score(tr_ineff, test_case).score == 0.5

    # Under-shooting (2 optimal, 1 actual -> min(1.0, 2/1) -> 1.0)
    steps_under = [TraceStep(step_id=1, type="tool_call", content="", tool_name="a", tool_args={}, tool_result=None, timestamp=0.0)]
    tr_under = AgentTrace(task_id="t", input="", steps=steps_under, final_output="", actual_state={}, metadata={})
    assert metric.score(tr_under, test_case).score == 1.0

def test_step_efficiency_zero_guards():
    metric = StepEfficiency()
    
    # 0 optimal, 0 actual -> 1.0
    tc1 = EvalTestCase(task_id="te", input="", expected_tool_calls=[], expected_final_state={}, expected_output_contains=[], max_steps=10, timeout_seconds=10.0)
    tr1 = AgentTrace(task_id="t", input="", steps=[], final_output="", actual_state={}, metadata={})
    assert metric.score(tr1, tc1).score == 1.0
    
    # 2 optimal, 0 actual -> 0.0
    tc2 = EvalTestCase(task_id="te", input="", expected_tool_calls=[{}, {}], expected_final_state={}, expected_output_contains=[], max_steps=10, timeout_seconds=10.0)
    tr2 = AgentTrace(task_id="t", input="", steps=[], final_output="", actual_state={}, metadata={})
    assert metric.score(tr2, tc2).score == 0.0

from openeval.metrics import GoalCompletionRate

def test_goal_completion_rate_perfect_match():
    metric = GoalCompletionRate()
    tc_perf = EvalTestCase(task_id="1", input="", expected_tool_calls=[], expected_final_state={"a": 1, "b": 2}, expected_output_contains=[], max_steps=10, timeout_seconds=10.0)
    tr_perf = AgentTrace(task_id="t", input="", steps=[], final_output="", actual_state={"a": 1, "b": 2}, metadata={})
    assert metric.score(tr_perf, tc_perf).score == 1.0

def test_goal_completion_rate_superset_ignores_extra_keys():
    metric = GoalCompletionRate()
    tc_perf = EvalTestCase(task_id="1", input="", expected_tool_calls=[], expected_final_state={"a": 1, "b": 2}, expected_output_contains=[], max_steps=10, timeout_seconds=10.0)
    tr_super = AgentTrace(task_id="t", input="", steps=[], final_output="", actual_state={"a": 1, "b": 2, "c": 3}, metadata={})
    assert metric.score(tr_super, tc_perf).score == 1.0

def test_goal_completion_rate_partial_match():
    metric = GoalCompletionRate()
    tc_perf = EvalTestCase(task_id="1", input="", expected_tool_calls=[], expected_final_state={"a": 1, "b": 2}, expected_output_contains=[], max_steps=10, timeout_seconds=10.0)
    tr_part = AgentTrace(task_id="t", input="", steps=[], final_output="", actual_state={"a": 1, "b": 99}, metadata={})
    res_part = metric.score(tr_part, tc_perf)
    assert res_part.score == 0.5
    assert res_part.passed is False

def test_goal_completion_rate_empty_expected():
    metric = GoalCompletionRate()
    tc_empty = EvalTestCase(task_id="2", input="", expected_tool_calls=[], expected_final_state={}, expected_output_contains=[], max_steps=10, timeout_seconds=10.0)
    tr_empty = AgentTrace(task_id="t", input="", steps=[], final_output="", actual_state={"a": 1}, metadata={})
    assert metric.score(tr_empty, tc_empty).score == 1.0

