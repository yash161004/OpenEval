import pytest
from uuid import uuid4
from datetime import datetime, timezone
from langchain_core.tracers.schemas import Run
from openeval.adapters.langchain import from_langchain_run

def test_clean_multi_tool_run():
    run = Run(
        id=uuid4(),
        name="AgentRun",
        run_type="chain",
        start_time=datetime.now(timezone.utc),
        inputs={"input": "do two things"},
        outputs={"output": "done"},
        child_runs=[
            Run(
                id=uuid4(),
                name="tool_a",
                run_type="tool",
                start_time=datetime.now(timezone.utc),
                end_time=datetime.now(timezone.utc),
                inputs={"args": "val1"},
                outputs={"output": "result1"}
            ),
            Run(
                id=uuid4(),
                name="tool_b",
                run_type="tool",
                start_time=datetime.now(timezone.utc),
                end_time=datetime.now(timezone.utc),
                inputs={"args": "val2"},
                outputs={"output": "result2"}
            )
        ]
    )
    trace = from_langchain_run(run)
    assert len(trace.steps) == 4
    
    assert trace.steps[0].type == "tool_call"
    assert trace.steps[0].tool_name == "tool_a"
    assert trace.steps[0].tool_args == {"args": "val1"}
    
    assert trace.steps[1].type == "tool_result"
    assert trace.steps[1].tool_result == "result1"
    
    assert trace.steps[2].type == "tool_call"
    assert trace.steps[2].tool_name == "tool_b"
    assert trace.steps[2].tool_args == {"args": "val2"}
    
    assert trace.steps[3].type == "tool_result"
    assert trace.steps[3].tool_result == "result2"
    
    assert trace.final_output == "done"

def test_tool_error_run():
    run = Run(
        id=uuid4(),
        name="AgentRun",
        run_type="chain",
        start_time=datetime.now(timezone.utc),
        inputs={"input": "fail"},
        outputs=None,
        error="Agent failed",
        child_runs=[
            Run(
                id=uuid4(),
                name="bad_tool",
                run_type="tool",
                start_time=datetime.now(timezone.utc),
                end_time=datetime.now(timezone.utc),
                inputs={"args": "val"},
                outputs=None,
                error="Tool crashed"
            )
        ]
    )
    trace = from_langchain_run(run)
    assert len(trace.steps) == 2
    assert trace.steps[0].type == "tool_call"
    assert trace.steps[0].tool_name == "bad_tool"
    assert trace.steps[1].type == "tool_result"
    assert trace.steps[1].error == "Tool crashed"
    assert trace.final_output == "Error: Agent failed"

def test_empty_no_tool_calls_run():
    run = Run(
        id=uuid4(),
        name="AgentRun",
        run_type="chain",
        start_time=datetime.now(timezone.utc),
        inputs={"input": "just talk"},
        outputs={"output": "hello"},
        child_runs=[]
    )
    trace = from_langchain_run(run)
    assert len(trace.steps) == 0
    assert trace.final_output == "hello"

def test_nested_run_handling():
    run = Run(
        id=uuid4(),
        name="AgentRun",
        run_type="chain",
        start_time=datetime.now(timezone.utc),
        inputs={"input": "start nested"},
        outputs={"output": "done nested"},
        child_runs=[
            Run(
                id=uuid4(),
                name="parent_tool",
                run_type="tool",
                start_time=datetime.now(timezone.utc),
                end_time=datetime.now(timezone.utc),
                inputs={"args": "parent"},
                outputs={"output": "parent result"},
                child_runs=[
                    Run(
                        id=uuid4(),
                        name="inner_chain",
                        run_type="chain",
                        start_time=datetime.now(timezone.utc),
                        end_time=datetime.now(timezone.utc),
                        inputs={"nested": "call"},
                        outputs={"result": "inner result"},
                        child_runs=[
                            Run(
                                id=uuid4(),
                                name="child_tool",
                                run_type="tool",
                                start_time=datetime.now(timezone.utc),
                                end_time=datetime.now(timezone.utc),
                                inputs={"args": "child"},
                                outputs={"output": "child result"}
                            )
                        ]
                    )
                ]
            )
        ]
    )
    trace = from_langchain_run(run)
    # The expected output will have the parent_tool call, then child_tool call, then child_tool result, then parent_tool result.
    # Non-tool runs like inner_chain are completely omitted from the trace.
    assert len(trace.steps) == 4
    
    assert trace.steps[0].type == "tool_call"
    assert trace.steps[0].tool_name == "parent_tool"
    
    assert trace.steps[1].type == "tool_call"
    assert trace.steps[1].tool_name == "child_tool"
    
    assert trace.steps[2].type == "tool_result"
    assert trace.steps[2].tool_name == "child_tool"
    assert trace.steps[2].tool_result == "child result"
    
    assert trace.steps[3].type == "tool_result"
    assert trace.steps[3].tool_name == "parent_tool"
    assert trace.steps[3].tool_result == "parent result"

