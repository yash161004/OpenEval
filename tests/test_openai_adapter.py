import pytest
from openeval.adapters.openai import from_openai_messages

def test_clean_multi_tool_call():
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "func_a", "arguments": '{"x": 1}'}},
            {"id": "call_2", "type": "function", "function": {"name": "func_b", "arguments": '{"y": 2}'}}
        ]},
        {"role": "tool", "tool_call_id": "call_1", "content": "res_a"},
        {"role": "tool", "tool_call_id": "call_2", "content": "res_b"},
        {"role": "assistant", "content": "done"}
    ]
    trace = from_openai_messages(messages)
    assert len(trace.steps) == 4
    
    assert trace.steps[0].type == "tool_call"
    assert trace.steps[0].tool_name == "func_a"
    assert trace.steps[0].tool_args == {"x": 1}
    assert trace.steps[0].error is None
    
    assert trace.steps[1].type == "tool_call"
    assert trace.steps[1].tool_name == "func_b"
    assert trace.steps[1].tool_args == {"y": 2}
    
    assert trace.steps[2].type == "tool_result"
    assert trace.steps[2].tool_name == "func_a"
    assert trace.steps[2].tool_result == "res_a"
    
    assert trace.steps[3].type == "tool_result"
    assert trace.steps[3].tool_name == "func_b"
    assert trace.steps[3].tool_result == "res_b"
    
    assert trace.input == "hello"
    assert trace.final_output == "done"

def test_dangling_tool_call():
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "func_a", "arguments": '{"x": 1}'}}
        ]}
    ]
    trace = from_openai_messages(messages)
    assert len(trace.steps) == 1
    assert trace.steps[0].type == "tool_call"
    assert trace.steps[0].tool_name == "func_a"
    
def test_malformed_json_arguments():
    messages = [
        {"role": "assistant", "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "func_a", "arguments": '{"x": 1'}}
        ]}
    ]
    trace = from_openai_messages(messages)
    assert len(trace.steps) == 1
    assert trace.steps[0].type == "tool_call"
    assert trace.steps[0].tool_name == "func_a"
    assert "raw" in trace.steps[0].tool_args
    assert trace.steps[0].tool_args["raw"] == '{"x": 1'
    assert trace.steps[0].error is not None
    assert "Expecting" in trace.steps[0].error or "Unterminated" in trace.steps[0].error

def test_empty_message_list():
    trace = from_openai_messages([])
    assert len(trace.steps) == 0
    assert trace.input == ""
    assert trace.final_output == ""

def test_out_of_order_tool_results():
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "func_a", "arguments": '{"x": 1}'}},
            {"id": "call_2", "type": "function", "function": {"name": "func_b", "arguments": '{"y": 2}'}}
        ]},
        {"role": "tool", "tool_call_id": "call_2", "content": "res_b"},
        {"role": "tool", "tool_call_id": "call_1", "content": "res_a"}
    ]
    trace = from_openai_messages(messages)
    assert len(trace.steps) == 4
    
    assert trace.steps[2].type == "tool_result"
    assert trace.steps[2].tool_name == "func_b"
    assert trace.steps[2].tool_result == "res_b"
    
    assert trace.steps[3].type == "tool_result"
    assert trace.steps[3].tool_name == "func_a"
    assert trace.steps[3].tool_result == "res_a"
