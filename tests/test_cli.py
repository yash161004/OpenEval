import json
import pytest
from pathlib import Path
from typer.testing import CliRunner
from openeval.cli import app

runner = CliRunner()

@pytest.fixture
def test_data(tmp_path: Path):
    tc_pass = {"task_id": "t1", "input": "", "expected_tool_calls": [], "expected_final_state": {}, "expected_output_contains": []}
    tr_pass = {"task_id": "t1", "input": "", "steps": [], "final_output": "done", "actual_state": {}, "metadata": {}}
    
    tc_fail = {"task_id": "t2", "input": "", "expected_tool_calls": [{"tool": "search"}], "expected_final_state": {}, "expected_output_contains": []}
    tr_fail = {"task_id": "t2", "input": "", "steps": [], "final_output": "failed", "actual_state": {}, "metadata": {}}
    
    malformed_json = "{bad json"
    
    return {
        "tc_pass": tc_pass,
        "tr_pass": tr_pass,
        "tc_fail": tc_fail,
        "tr_fail": tr_fail,
        "malformed": malformed_json,
        "dir": tmp_path
    }

def test_single_trace_pass(test_data):
    d = test_data["dir"]
    tc_file = d / "tc_pass.json"
    tr_file = d / "tr_pass.json"
    
    with open(tc_file, "w") as f: json.dump(test_data["tc_pass"], f)
    with open(tr_file, "w") as f: json.dump(test_data["tr_pass"], f)
    
    result = runner.invoke(app, ["run", "--trace", str(tr_file), "--testcase", str(tc_file)])
    assert result.exit_code == 0
    assert "t1" in result.output

def test_single_trace_fail(test_data):
    d = test_data["dir"]
    tc_file = d / "tc_fail.json"
    tr_file = d / "tr_fail.json"
    
    with open(tc_file, "w") as f: json.dump(test_data["tc_fail"], f)
    with open(tr_file, "w") as f: json.dump(test_data["tr_fail"], f)
    
    result = runner.invoke(app, ["run", "--trace", str(tr_file), "--testcase", str(tc_file)])
    assert result.exit_code == 1
    assert "t2" in result.output

def test_single_trace_malformed(test_data):
    d = test_data["dir"]
    tc_file = d / "tc_mal.json"
    tr_file = d / "tr_mal.json"
    
    with open(tc_file, "w") as f: json.dump(test_data["tc_pass"], f)
    with open(tr_file, "w") as f: f.write(test_data["malformed"])
    
    result = runner.invoke(app, ["run", "--trace", str(tr_file), "--testcase", str(tc_file)])
    assert result.exit_code == 1
    assert "Error: Failed to parse task files" in result.output

def test_suite_mix_pass_fail(test_data):
    d = test_data["dir"]
    suite_dir = d / "suite"
    (suite_dir / "testcases").mkdir(parents=True)
    (suite_dir / "traces").mkdir(parents=True)
    out_dir = d / "out"
    
    with open(suite_dir / "testcases/t1.json", "w") as f: json.dump(test_data["tc_pass"], f)
    with open(suite_dir / "traces/t1.json", "w") as f: json.dump(test_data["tr_pass"], f)
    
    with open(suite_dir / "testcases/t2.json", "w") as f: json.dump(test_data["tc_fail"], f)
    with open(suite_dir / "traces/t2.json", "w") as f: json.dump(test_data["tr_fail"], f)
    
    result = runner.invoke(app, ["run", "--suite", str(suite_dir), "--output", str(out_dir)])
    assert result.exit_code == 1
    
    out_files = list(out_dir.glob("*.json"))
    assert len(out_files) == 2
    
    with open(out_dir / "t1.json") as f:
        t1_res = json.load(f)
        assert all(m["passed"] for m in t1_res["metrics"].values())
        
    with open(out_dir / "t2.json") as f:
        t2_res = json.load(f)
        assert not all(m["passed"] for m in t2_res["metrics"].values())

def test_suite_malformed_continues(test_data):
    d = test_data["dir"]
    suite_dir = d / "suite2"
    (suite_dir / "testcases").mkdir(parents=True)
    (suite_dir / "traces").mkdir(parents=True)
    out_dir = d / "out2"
    
    with open(suite_dir / "testcases/t1.json", "w") as f: json.dump(test_data["tc_pass"], f)
    with open(suite_dir / "traces/t1.json", "w") as f: json.dump(test_data["tr_pass"], f)
    
    with open(suite_dir / "testcases/t_bad.json", "w") as f: json.dump({"task_id": "t_bad"}, f)
    with open(suite_dir / "traces/t_bad.json", "w") as f: f.write(test_data["malformed"])
    
    result = runner.invoke(app, ["run", "--suite", str(suite_dir), "--output", str(out_dir)])
    assert result.exit_code == 1
    
    out_files = list(out_dir.glob("*.json"))
    assert len(out_files) == 2
    
    with open(out_dir / "t_bad.json") as f:
        bad_res = json.load(f)
        assert "error" in bad_res
        assert "Failed to parse task files" in bad_res["error"]
