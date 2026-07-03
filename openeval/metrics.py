# No LLM calls anywhere in this file. No API clients. Pure functions/deterministic logic only.

from abc import ABC, abstractmethod
from collections import Counter
from openeval.models import AgentTrace, EvalTestCase, MetricResult

class BaseMetric(ABC):
    name: str
    description: str

    @abstractmethod
    def score(self, trace: AgentTrace, test_case: EvalTestCase) -> MetricResult: ...

    @abstractmethod
    def explain(self, result: MetricResult) -> str: ...


class ToolSelectionAccuracy(BaseMetric):
    """
    ToolSelectionAccuracy: correct_tools_called / total_expected_tools
    """
    name = "Tool Selection Accuracy"
    description = "Measures the accuracy of tool selection against expected tools."

    def score(self, trace: AgentTrace, test_case: EvalTestCase) -> MetricResult:
        actual_tools = [step.tool_name for step in trace.steps if step.type == "tool_call" and step.tool_name is not None]
        expected_tools = [call["tool"] for call in test_case.expected_tool_calls]

        if not expected_tools:
            score = 1.0 if not actual_tools else 0.0
            passed = score == 1.0
            details = "No tools expected and none called." if passed else f"Expected 0 tools, but {len(actual_tools)} were called."
            return MetricResult(self.name, score, passed, details)

        actual_count = Counter(actual_tools)
        expected_count = Counter(expected_tools)
        
        correct_tools_called = sum(min(actual_count[tool], expected_count[tool]) for tool in expected_count)
        total_expected_tools = len(expected_tools)
        
        tool_selection_acc = correct_tools_called / total_expected_tools
        passed = tool_selection_acc == 1.0
        details = f"Correctly called {correct_tools_called} of {total_expected_tools} expected tools."
        
        return MetricResult(self.name, float(tool_selection_acc), passed, details)

    def explain(self, result: MetricResult) -> str:
        return f"{self.name}: {result.details} Score: {result.score:.2f}."


class ArgumentCorrectness(BaseMetric):
    """
    ArgumentCorrectness: correct_args / total_args_evaluated
    """
    name = "Argument Correctness"
    description = "Measures the correctness of arguments provided to tools."

    def score(self, trace: AgentTrace, test_case: EvalTestCase) -> MetricResult:
        expected_tool_calls = test_case.expected_tool_calls
        tool_names = [c["tool"] for c in expected_tool_calls]
        if len(tool_names) != len(set(tool_names)):
            raise ValueError("ArgumentCorrectness (v0.1) does not support duplicate tool names in expected_tool_calls")
            
        if not expected_tool_calls:
            return MetricResult(self.name, 1.0, True, "No expected tools, no arguments to evaluate.")

        actual_tools = [step for step in trace.steps if step.type == "tool_call" and step.tool_name is not None]
        # Since duplicate names are asserted away, we can just use the first instance safely in a dict
        actual_by_name = {step.tool_name: (step.tool_args or {}) for step in actual_tools}

        correct_args = 0
        total_args_evaluated = 0
        called_expected_tools = 0

        for expected_call in expected_tool_calls:
            tool_name = expected_call["tool"]
            expected_args = expected_call.get("args", {})
            
            if tool_name not in actual_by_name:
                continue
                
            called_expected_tools += 1
            actual_args = actual_by_name[tool_name]
            total_args_evaluated += len(expected_args)
            
            for k, expected_v in expected_args.items():
                if actual_args.get(k) == expected_v:
                    correct_args += 1

        if called_expected_tools == 0:
            expected_count = len(expected_tool_calls)
            details = f"0 of {expected_count} expected tool calls occurred; argument correctness cannot be assessed — treated as failed."
            return MetricResult(self.name, 0.0, False, details)

        if total_args_evaluated == 0:
            return MetricResult(self.name, 1.0, True, "Called tools had no expected arguments to evaluate.")

        score = correct_args / total_args_evaluated
        passed = score == 1.0
        details = f"Correctly matched {correct_args} of {total_args_evaluated} expected arguments."
        return MetricResult(self.name, float(score), passed, details)

    def explain(self, result: MetricResult) -> str:
        return f"{self.name}: {result.details} Score: {result.score:.2f}."


class StepEfficiency(BaseMetric):
    """
    StepEfficiency: optimal_steps / actual_steps_taken (penalizes unnecessary calls)
    """
    name = "Step Efficiency"
    description = "Measures the efficiency of the agent's steps."

    def score(self, trace: AgentTrace, test_case: EvalTestCase) -> MetricResult:
        optimal_steps = len(test_case.expected_tool_calls)
        actual_steps_taken = sum(1 for step in trace.steps if step.type == "tool_call")

        if actual_steps_taken == 0:
            score = 1.0 if optimal_steps == 0 else 0.0
            passed = score == 1.0
            details = "No steps taken, no steps expected." if passed else f"Expected {optimal_steps} steps, but 0 were taken."
            return MetricResult(self.name, score, passed, details)

        raw_efficiency = optimal_steps / actual_steps_taken
        # Cap at 1.0 so we don't reward under-shooting above perfect score
        efficiency = min(1.0, raw_efficiency)
        
        passed = efficiency == 1.0
        details = f"Took {actual_steps_taken} steps vs optimal {optimal_steps} steps."
        return MetricResult(self.name, float(efficiency), passed, details)

    def explain(self, result: MetricResult) -> str:
        return f"{self.name}: {result.details} Score: {result.score:.2f}."


class GoalCompletionRate(BaseMetric):
    """
    GoalCompletionRate: state_matches_expected(actual_state, expected_state)
    """
    name = "Goal Completion Rate"
    description = "Measures whether the final state matches the expected state."

    def score(self, trace: AgentTrace, test_case: EvalTestCase) -> MetricResult:
        expected_state = test_case.expected_final_state
        actual_state = trace.actual_state
        
        if not expected_state:
            return MetricResult(self.name, 1.0, True, "No expected state to match.")
            
        correct_keys = 0
        total_keys = len(expected_state)
        
        for k, expected_v in expected_state.items():
            if k in actual_state and actual_state[k] == expected_v:
                correct_keys += 1
                
        score = correct_keys / total_keys
        passed = score == 1.0
        details = f"Matched {correct_keys} of {total_keys} expected state keys."
        return MetricResult(self.name, float(score), passed, details)

    def explain(self, result: MetricResult) -> str:
        return f"{self.name}: {result.details} Score: {result.score:.2f}."


class TrajectoryDeviationScore(BaseMetric):
    """
    TrajectoryDeviationScore: sum(step_violations) / len(steps)
    """
    name = "Trajectory Deviation Score"
    description = "Measures how much the agent deviated from the optimal trajectory."

    def score(self, trace: AgentTrace, test_case: EvalTestCase) -> MetricResult:
        raise NotImplementedError()

    def explain(self, result: MetricResult) -> str:
        raise NotImplementedError()
