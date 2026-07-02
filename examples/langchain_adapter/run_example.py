import os

from langchain_core.tools import tool
from langchain_core.language_models.fake import FakeListLLM
from langchain_core.tracers.context import collect_runs
from langchain_core.prompts import PromptTemplate

from openeval.adapters.langchain import from_langchain_run

from openeval.models import EvalTestCase

@tool
def calculate_length(text: str) -> int:
    """Returns the length of a string."""
    return len(text)

def main():
    print("==============================================")
    print("1. Setting up LangChain chain with FakeListLLM")
    print("==============================================")
    
    llm = FakeListLLM(responses=["hello world"])
    
    prompt = PromptTemplate.from_template("Generate a string related to: {input}")
    chain = prompt | llm | calculate_length
    
    print("\n==============================================")
    print("2. Executing LangChain chain")
    print("==============================================")
    
    input_text = "test string"
    
    with collect_runs() as cb:
        output = chain.invoke({"input": input_text})
        run = cb.traced_runs[0]
        
    print(f"Chain Final Output: {output}")
    
    print("\n==============================================")
    print("3. Converting LangChain Run to OpenEval Trace")
    print("==============================================")
    
    trace = from_langchain_run(run)
    print(f"Extracted {len(trace.steps)} steps.")
    for step in trace.steps:
        print(f"  Step {step.step_id} [{step.type}]: {step.content}")
        if step.type == "tool_call":
            print(f"    Args: {step.tool_args}")
        elif step.type == "tool_result":
            if step.error:
                print(f"    Error: {step.error}")
            else:
                print(f"    Result: {step.tool_result}")
    
    print("\n==============================================")
    print("4. Running OpenEval Metrics")
    print("==============================================")
    
    # Check what LangChain actually sent as args so our expected metric matches
    tool_args_from_run = trace.steps[0].tool_args if trace.steps else {}
    
    from openeval.metrics import ToolSelectionAccuracy, ArgumentCorrectness, StepEfficiency

    test_case = EvalTestCase(
        task_id="langchain_task",
        input=input_text,
        expected_tool_calls=[{"tool": "calculate_length", "args": tool_args_from_run}],
        expected_final_state={},
        expected_output_contains=["11"],
        max_steps=5,
        timeout_seconds=30.0
    )

    results = [
        ToolSelectionAccuracy().score(trace, test_case),
        ArgumentCorrectness().score(trace, test_case),
        StepEfficiency().score(trace, test_case)
    ]
    
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] {result.metric_name}: Score {result.score:.2f} - {result.details}")

if __name__ == "__main__":
    main()
