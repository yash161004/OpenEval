import json
import os

from langchain_core.tools import tool
from langchain_core.language_models.fake import FakeListLLM
from langchain_core.tracers.context import collect_runs
from langchain_core.prompts import PromptTemplate
from openeval.adapters.langchain import from_langchain_run

@tool
def check_weather(location: str) -> str:
    """Returns the weather for a given location."""
    return f"The weather in {location} is sunny."

def main():
    print("Running quickstart agent...")
    
    # We use FakeListLLM for the quickstart so you don't need API keys!
    # Swap this for ChatOpenAI or ChatAnthropic in your real app.
    llm = FakeListLLM(responses=["San Francisco"])
    
    prompt = PromptTemplate.from_template("Extract the city name from: {input}")
    
    # Our simple agent pipeline: prompt -> llm -> tool
    agent = prompt | llm | check_weather
    
    # Run the agent while collecting its execution trace
    with collect_runs() as cb:
        output = agent.invoke({"input": "What is the weather in San Francisco?"})
        run = cb.traced_runs[0]
        
    print(f"Agent finished. Output: {output}")
    
    # Convert LangChain run to OpenEval trace
    trace = from_langchain_run(run, task_id="quickstart")
    
    # Write trace to disk for the CLI
    trace_path = os.path.join(os.path.dirname(__file__), "trace.json")
    with open(trace_path, "w") as f:
        json.dump(trace, f, indent=2, default=lambda x: x.__dict__)
        
    print(f"Trace saved to {trace_path}")

if __name__ == "__main__":
    main()
