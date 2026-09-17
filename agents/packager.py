from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from core.state import AgentState
from config import LLM_BASE_URL, LLM_API_KEY, MODEL_NAME

def packager_agent(state: AgentState) -> dict:
    llm = ChatOpenAI(
        model=MODEL_NAME, 
        openai_api_key=LLM_API_KEY, 
        base_url=LLM_BASE_URL
    )
    
    system_prompt = """You are the Lead Execution Engineer. Your job is to take a validated, profitable Python prototype and format it for the C++ high-frequency trading team. 
Take the validated `current_code` and generate a comprehensive Markdown report containing: 
1. Production Code: The finalized Python script with thorough algorithmic docstrings. 
2. Complexity Analysis: Big-O time and space complexity of the signal generation. 
3. Execution Edge Cases: Explicit warnings on slippage assumptions, missing tick data handling, and corporate action impacts. 
4. C++ Translation Architecture: Directives for translating the specific pandas vectorization into optimized C++ (e.g., Eigen or Armadillo library recommendations)."""

    human_content = f"Validated Python Code:\n{state.get('current_code')}\n\nPlease generate the final Markdown report."

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_content)
    ]
    
    response = llm.invoke(messages)
    
    return {
        "final_report": response.content,
        "status": "packaged"
    }
