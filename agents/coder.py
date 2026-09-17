from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from core.state import AgentState
from config import LLM_BASE_URL, LLM_API_KEY, MODEL_NAME

def coder_agent(state: AgentState) -> dict:
    llm = ChatOpenAI(
        model=MODEL_NAME, 
        openai_api_key=LLM_API_KEY, 
        base_url=LLM_BASE_URL
    )
    
    system_prompt = """You are a Senior Quantitative Developer. Your objective is to translate theoretical alpha models into zero-latency, heavily vectorized Python backtesting code. 
Strict Directives: 
1. You must write ONLY raw, executable Python code containing the exact function signature: `def generate_signals(df: pd.DataFrame) -> pd.Series:`. 
2. No Markdown formatting, no code block fences (```), and no explanatory text in your response. Just the raw code. 
3. Performance: You are strictly forbidden from using `for` loops or `.iterrows()`. You must use pure `pandas` and `numpy` vectorized matrix operations. 
4. Lookahead Bias Prevention: You must never use future data to calculate current signals. Ensure proper indexing and apply `.shift()` appropriately to rolling windows. 
5. Error Handling: Gracefully handle `NaN` values, division by zero, and missing data gaps. 
Address any feedback from the Auditor precisely."""

    human_content = f"Market Theory:\n{state.get('market_theory', '')}\n\n"
    
    if state.get('error_logs'):
        human_content += f"Execution Error Logs:\n{state.get('error_logs')}\n\n"
        
    if state.get('auditor_feedback'):
        human_content += f"Auditor Feedback:\n{state.get('auditor_feedback')}\n\n"
        
    if state.get('current_code'):
        human_content += f"Previous Code:\n{state.get('current_code')}\n\n"
        
    human_content += "Provide ONLY the raw, executable Python code with the generate_signals function."

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_content)
    ]
    
    response = llm.invoke(messages)
    
    # Strip markdown if the LLM ignores the prompt
    code = response.content.strip()
    if code.startswith("```python"):
        code = code[9:]
    elif code.startswith("```"):
        code = code[3:]
    if code.endswith("```"):
        code = code[:-3]
    code = code.strip()
    
    current_iteration = state.get('iteration_count', 0)
    
    return {
        "current_code": code,
        "iteration_count": current_iteration + 1
    }
