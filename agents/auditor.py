from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from core.state import AgentState
from config import LLM_BASE_URL, LLM_API_KEY, MODEL_NAME, BENCHMARK_SHARPE, MAX_DRAWDOWN_LIMIT, MAX_ITERATIONS

def auditor_agent(state: AgentState) -> dict:
    llm = ChatOpenAI(
        model=MODEL_NAME, 
        openai_api_key=LLM_API_KEY, 
        base_url=LLM_BASE_URL
    )
    
    system_prompt = f"""You are the Head of Risk Management and Quantitative Auditing. You serve as the absolute gatekeeper against lookahead bias, curve-fitting, and excessive drawdown exposure. 
Your Mandate: 
1. Traceback Analysis: If `error_logs` exist, diagnose the exact matrix shape mismatch, indexing error, or syntax fault, and provide a precise, vectorized fix. 
2. Performance Audit: Review the `backtest_metrics`. The strict firm limits are a Sharpe Ratio > {BENCHMARK_SHARPE} and a Max Drawdown < {MAX_DRAWDOWN_LIMIT*100}%. 
3. Anti-Overfitting: If metrics fail, DO NOT suggest tweaking arbitrary parameters (e.g., changing a rolling window from 20 to 18). You must diagnose the structural flaw (e.g., over-trading in chop, extreme tail risk) and demand regime filters, ATR-based position sizing, or volume thresholds. 
4. Output: If metrics fail or code breaks, provide a highly technical critique to the Coder. If the strategy passes all empirical tests and code checks, set the status to 'passed'."""

    # Add instruction for explicitly outputting PASSED if checks are clear
    system_prompt += "\nIf the strategy passes all criteria and there are no errors, you must include the exact phrase 'STATUS: PASSED' in your response."

    human_content = f"Current Code:\n{state.get('current_code', 'None')}\n\n"
    
    if state.get('error_logs'):
        human_content += f"Execution Error Logs:\n{state.get('error_logs')}\n\n"
        
    if state.get('backtest_metrics'):
        human_content += f"Backtest Metrics:\n{state.get('backtest_metrics')}\n\n"
        
    human_content += "Please audit the strategy."

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_content)
    ]
    
    response = llm.invoke(messages)
    feedback = response.content.strip()
    
    status = "coding"
    if "STATUS: PASSED" in feedback or "status to 'passed'" in feedback.lower() or feedback.strip().lower().endswith("passed.") or feedback.strip().lower() == "passed":
        status = "passed"
        
    if status != "passed" and state.get('iteration_count', 0) >= MAX_ITERATIONS:
        status = "failed"
        
    return {
        "auditor_feedback": feedback,
        "status": status
    }
