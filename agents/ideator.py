from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from core.state import AgentState
from config import LLM_BASE_URL, LLM_API_KEY, MODEL_NAME

def ideator_agent(state: AgentState) -> dict:
    llm = ChatOpenAI(
        model=MODEL_NAME, 
        openai_api_key=LLM_API_KEY, 
        base_url=LLM_BASE_URL
    )
    
    system_prompt = """You are an elite quantitative research theorist at a tier-one proprietary trading firm. Your objective is to formulate robust, mathematically rigorous trading hypotheses that exploit structural market inefficiencies, statistical arbitrage, or regime-dependent anomalies. Avoid generic retail technical indicators. You must define: 
1. The Alpha Rationale: Why does this inefficiency exist (e.g., liquidity constraints, behavioral overreaction)? 
2. Mathematical Formulation: Precise entry and exit logic using statistical thresholds (e.g., rolling Z-scores, cointegration spreads, volatility-adjusted momentum). 
3. Regime Filters: Conditions under which the strategy should halt trading (e.g., VIX > 30, trending vs. ranging environments). 
Output a highly structured theory and reset `iteration_count` to 0."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content="Please generate a new quantitative trading theory.")
    ]
    
    response = llm.invoke(messages)
    
    return {
        "market_theory": response.content,
        "iteration_count": 0
    }
