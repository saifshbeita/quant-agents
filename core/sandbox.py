import pandas as pd
import numpy as np
import traceback

def execute_strategy(code_str: str, df: pd.DataFrame) -> dict:
    """
    Safely executes AI-generated trading strategy code and evaluates its performance.
    """
    df = df.copy()
    try:
        # a. Execute code and extract generate_signals
        local_scope = {}
        # Provide common data science libraries to the exec scope
        exec(code_str, {"pd": pd, "np": np}, local_scope)
        
        if 'generate_signals' not in local_scope:
            raise ValueError("The code must define a 'generate_signals(df: pd.DataFrame) -> pd.Series' function.")
            
        generate_signals = local_scope['generate_signals']
        
        # b. Generate raw signals. Ensure signals are values in {-1, 0, 1}
        signals = generate_signals(df.copy())
        
        if not isinstance(signals, pd.Series):
            raise TypeError("generate_signals must return a pandas Series.")
            
        # Ensure values are strictly -1, 0, or 1
        df['Signal'] = signals.fillna(0).apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        
        # c. CRITICAL: Eliminate lookahead bias
        df['Position'] = df['Signal'].shift(1)
        
        # d. Calculate Strategy_Return
        df['Strategy_Return'] = df['Position'] * df['Market_Return']
        
        # e. Calculate metrics
        returns = df['Strategy_Return'].dropna()
        
        if len(returns) == 0:
            raise ValueError("No valid returns generated to compute metrics.")
            
        # Sharpe Ratio (annualized, 252 trading days)
        mean_ret = returns.mean()
        std_ret = returns.std()
        sharpe = float((mean_ret / std_ret) * np.sqrt(252)) if std_ret != 0 else 0.0
        
        # Total Return (cumulative product of 1 + strategy return - 1)
        cum_returns = (1 + returns).cumprod()
        total_return = float(cum_returns.iloc[-1] - 1) if len(cum_returns) > 0 else 0.0
        
        # Max Drawdown: peak-to-trough drop from cumulative returns
        rolling_max = cum_returns.cummax()
        drawdowns = (cum_returns - rolling_max) / rolling_max
        max_drawdown = float(abs(drawdowns.min())) if len(drawdowns) > 0 else 0.0
        
        # Win Rate: percentage of positive trading days
        win_rate = float((returns > 0).sum() / len(returns)) if len(returns) > 0 else 0.0
        
        return {
            "success": True,
            "metrics": {
                "sharpe": sharpe,
                "max_drawdown": max_drawdown,
                "total_return": total_return,
                "win_rate": win_rate
            },
            "error": None
        }
        
    except Exception:
        # Catch full traceback if syntax error or runtime error occurs
        return {
            "success": False,
            "metrics": None,
            "error": str(traceback.format_exc())
        }
