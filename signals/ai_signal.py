"""
signals/ai_signal.py
Claude API integration for AI-powered investment reasoning.
Claude acts as the senior analyst that reviews quantitative signals
and makes the final BUY/HOLD/SELL decision with a written thesis.
"""

import json
import anthropic
import sys
sys.path.append("..")
import config
from signals.dcf import BANK_TICKERS

_client = None

def get_client():
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    return _client


def analyze_stock(
    symbol: str,
    fundamentals: dict,
    dcf_result: dict,
    factor_scores: dict,
    news_headlines: list = None,
) -> dict:
    """
    Send all quantitative signals to Claude and get a structured
    investment decision with thesis, risks, and position sizing.

    Returns:
        dict with action, conviction, position_size_pct, thesis, risks, red_flags
    """
    if news_headlines is None:
        news_headlines = []

    is_bank = symbol in BANK_TICKERS

    # Build valuation section depending on model used
    if is_bank:
        valuation_section = f"""=== BANK VALUATION (P/B + ROE MODEL) ===
Note: DCF is not applicable for banks. Using P/B + ROE methodology.
P/B Ratio: {dcf_result.get('pb_ratio', 'N/A')}
ROE: {dcf_result.get('roe_pct', 'N/A')}%
P/B Score (0-100): {dcf_result.get('pb_score', 'N/A')}
ROE Score (0-100): {dcf_result.get('roe_score', 'N/A')}
Composite Score (0-100): {dcf_result.get('composite_score', 'N/A')}
Bank Valuation Verdict: {dcf_result.get('verdict', 'N/A')}"""
        buy_rule = "- Only recommend BUY if conviction >= 6 AND Bank Composite Score >= 55 (or verdict is UNDERVALUED/FAIR)"
    else:
        valuation_section = f"""=== DCF VALUATION ===
Intrinsic Value: {dcf_result.get('intrinsic_price', 'N/A')} THB
Margin of Safety: {dcf_result.get('margin_of_safety', 'N/A')}%
WACC: {dcf_result.get('wacc_pct', 'N/A')}%
FCF Growth Used: {dcf_result.get('fcf_growth_pct', 'N/A')}%
DCF Verdict: {dcf_result.get('verdict', 'N/A')}"""
        buy_rule = "- Only recommend BUY if conviction >= 6 AND margin of safety > 10%"

    # Build a rich prompt with all available data
    prompt = f"""You are a senior equity analyst specializing in the Stock Exchange of Thailand (SET).
Analyze the following data and provide a structured investment decision.

=== STOCK: {symbol} ===
Name: {fundamentals.get('name', symbol)}
Sector: {fundamentals.get('sector', 'Unknown')}
Current Price: {fundamentals.get('current_price', 'N/A')} THB

{valuation_section}

=== VALUATION MULTIPLES ===
P/E Ratio: {fundamentals.get('pe_ratio', 'N/A')}
P/B Ratio: {fundamentals.get('pb_ratio', 'N/A')}
EV/EBITDA: {fundamentals.get('ev_ebitda', 'N/A')}
Dividend Yield: {str(round(fundamentals.get('dividend_yield', 0) * 100, 2)) + '%' if fundamentals.get('dividend_yield') else 'N/A'}

=== QUALITY METRICS ===
ROE: {str(round(fundamentals.get('roe', 0) * 100, 1)) + '%' if fundamentals.get('roe') else 'N/A'}
Profit Margin: {str(round(fundamentals.get('profit_margin', 0) * 100, 1)) + '%' if fundamentals.get('profit_margin') else 'N/A'}
Debt/Equity: {fundamentals.get('debt_equity', 'N/A')}
Revenue Growth: {str(round(fundamentals.get('revenue_growth', 0) * 100, 1)) + '%' if fundamentals.get('revenue_growth') else 'N/A'}

=== FACTOR SCORES (0-100) ===
Value Score: {factor_scores.get('value', {}).get('score', 'N/A')}
Quality Score: {factor_scores.get('quality', {}).get('score', 'N/A')}
Momentum Score: {factor_scores.get('momentum', {}).get('score', 'N/A')}
Yield Score: {factor_scores.get('yield', {}).get('score', 'N/A')}
COMPOSITE: {factor_scores.get('composite', 'N/A')} / 100
Quant Signal: {factor_scores.get('signal', 'N/A')}

=== RECENT NEWS / CONTEXT ===
{chr(10).join(f"- {h}" for h in news_headlines) if news_headlines else "No recent news available."}

=== YOUR TASK ===
Based on all the above, provide your investment decision as a JSON object with EXACTLY this structure:
{{
  "action": "BUY | HOLD | SELL | AVOID",
  "conviction": <integer 1-10>,
  "position_size_pct": <integer 2-15, percentage of portfolio>,
  "holding_period": "short (days) | medium (weeks) | long (months)",
  "thesis": "<2-3 sentence investment thesis>",
  "bull_case": "<1-2 sentence upside scenario>",
  "bear_case": "<1-2 sentence downside scenario>",
  "key_risks": ["<risk1>", "<risk2>", "<risk3>"],
  "red_flags": ["<flag1>"] or [],
  "stop_loss_pct": <number 5-15, recommended stop-loss percentage>,
  "target_price": <number or null>
}}

Rules:
- {buy_rule}
- Flag any accounting red flags (high receivables growth, negative FCF, >2x debt/equity)
- Consider SET market context: Thai baht risk, BOT policy, export exposure
- Respond with ONLY valid JSON. No markdown, no explanation outside the JSON.
"""

    try:
        client = get_client()
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=800,
            system="You are a Thai equity analyst. Respond with valid JSON only. No markdown backticks.",
            messages=[{"role": "user", "content": prompt}]
        )

        raw = response.content[0].text.strip()

        # Strip markdown fences if Claude adds them despite instructions
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        result = json.loads(raw)
        result["symbol"] = symbol
        result["ai_analyzed"] = True
        return result

    except json.JSONDecodeError as e:
        print(f"  [WARN] JSON parse error for {symbol}: {e}")
        return _fallback_signal(symbol, factor_scores)
    except Exception as e:
        print(f"  [WARN] Claude API error for {symbol}: {e}")
        return _fallback_signal(symbol, factor_scores)


def _fallback_signal(symbol: str, factor_scores: dict) -> dict:
    """
    Fallback when Claude API is unavailable.
    Uses composite quant score to generate a basic signal.
    """
    composite = factor_scores.get("composite", 50)
    quant_signal = factor_scores.get("signal", "HOLD")

    action_map = {
        "STRONG BUY": ("BUY", 7, 8),
        "BUY": ("BUY", 6, 6),
        "HOLD": ("HOLD", 5, 0),
        "WEAK": ("AVOID", 3, 0),
        "AVOID": ("AVOID", 2, 0),
    }
    action, conviction, size = action_map.get(quant_signal, ("HOLD", 5, 0))

    return {
        "symbol":           symbol,
        "action":           action,
        "conviction":       conviction,
        "position_size_pct": size,
        "holding_period":   "medium (weeks)",
        "thesis":           f"Quantitative signal ({quant_signal}) based on composite score of {composite}/100.",
        "bull_case":        "Strong factor scores suggest potential outperformance.",
        "bear_case":        "Broader market downturn could override stock-specific signals.",
        "key_risks":        ["Market risk", "Liquidity risk", "Currency risk"],
        "red_flags":        [],
        "stop_loss_pct":    8,
        "target_price":     None,
        "ai_analyzed":      False,
    }
