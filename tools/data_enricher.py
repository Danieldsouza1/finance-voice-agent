import os
import requests

AV_KEY = os.getenv("ALPHA_VANTAGE_KEY", "")

def get_alpha_vantage_quote(ticker: str) -> dict:
    """
    Fetches quote data from Alpha Vantage as a fallback/enrichment source.
    ticker should be in format like 'RELIANCE.BSE' for Indian stocks.
    """
    if not AV_KEY:
        return {}
    try:
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={ticker}&apikey={AV_KEY}"
        r = requests.get(url, timeout=5)
        data = r.json().get("Global Quote", {})
        if not data:
            return {}
        return {
            "price": data.get("05. price"),
            "change": data.get("09. change"),
            "change_pct": data.get("10. change percent"),
            "volume": data.get("06. volume"),
            "prev_close": data.get("08. previous close"),
        }
    except Exception:
        return {}