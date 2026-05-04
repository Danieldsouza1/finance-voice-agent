import yfinance as yf
from langchain.tools import tool

TICKER_MAP = {
    "reliance": "RELIANCE.NS",
    "reliance industries": "RELIANCE.NS",
    "tcs": "TCS.NS",
    "tata consultancy": "TCS.NS",
    "tata consultancy services": "TCS.NS",
    "infosys": "INFY.NS",
    "infy": "INFY.NS",
    "hdfc": "HDFCBANK.NS",
    "hdfc bank": "HDFCBANK.NS",
    "icici": "ICICIBANK.NS",
    "icici bank": "ICICIBANK.NS",
    "wipro": "WIPRO.NS",
    "tata motors": "TATAMOTORS.NS",
    "tatamotors": "TATAMOTORS.NS",
    "bajaj finance": "BAJFINANCE.NS",
    "bajaj": "BAJFINANCE.NS",
    "sbi": "SBIN.NS",
    "state bank": "SBIN.NS",
    "state bank of india": "SBIN.NS",
    "adani": "ADANIENT.NS",
    "adani enterprises": "ADANIENT.NS",
    "sun pharma": "SUNPHARMA.NS",
    "sunpharma": "SUNPHARMA.NS",
    "maruti": "MARUTI.NS",
    "maruti suzuki": "MARUTI.NS",
    "asian paints": "ASIANPAINT.NS",
    "titan": "TITAN.NS",
    "ltimindtree": "LTIM.NS",
    "hcl": "HCLTECH.NS",
    "hcl tech": "HCLTECH.NS",
    "kotak": "KOTAKBANK.NS",
    "kotak bank": "KOTAKBANK.NS",
    "kotak mahindra": "KOTAKBANK.NS",
    "nestle": "NESTLEIND.NS",
    "ongc": "ONGC.NS",
    "ntpc": "NTPC.NS",
    "power grid": "POWERGRID.NS",
    "ultracemco": "ULTRACEMCO.NS",
    "ultratech cement": "ULTRACEMCO.NS",
}

def resolve_ticker(company_name: str) -> str:
    """Resolves a company name or ticker string to a valid NSE ticker."""
    cleaned = company_name.strip().lower()
    if cleaned in TICKER_MAP:
        return TICKER_MAP[cleaned]
    # If it already looks like a ticker (e.g. "RELIANCE.NS"), return as-is
    if "." in company_name or company_name.isupper():
        return company_name.upper()
    # Partial match fallback
    for key, val in TICKER_MAP.items():
        if cleaned in key or key in cleaned:
            return val
    # Last resort: assume NSE format
    return company_name.upper().replace(" ", "") + ".NS"


@tool
def get_stock_price(company_name: str) -> str:
    """
    Gets the current stock price and basic market data for an Indian stock.
    Input: company name (e.g. 'Reliance', 'TCS', 'Infosys')
    """
    try:
        ticker = resolve_ticker(company_name)
        stock = yf.Ticker(ticker)
        info = stock.info

        price = info.get("currentPrice") or info.get("regularMarketPrice")
        prev_close = info.get("previousClose")
        day_high = info.get("dayHigh")
        day_low = info.get("dayLow")
        volume = info.get("volume")
        market_cap = info.get("marketCap")

        if not price:
            return f"Could not fetch price data for {company_name}. The market may be closed or the ticker could not be resolved."

        change = round(price - prev_close, 2) if prev_close else "N/A"
        change_pct = round((change / prev_close) * 100, 2) if prev_close and isinstance(change, float) else "N/A"
        mcap_cr = f"₹{round(market_cap / 1e7):,} Cr" if market_cap else "N/A"

        return (
            f"Stock: {info.get('longName', ticker)} ({ticker})\n"
            f"Current Price: ₹{price}\n"
            f"Change: ₹{change} ({change_pct}%)\n"
            f"Day High: ₹{day_high} | Day Low: ₹{day_low}\n"
            f"Volume: {volume:,}\n"
            f"Market Cap: {mcap_cr}"
        )
    except Exception as e:
        return f"Error fetching price for {company_name}: {str(e)}"