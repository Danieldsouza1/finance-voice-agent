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
    "itc": "ITC.NS",
    "hul": "HINDUNILVR.NS",
    "hindustan unilever": "HINDUNILVR.NS",
    "airtel": "BHARTIARTL.NS",
    "bharti airtel": "BHARTIARTL.NS",
    "l&t": "LT.NS",
    "larsen": "LT.NS",
    "larsen and toubro": "LT.NS",
    "axis bank": "AXISBANK.NS",
    "axis": "AXISBANK.NS",
    "m&m": "M&M.NS",
    "mahindra": "M&M.NS",
    "tata steel": "TATASTEEL.NS",
    "bajaj finserv": "BAJAJFINSV.NS",
    "tech mahindra": "TECHM.NS",
    "techm": "TECHM.NS",
    "dr reddy": "DRREDDY.NS",
    "cipla": "CIPLA.NS",
    "hindalco": "HINDALCO.NS",
    "zomato": "ZOMATO.BO",  # Changed from ZOMATO.NS
    "swiggy": "SWIGGY.NS",
    "paytm": "PAYTM.NS",
    "nykaa": "FSN.NS",
    "policybazaar": "PBFINTECH.NS",
    "jio": "JIOFIN.NS",
    "jio financial": "JIOFIN.NS",
    "jio financial services": "JIOFIN.NS",
    "dmart": "DMART.NS",
    "avenue supermarts": "DMART.NS",
    "irfc": "IRFC.NS",
    "ireda": "IREDA.NS",
    "rvnl": "RVNL.NS",
    "coal india": "COALINDIA.NS",
    "suzlon": "SUZLON.NS",
    "tata power": "TATAPOWER.NS",
    "vedanta": "VEDL.NS",
    "mrf": "MRF.NS",
    "page industries": "PAGEIND.NS",
    "page": "PAGEIND.NS",
    "honeywell": "HONAUT.NS",
    "honeywell automation": "HONAUT.NS",
    "abbott india": "ABBOTINDIA.NS",
    "abbott": "ABBOTINDIA.NS",
    "bosch": "BOSCHLTD.NS",
    "3m india": "3MINDIA.NS",
    "shree cement": "SHREECEM.NS",
    "divis lab": "DIVISLAB.NS",
    "divis laboratories": "DIVISLAB.NS",
    "p&g": "PGHH.NS",
    "procter gamble": "PGHH.NS",
}


def resolve_ticker(company_name: str) -> str:
    """Resolves a company name or ticker string to a valid NSE ticker."""
    cleaned = company_name.strip().lower()
    if cleaned in TICKER_MAP:
        return TICKER_MAP[cleaned]
    if "." in company_name or company_name.isupper():
        return company_name.upper()
    for key, val in TICKER_MAP.items():
        if cleaned in key or key in cleaned:
            return val
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

        # Fallback 1: fast_info
        if not price:
            try:
                fast = stock.fast_info
                price = getattr(fast, 'last_price', None)
                if price:
                    price = round(float(price), 2)
            except Exception:
                pass

        # Fallback 2: recent history
        if not price:
            try:
                hist = stock.history(period="5d")
                if not hist.empty:
                    price = round(float(hist['Close'].iloc[-1]), 2)
                    prev_close = round(float(hist['Close'].iloc[-2]), 2) if len(hist) > 1 else None
            except Exception:
                pass

        # Fallback 3: BSE ticker
        if not price:
            try:
                bse_ticker = ticker.replace(".NS", ".BO")
                bse_stock = yf.Ticker(bse_ticker)
                bse_info = bse_stock.info
                bse_price = bse_info.get("currentPrice") or bse_info.get("regularMarketPrice")
                if bse_price:
                    price = bse_price
                    info = bse_info
                    ticker = bse_ticker
                    prev_close = bse_info.get("previousClose")
            except Exception:
                pass

        if not price:
            return (
                f"Live price data is currently unavailable for {company_name} ({ticker}). "
                f"This typically happens outside NSE trading hours (9:15 AM to 3:30 PM IST) "
                f"or when Yahoo Finance has limited coverage for this stock. "
                f"Please check NSE directly at nseindia.com for real-time prices."
            )

        day_high = info.get("dayHigh")
        day_low = info.get("dayLow")
        volume = info.get("volume")
        market_cap = info.get("marketCap")

        change = round(price - prev_close, 2) if prev_close else "N/A"
        change_pct = round((change / prev_close) * 100, 2) if prev_close and isinstance(change, float) else "N/A"
        mcap_cr = f"₹{round(market_cap / 1e7):,} Cr" if market_cap else "N/A"
        vol_str = f"{volume:,}" if volume else "N/A"

        return (
            f"Stock: {info.get('longName', ticker)} ({ticker})\n"
            f"Current Price: ₹{price}\n"
            f"Change: ₹{change} ({change_pct}%)\n"
            f"Day High: ₹{day_high or 'N/A'} | Day Low: ₹{day_low or 'N/A'}\n"
            f"Volume: {vol_str}\n"
            f"Market Cap: {mcap_cr}\n"
            f"Data Source: Yahoo Finance (15 min delay, NSE trading hours only)"
        )
    except Exception as e:
        return f"Error fetching price for {company_name}: {str(e)}"