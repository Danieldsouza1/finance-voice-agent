import yfinance as yf
from langchain.tools import tool
from tools.stock_price import resolve_ticker

@tool
def get_stock_fundamentals(company_name: str) -> str:
    """
    Gets fundamental financial data for an Indian stock.
    Input: company name (e.g. 'Reliance', 'TCS', 'Infosys')
    Includes: PE ratio, EPS, 52-week range, dividend yield, ROE, debt-to-equity.
    """
    try:
        ticker = resolve_ticker(company_name)
        stock = yf.Ticker(ticker)
        info = stock.info

        pe = info.get("trailingPE")
        eps = info.get("trailingEps")
        week_high = info.get("fiftyTwoWeekHigh")
        week_low = info.get("fiftyTwoWeekLow")
        dividend_yield = info.get("dividendYield")
        roe = info.get("returnOnEquity")
        de_ratio = info.get("debtToEquity")
        profit_margins = info.get("profitMargins")
        revenue_growth = info.get("revenueGrowth")

        # FIX: Using 'is not None' handles 0.0 values correctly
        pe_str = f"{round(pe, 2)}" if pe is not None else "N/A"
        eps_str = f"₹{round(eps, 2)}" if eps is not None else "N/A"

        if dividend_yield is not None:
            dy_pct = round(dividend_yield * 100, 2)
            dy_str = f"{dy_pct}%" if dy_pct < 30 else f"{round(dividend_yield, 2)}%"
        else:
            dy_str = "N/A"

        roe_str = f"{round(roe * 100, 2)}%" if roe is not None else "N/A"
        de_str = f"{round(de_ratio, 2)}" if de_ratio is not None else "N/A"
        pm_str = f"{round(profit_margins * 100, 2)}%" if profit_margins is not None else "N/A"
        rg_str = f"{round(revenue_growth * 100, 2)}%" if revenue_growth is not None else "N/A"

        week_high_str = f"₹{week_high}" if week_high is not None else "N/A"
        week_low_str = f"₹{week_low}" if week_low is not None else "N/A"

        return (
            f"Fundamentals: {info.get('longName', ticker)} ({ticker})\n"
            f"P/E Ratio: {pe_str}\n"
            f"EPS: {eps_str}\n"
            f"52-Week High: {week_high_str} | Low: {week_low_str}\n"
            f"Dividend Yield: {dy_str}\n"
            f"Return on Equity: {roe_str}\n"
            f"Debt-to-Equity: {de_str}\n"
            f"Profit Margin: {pm_str}\n"
            f"Revenue Growth (YoY): {rg_str}"
        )

    except Exception as e:
        return f"Error fetching fundamentals for {company_name}: {str(e)}"