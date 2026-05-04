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

        pe_str = f"{round(pe, 2)}" if pe else "N/A"
        eps_str = f"₹{round(eps, 2)}" if eps else "N/A"
        dy_pct = round(dividend_yield * 100, 2)
        dy_str = f"{dy_pct}%" if dividend_yield and dy_pct < 30 else (f"{round(dividend_yield, 2)}%" if dividend_yield else "N/A")
        roe_str = f"{round(roe * 100, 2)}%" if roe else "N/A"
        de_str = f"{round(de_ratio, 2)}" if de_ratio else "N/A"
        pm_str = f"{round(profit_margins * 100, 2)}%" if profit_margins else "N/A"
        rg_str = f"{round(revenue_growth * 100, 2)}%" if revenue_growth else "N/A"

        return (
            f"Fundamentals: {info.get('longName', ticker)} ({ticker})\n"
            f"P/E Ratio: {pe_str}\n"
            f"EPS: {eps_str}\n"
            f"52-Week High: ₹{week_high} | Low: ₹{week_low}\n"
            f"Dividend Yield: {dy_str}\n"
            f"Return on Equity: {roe_str}\n"
            f"Debt-to-Equity: {de_str}\n"
            f"Profit Margin: {pm_str}\n"
            f"Revenue Growth (YoY): {rg_str}"
        )
    except Exception as e:
        return f"Error fetching fundamentals for {company_name}: {str(e)}"