import yfinance as yf
from langchain.tools import tool
from tools.stock_price import resolve_ticker


@tool
def get_stock_news(company_name: str) -> str:
    """
    Gets the latest news headlines for an Indian stock.
    Input: company name (e.g. 'Reliance', 'TCS', 'Infosys')
    Returns up to 5 recent news items with title, source, and summary.
    """
    try:
        ticker = resolve_ticker(company_name)
        stock = yf.Ticker(ticker)
        news = stock.news

        if not news:
            return f"No recent news found for {company_name}."

        result = f"Latest news for {company_name} ({ticker}):\n\n"
        for i, item in enumerate(news[:5], 1):
            content = item.get("content", {})
            title = content.get("title", "No title")
            source = content.get("provider", {}).get("displayName", "Unknown source")
            summary = content.get("summary", "")
            summary_short = summary[:200] + "..." if len(summary) > 200 else summary

            result += f"{i}. {title}\n"
            result += f"   Source: {source}\n"
            if summary_short:
                result += f"   {summary_short}\n"
            result += "\n"

        return result.strip()
    except Exception as e:
        return f"Error fetching news for {company_name}: {str(e)}"