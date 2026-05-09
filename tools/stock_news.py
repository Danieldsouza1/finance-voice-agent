import yfinance as yf
import re
from langchain.tools import tool
from tools.stock_price import resolve_ticker

@tool
def get_stock_news(company_name: str) -> str:
    """
    Gets the latest relevant news headlines for an Indian stock.
    Input: company name (e.g. 'Reliance', 'TCS', 'Infosys')
    Returns up to 5 recent news items filtered for relevance.
    """
    try:
        ticker = resolve_ticker(company_name)
        stock = yf.Ticker(ticker)
        news = stock.news

        if not news:
            return f"No recent news found for {company_name} ({ticker})."

        # Extract company name variations for relevance filtering
        company_lower = company_name.lower()
        ticker_base = ticker.replace(".NS", "").replace(".BO", "").lower()
        
        relevant = []
        generic = []

        for item in news[:10]:
            content = item.get("content", {})
            title = content.get("title", "")
            summary = content.get("summary", "")
            combined = (title + " " + summary).lower()

            # Check if this news item mentions the company
            is_relevant = (
                company_lower in combined or
                ticker_base in combined or
                any(word in combined for word in company_lower.split() if len(word) > 3)
            )

            # Clean and truncate the summary
            summary_clean = re.sub(r'\$(\d)', r' \1 dollars ', summary)
            summary_clean = re.sub(r'Source:.*', '', summary_clean).strip()
            summary_clean = re.sub(r'\$(\d)', r' \1 USD ', summary)
            summary_clean = re.sub(r'([a-z])([A-Z])', r'\1 \2', summary_clean)  # fix collapsed words
            summary_short = summary_clean[:180] + "..." if len(summary_clean) > 180 else summary_clean

            entry = {
                "title": title,
                "source": content.get("provider", {}).get("displayName", "Unknown"),
                "summary": summary_short,
                "relevant": is_relevant
            }

            if is_relevant:
                relevant.append(entry)
            else:
                generic.append(entry)

        # Prefer relevant news, fall back to generic if not enough
        final_news = relevant[:5] if len(relevant) >= 2 else (relevant + generic)[:5]

        if not final_news:
            return f"No news articles found for {company_name} ({ticker}) at this time."

        result = f"Latest news for {company_name} ({ticker}):\n\n"
        for i, item in enumerate(final_news, 1):
            tag = "" if item["relevant"] else " [General Market News]"
            result += f"{i}. {item['title']}{tag}\n"
            result += f"   Source: {item['source']}\n"
            if item["summary"]:
                result += f"   {item['summary']}\n"
            result += "\n"

        return result.strip()

    except Exception as e:
        return f"Could not fetch news for {company_name}. The ticker may not be recognized. Please try using the full company name like 'Tata Power' or 'Reliance Industries'."