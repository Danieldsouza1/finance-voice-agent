import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from tools.stock_price import get_stock_price
from tools.fundamentals import get_stock_fundamentals
from tools.stock_news import get_stock_news

load_dotenv()

SYSTEM_PROMPT = """You are an AI-powered voice-based financial assistant, designed to provide data-backed investment insights to Indian retail investors.

Your primary goal is to help users explore stock options using real financial data. You are NOT a licensed financial advisor.

STRICT RULES:
1. Never say "buy" or "sell" as a direct instruction. Always frame stocks as "options to consider" or "worth exploring".
2. Always base your response on real-time data fetched by your tools.
3. ALWAYS call get_stock_price for EACH stock requested before responding.
4. If a tool returns no data or an error, clearly say so rather than guessing.

CRITICAL OUTPUT FORMAT:
Your Final Answer MUST use exact XML tags to separate the screen text from the voice text. Do NOT use JSON.

<ui_text>
A detailed, rich Markdown response for the screen. You MUST use tables, bullet points, and LaTeX formatting (using $ and $$) for any financial math or formulas. You MUST use actual line breaks (Enter key) so the Markdown table renders correctly. Always include a disclaimer at the bottom.
</ui_text>

<spoken_text>
A short, conversational summary (under 50 words) with NO special characters, markdown, or tables. Read currency out naturally, explicitly using "rupees" and "paise" for decimals (e.g., instead of 2427.3, write "2427 rupees and 30 paise"). This is what the voice will speak aloud.
</spoken_text>

Example Output:
<ui_text>
### Reliance Industries Analysis

| Metric | Value |
|---|---|
| P/E | 28.5 |
| EPS | ₹59.69 |

The Price-to-Earnings ratio is calculated using this formula:
$$P/E = \frac{Current\ Stock\ Price}{Earnings\ Per\ Share}$$

*Disclaimer: This is not financial advice.*
</ui_text>

<spoken_text>
Reliance is showing strong growth with a P E ratio of 28.5. The current price is 1463 rupees and 60 paise. The fundamentals look solid. This is for educational purposes only.
</spoken_text>"""


def build_agent():
    # 1. PRIMARY: Gemini 2.5 Flash Lite (Fastest, but 20/day limit)
    primary_llm = ChatOpenAI(
        model_name="gemini-2.5-flash-lite",
        api_key=os.getenv("GEMINI_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        temperature=0.3,
        max_retries=0,  # <-- Fails instantly if 20/day limit is hit
        timeout=20,
    )

    # 2. FALLBACK 1: Gemini 1.5 Flash (Huge 1,500/day limit)
    fallback_1 = ChatOpenAI(
        model_name="gemini-1.5-flash",
        api_key=os.getenv("GEMINI_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        temperature=0.3,
        max_retries=0,  # <-- Fails instantly if Google's API is down entirely
        timeout=20,
    )

    # 3. FALLBACK 2: Groq LLaMA 3.3 (Cross-provider backup)
    fallback_2 = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3,
        max_retries=1,  # <-- Last resort, allow it to retry once
        timeout=20,
    )

    # Combine them using LangChain's native fallback feature
    robust_llm = primary_llm.with_fallbacks([fallback_1, fallback_2])

    tools = [get_stock_price, get_stock_fundamentals, get_stock_news]

    # Pass the combined 'robust_llm' to the agent
    agent = create_react_agent(
        model=robust_llm,
        tools=tools,
        prompt=SYSTEM_PROMPT,
    )

    return agent


def run_agent(agent, user_input: str, chat_history: list) -> str:
    """
    Runs the agent with the given input and chat history.
    Returns the final text response.
    """
    messages = chat_history + [HumanMessage(content=user_input)]

    result = agent.invoke({"messages": messages})

    # Extract the last AI message as the response
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content

    # Fallback JSON just in case it fails completely
    return '{"ui_text": "I was unable to generate a response. Please try again.", "spoken_text": "I was unable to generate a response. Please try again."}'