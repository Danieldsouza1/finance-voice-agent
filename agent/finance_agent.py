import os
import re
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent

from tools.stock_price import get_stock_price
from tools.fundamentals import get_stock_fundamentals
from tools.stock_news import get_stock_news

load_dotenv()

SYSTEM_PROMPT = r"""You are an AI-powered voice-based financial assistant designed to provide data-backed investment insights to Indian retail investors. You are NOT a licensed financial advisor.

TOOL SELECTION RULES:
- User asks for NEWS/UPDATES → call get_stock_news ONLY
- User asks for PRICE → call get_stock_price ONLY
- User asks for FUNDAMENTALS/PE/EPS → call get_stock_fundamentals ONLY
- User asks for ANALYSIS or SHOULD I INVEST → call ALL THREE tools
- User asks to COMPARE → call tools for EACH stock explicitly.
- User asks for SECTOR OVERVIEW (e.g., "banking sector") → identify 2-3 key Indian stocks from that sector, call get_stock_price and get_stock_fundamentals for EACH, and provide a comparative overview.

STRICT RULES:
1. Always base your response on real-time data fetched by your tools. Do not hallucinate data.
2. Match the spoken_text content to what was actually asked.
3. PHONETIC FORGIVENESS: User input comes from Speech-to-Text and may contain mishearings. If a name seems phonetically close to a known stock, resolve it silently and fetch data for the correct stock. 
4. NO PLAIN TEXT RESPONSES: Even if a tool returns an error or data is unavailable (e.g., if Zomato data is missing), you MUST wrap your error message in the exact XML tags required below. Never break format.

CRITICAL OUTPUT FORMAT:
Your Final Answer MUST ALWAYS use exact XML tags. Do NOT use plain text or JSON.

<ui_text>
A detailed Markdown response for the screen. Use tables and bullet points. Always include a disclaimer at the bottom.
</ui_text>

<spoken_text>
A natural conversational response of 60 to 80 words. Must directly answer what the user asked using the actual data fetched. End with one short disclaimer sentence.
</spoken_text>
"""

def build_agent():
    groq_key = os.getenv("GROQ_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")

    # 1. PRIMARY: Groq (Llama 3.3) - High RPM limit, incredibly fast
    primary_llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=groq_key,
        temperature=0.3,
        max_retries=0, # Fails fast if exhausted to trigger Gemini fallback
        timeout=20,
    )

    # 2. FALLBACK 1: Gemini 2.5 Flash Lite
    fallback_1 = ChatOpenAI(
        model_name="gemini-2.5-flash-lite",
        api_key=gemini_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        temperature=0.3,
        max_retries=0,
        timeout=20,
    )

    # 3. FALLBACK 2: Gemini 1.5 Flash - The ultimate safety net
    fallback_2 = ChatOpenAI(
        model_name="gemini-1.5-flash",
        api_key=gemini_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        temperature=0.3,
        max_retries=0,
        timeout=20,
    )

    # Chain the three models together
    robust_llm = primary_llm.with_fallbacks([fallback_1, fallback_2])
    tools = [get_stock_price, get_stock_fundamentals, get_stock_news]

    agent = create_react_agent(
        model=robust_llm,
        tools=tools,
        prompt=SYSTEM_PROMPT,
    )

    return agent

def run_agent(agent, user_input: str, chat_history: list) -> str:
    messages = chat_history + [HumanMessage(content=user_input)]
    result = agent.invoke({"messages": messages})

    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content

    return "<ui_text>Error generating response. Please try asking again.</ui_text><spoken_text>Sorry, I encountered an error generating my response.</spoken_text>"