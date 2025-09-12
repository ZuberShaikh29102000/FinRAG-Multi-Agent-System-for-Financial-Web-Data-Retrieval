from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import openai
import os
from dotenv import load_dotenv
load_dotenv()
openai.api_key=os.getenv("OPENAI_API_KEY")

# web Search Agent
web_search_agent=Agent(
    name="Web Search Agent",
    role="Search the web for the information",
    model=Groq(id="openai/gpt-oss-120b"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=True,

)

# Financial Agent
finance_agent=Agent(
    name="Finance AI Agent",
    model=Groq(id="openai/gpt-oss-120b"),
    tools=[
        YFinanceTools(stock_price=True,analyst_recommendations=True,stock_fundamentals=True,company_news=True),
    ],
    instructions=["Use tables to display the data"],
    show_tool_calls=True,
    markdown=True,
)
# multi agent application
multi_ai_agent = Agent(
    team=[web_search_agent, finance_agent],
    model=Groq(id="openai/gpt-oss-120b"),
    instructions=["Always include sources","Use table to display the data"],
    show_tool_calls=True,
    markdown=True
)

multi_ai_agent.print_response("Summarize analyst recommendation and share the latest news for NVDA",stream=True)