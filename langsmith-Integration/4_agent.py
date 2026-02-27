# pip install -U langchain langchain-google-genai langchain-community python-dotenv duckduckgo-search

# ========================================================================
# USING GOOGLE GEMINI (FREE TIER):
# -----------------------------------------------------------------------
# This agent uses ChatGoogleGenerativeAI (gemini-pro) instead of OpenAI.
# Make sure GOOGLE_API_KEY is set in your .env file.
# The free tier allows 15 requests per minute for gemini-pro.
# ========================================================================

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
import requests
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv

load_dotenv()  # expects GOOGLE_API_KEY in .env

search_tool = DuckDuckGoSearchRun()

@tool
def get_weather_data(city: str) -> str:
  """
  This function fetches the current weather data for a given city
  """
  url = f'https://api.weatherstack.com/current?access_key=f07d9636974c4120025fadf60678771b&query={city}'

  response = requests.get(url)

  return response.json()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# Step 2: Create the ReAct agent using langgraph
agent_executor = create_react_agent(
    model=llm,
    tools=[search_tool, get_weather_data],
)

# What is the release date of Dhadak 2?
# What is the current temp of gurgaon
# Identify the birthplace city of Kalpana Chawla (search) and give its current temperature.

# Step 3: Invoke
response = agent_executor.invoke({"messages": [{"role": "user", "content": "What is the current temp of gurgaon"}]})
print(response)

# Print the final AI message
print(response['messages'][-1].content)
