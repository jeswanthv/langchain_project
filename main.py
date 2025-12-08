from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langsmith import Client
from langchain_tavily import TavilySearch
from dotenv import load_dotenv

load_dotenv()
client = Client()

llm = ChatGroq(model="llama-3.1-8b-instant")
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools)


def main():
    print("Hello from langchain-project!")
    response = agent.invoke({
        "messages": [
            HumanMessage(
                content="What's the weather like in Tokyo?")
        ]
    })
    print("Agent response:", response)


if __name__ == "__main__":
    main()
