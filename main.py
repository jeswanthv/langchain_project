from typing import List, Optional
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langsmith import Client
from langchain_tavily import TavilySearch
from dotenv import load_dotenv

load_dotenv()
client = Client()


class Source(BaseModel):
    """Schema for a source by the agent."""

    url: str = Field(description="The URL of the source.")


class AgentResponse(BaseModel):
    """Schema for the agent response."""

    answer: str = Field(description="The agent's answer to the question.")
    sources: Optional[List[Source]] = Field(
        default_factory=list,
        description="The list of sources used by the agent to answer the question."
    )


llm = ChatOllama(model="nemotron-3-nano", base_url="{{BASE_URL}}")
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


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
