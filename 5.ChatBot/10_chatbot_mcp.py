from langgraph.graph import StateGraph, START,END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver

from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct")

## MCP client for local FastMCP server
client= MultiServerMCPClient(
    {
        "arith":{
            "transport":"stdio",
            "command": "python",
            "args": ["C:/Users/Admin/Desktop/work/Agentic_AI/ChatBot/main.py"]
        },
        "expenses":{
            "transport":"http",
            "url":"https://splendid-gold-dingo.fastmcp.app/mcp"
        }
    }
)


## State
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage],add_messages]


async def built_graph():
    tools= await client.get_tools()
    print(tools)

    llm_with_tools=llm.bind_tools(tools)

    ## graph nodes
    async def chat_node(state: ChatState):
        """ 
        LLm node that may answer or request a tool call.
        """
        messages= state['messages']
        response= await llm_with_tools.ainvoke(messages) 
        return {"messages":[response]}

    tool_node= ToolNode(tools)

    ## graph structure
    graph=StateGraph(ChatState)
    graph.add_node("chat_node", chat_node)
    graph.add_node("tools",tool_node)

    graph.add_edge(START,"chat_node")

    ## If the LLm asked for a tool, go to ToolNode else finish
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools","chat_node")

    chatbot=graph.compile()

    return chatbot

async def main():
    chatbot= await built_graph()

    ## running thr graph
    result= await chatbot.ainvoke({"messages":[HumanMessage(content="Give me all expenses of the month of nov from 1 nov to 30 nov")]})

    print(result['messages'][-1].content)

if __name__ == '__main__':
    asyncio.run(main())