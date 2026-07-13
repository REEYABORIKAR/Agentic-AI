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

import requests 
import sqlite3

load_dotenv()


llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct")

## Tools
search_tool= DuckDuckGoSearchRun(region="us-en")

@tool
def calculator(first_num: float, second_num: float, operation: str)-> dict:
    """
    Perform a basic arithematic operation on two numbers.
    Supported operations: add, sub, mul, div
    """

    try:
        if operation =="add":
            result= first_num + second_num
        elif operation =="sub":
            result= first_num - second_num
        elif operation =="mul":
            result= first_num * second_num
        elif operation =="div":
            if second_num == 0:
                return {"error":"Division by Zero is not allowed"}
            result= first_num / second_num
        else:
            return{"error":f"Unsupported operation '{operation}'"}
        
        return {'first_sum':first_num, "second_num":second_num, "operation":operation, "result":result}
    
    except Exception as e:
        return {"error": str(e)}
    
@tool
def get_stock_price(symbol: str)-> dict:
    """
    Fetch latedt stock price for a given symbol (e.g 'AAPL', 'TSLA')
    using Alpha Vantage with API key in the URl
    """

    url=f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=QT4UR41Q10DCKJ05"
    r= requests.get(url)
    return r.json()

SERVER={
    "github":{
        "transport":"studio",
        "command":"usr/bin/python3",
        "args":["/path/to/github_mcp_sferver.py"]
    }
}

## Make tool list
tools=[get_stock_price, search_tool, calculator]

# Make the LLM tool-aware
llm_with_tools= llm.bind_tools(tools)

## State
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage],add_messages]

## graph nodes
def chat_node(state: ChatState):
    """ 
    LLm node that may answer or request a tool call.
    """
    messages= state['messages']
    response= llm_with_tools.invoke(messages) 
    return {"messages":[response]}

tool_node= ToolNode(tools)

## Checkpointer
conn= sqlite3.connect(database="chatbot.db",check_same_thread=False)
checkpointer= SqliteSaver(conn=conn)

## graph structure
graph=StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools",tool_node)

graph.add_edge(START,"chat_node")

## If the LLm asked for a tool, go to ToolNode else finish
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools","chat_node")

chatbot=graph.compile(checkpointer=checkpointer)


## Helper
def retrieve_all_thread():
    all_threads=set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)