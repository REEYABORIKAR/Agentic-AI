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

## Make tool list
tools=[search_tool, calculator]

# Make the LLM tool-aware
llm_with_tools= llm.bind_tools(tools)

## State
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage],add_messages]


def built_graph():
   

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
    chatbot=built_graph()

    ## running thr graph
    result= await chatbot.ainvoke({"messages":[HumanMessage(content="Find the modulus od 132354 and 23 and "
    "give answer like cricket commentator")]})

    print(result['messages'][-1].content)

if __name__ == '__main__':
    asyncio.run(main())