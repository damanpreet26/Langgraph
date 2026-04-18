from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated, List
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver


# Load env
load_dotenv()

# LLM
llm = ChatOpenAI(model="gpt-4o-mini")


# State
class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


# Node
def chat_node(state: ChatState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


# Graph

checkpointer =  MemorySaver()

graph = StateGraph(ChatState)
graph.add_node("chat", chat_node)
graph.add_edge(START, "chat")
graph.add_edge("chat", END)

workflow = graph.compile(checkpointer=checkpointer)


# Run loop
print("Chat started (type 'exit' to quit)\n")
thread_id ='1'
while True:
    query = input("You: ")
    if query.strip().lower() in ["exit", "quit", "bye"]:
        break
    config = {"configurable": {"thread_id": thread_id}}
    result = workflow.invoke({
        "messages": [HumanMessage(content=query)]}, config=config)
    print("AI:", result["messages"][-1].content)
    
    
    
