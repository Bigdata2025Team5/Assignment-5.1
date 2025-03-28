import os
from getpass import getpass
from langchain_community.embeddings import OpenAIEmbeddings
import requests
import openai
from langchain_core.tools import tool
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Optional
import snowflake
import pandas as pd
# Initialize FastAPI
app = FastAPI()

# Define request model
class ReportRequest(BaseModel):
    query: str
    year: int = None  # Optional for non-year-based queries
    quarter: int = None  # Optional for non-quarter-based queries

@app.post("/generate_report")
def generate_report(request: ReportRequest):
    """
    Generates a research report based on the query and optional year/quarter.
    """
    try:
        # Invoke the LangGraph runnable
        output = runnable.invoke({
            "input": request.query,
            "chat_history": [],
        })

        # Format the output into a report
        report = build_report(output["intermediate_steps"][-1].tool_input)
        return {"report": report}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
openai.api_key = os.getenv("GPT4o_API_KEY")
openai_embeddings = OpenAIEmbeddings(api_key=os.getenv("GPT4o_API_KEY"), model="text-embedding-3-small")
from pinecone import Pinecone

# initialize connection to pinecone (get API key at app.pinecone.io)
api_key = os.getenv("PINECONE_API_KEY") 
# configure client
pc = Pinecone(api_key=api_key)
from pinecone import ServerlessSpec

spec = ServerlessSpec(
    cloud="aws", region="us-east-1"  # us-east-1
)

import time

index_name = "financial-reports"


# connect to index
index = pc.Index(index_name)
time.sleep(1)

from typing import TypedDict, Annotated, List, Union
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage
import operator


class AgentState(TypedDict):
    input: str
    chat_history: list[BaseMessage]
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]

@tool(description="Fetch valuation measures from Snowflake based on year and quarter.")
def fetch_snowflake_data(year, quarter):
    conn = snowflake.connector.connect(
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema=os.getenv("SNOWFLAKE_SCHEMA"),
        role="ACCOUNTADMIN"
    )

    cursor = conn.cursor()
    cursor.execute("USE ROLE ACCOUNTADMIN;")
    cursor.execute("USE DATABASE ASSIGNMENT;")
    cursor.execute("USE SCHEMA NVDA_STAGE;")

    query = f"""
        SELECT metric, value
        FROM "{os.getenv("SNOWFLAKE_DATABASE")}"."{os.getenv("SNOWFLAKE_SCHEMA")}"."{os.getenv("SNOWFLAKE_TABLE")}"
        WHERE year = {year} AND quarter = {quarter}
    """
    cursor.execute(query)
    data = cursor.fetchall()
    columns = [col[0] for col in cursor.description]
    df = pd.DataFrame(data, columns=columns)

    if df.empty:
        return {"summary": f"No financial data found for Q{quarter} {year}", "chart_data": []}

    df.columns = [col.lower() for col in df.columns]
    df["metric"] = df["metric"].str.strip().str.lower()

    def get_metric_value(name):
        row = df[df["metric"] == name.lower()]
        return row["value"].values[0] if not row.empty else "N/A"

    summary = (
        f"Valuation summary for Q{quarter} {year}:\n"
        f"• Market Cap: {get_metric_value('Market Cap')}\n"
        f"• Enterprise Value: {get_metric_value('Enterprise Value')}\n"
        f"• Trailing P/E: {get_metric_value('Trailing P/E')}\n"
        f"• Forward P/E: {get_metric_value('Forward P/E')}\n"
        f"• PEG Ratio (5yr expected): {get_metric_value('PEG Ratio (5yr expected)')}\n"
        f"• Price/Sales: {get_metric_value('Price/Sales')}\n"
        f"• Price/Book: {get_metric_value('Price/Book')}\n"
        f"• EV/Revenue: {get_metric_value('Enterprise Value/Revenue')}\n"
        f"• EV/EBITDA: {get_metric_value('Enterprise Value/EBITDA')}"
    )

    chart_metrics = ['trailing p/e', 'forward p/e', 'price/sales', 'price/book']
    df_chart = df[df["metric"].isin(chart_metrics)].copy()
    df_chart["metric"] = df_chart["metric"].str.title()
    df_chart["value_num"] = pd.to_numeric(df_chart["value"].str.replace("T", "e12", regex=False), errors="coerce")

    return {
        "summary": summary,
        "chart_data": df_chart[["metric", "value_num"]].to_dict(orient="records")
    }
       
@tool(description="Fetch valuation measures from Snowflake based on year and quarter.")
def web_search(query: str):
    try:
        # Use your real API key or secure retrieval method in production
        # api_key = os.getenv("WEB_SEARCH_API_KEY")
        api_key = "95f8f0a2bd9e0beb34b0b5fd74e6e2ed318495ac9d813985ed2707553483ec45"
        
        print(f"Using API Key: {api_key}")  # Debugging line
        
        # Perform the web search using the API
        response = requests.get(f"https://serpapi.com/search?q={query}+NVIDIA&api_key={api_key}")
        
        if response.status_code != 200:
            print(f"Failed to fetch data from SerpAPI. Status code: {response.status_code}")
            return {"insights": "Failed to fetch web insights."}
        
        data = response.json()
        
        # Check if organic results are present
        if "organic_results" in data and data["organic_results"]:
            top_result = data["organic_results"][0]
            insight = top_result.get("snippet", "No relevant insight found.")
            links = [result.get("link", "No link found.") for result in data["organic_results"]]
        else:
            insight = "No relevant insights found."
            links = []
        
        print(f"Fetched web data: {insight}")
        return {"insights": insight, "links": links}
    
    except Exception as e:
        # Log the error if the web request fails
        print(f"Error occurred while fetching web insights: {str(e)}")
        return {"insights": "Failed to fetch web insights due to an error.", "links": []}
    
    

from langchain_core.tools import tool

def format_rag_contexts(matches: list):
    contexts = []
    for x in matches:
        text = (
            f"quarter: {x['metadata']['quarter']}\n"
            f"year: {x['metadata']['year']}\n"
            f"text: {x['metadata']['text']}\n"
        )
        contexts.append(text)
    context_str = "\n---\n".join(contexts)
    return context_str

@tool("rag_search_filter")
def rag_search_filter(query: str, year: int, quarter: int):
    """Finds information from our ArXiv database using a natural language query
    and a specific ArXiv ID. Allows us to learn more details about a specific paper."""
    xq = openai_embeddings.embed_query(query)
    xc = index.query(vector=xq, top_k=6, include_metadata=True, filter={"year": year, "quarter": quarter})
    context_str = format_rag_contexts(xc["matches"])
    return context_str

@tool("rag_search")
def rag_search(query: str):
    """Finds specialist information on AI using a natural language query."""
    xq = openai_embeddings.embed_query(query)
    xc = index.query(vector=xq, top_k=4, include_metadata=True)
    context_str = format_rag_contexts(xc["matches"])
    return context_str

@tool("final_answer")
def final_answer(
    introduction: str,
    research_steps: str,
    main_body: str,
    conclusion: str,
    sources: str
):
    """Returns a natural language response to the user in the form of a research
    report. There are several sections to this report, those are:
    - ⁠ introduction ⁠: a short paragraph introducing the user's question and the
    topic we are researching.
    - ⁠ research_steps ⁠: a few bullet points explaining the steps that were taken
    to research your report.
    - ⁠ main_body ⁠: this is where the bulk of high quality and concise
    information that answers the user's question belongs. It is 3-4 paragraphs
    long in length.
    - ⁠ conclusion ⁠: this is a short single paragraph conclusion providing a
    concise but sophisticated view on what was found.
    - ⁠ sources ⁠: a bulletpoint list provided detailed sources for all information
    referenced during the research process
    """
    if type(research_steps) is list:
        research_steps = "\n".join([f"- {r}" for r in research_steps])
    if type(sources) is list:
        sources = "\n".join([f"- {s}" for s in sources])
    return ""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

system_prompt = """You are the oracle, the great AI decision maker.
Given the user's query you must decide what to do with it based on the
list of tools provided to you.

If you see that a tool has been used (in the scratchpad) with a particular
query, do NOT use that same tool with the same query again. Also, do NOT use
any tool more than twice (ie, if the tool appears in the scratchpad twice, do
not use it again).

You should aim to collect information from a diverse range of sources before
providing the answer to the user. Once you have collected plenty of information
to answer the user's question (stored in the scratchpad) use the final_answer
tool."""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("assistant", "scratchpad: {scratchpad}"),
])

from langchain_core.messages import ToolCall, ToolMessage
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o",
    openai_api_key=os.environ["GPT4o_API_KEY"],
    temperature=0
)

tools=[
    fetch_snowflake_data,
    rag_search_filter,
    rag_search,
    web_search,
    final_answer
]

# define a function to transform intermediate_steps from list
# of AgentAction to scratchpad string
def create_scratchpad(intermediate_steps: list[AgentAction]):
    research_steps = []
    for i, action in enumerate(intermediate_steps):
        if action.log != "TBD":
            # this was the ToolExecution
            research_steps.append(
                f"Tool: {action.tool}, input: {action.tool_input}\n"
                f"Output: {action.log}"
            )
    return "\n---\n".join(research_steps)

oracle = (
    {
        "input": lambda x: x["input"],
        "chat_history": lambda x: x["chat_history"],
        "scratchpad": lambda x: create_scratchpad(
            intermediate_steps=x["intermediate_steps"]
        ),
    }
    | prompt
    | llm.bind_tools(tools, tool_choice="any")
)



def run_oracle(state: list):
    print("run_oracle")
    print(f"intermediate_steps: {state['intermediate_steps']}")
    out = oracle.invoke(state)
    tool_name = out.tool_calls[0]["name"]
    tool_args = out.tool_calls[0]["args"]
    action_out = AgentAction(
        tool=tool_name,
        tool_input=tool_args,
        log="TBD"
    )
    return {
        "intermediate_steps": [action_out]
    }

def router(state: list):
    # return the tool name to use
    if isinstance(state["intermediate_steps"], list):
        return state["intermediate_steps"][-1].tool
    else:
        # if we output bad format go to final answer
        print("Router invalid format")
        return "final_answer"
    
tool_str_to_func = {
    "fetch_snowflake_data":fetch_snowflake_data,
    "rag_search_filter": rag_search_filter,
    "rag_search": rag_search,
    "web_search": web_search,
    "final_answer": final_answer
}

def run_tool(state: list):
    # use this as helper function so we repeat less code
    tool_name = state["intermediate_steps"][-1].tool
    tool_args = state["intermediate_steps"][-1].tool_input
    print(f"{tool_name}.invoke(input={tool_args})")
    # run tool
    out = tool_str_to_func[tool_name].invoke(input=tool_args)
    action_out = AgentAction(
        tool=tool_name,
        tool_input=tool_args,
        log=str(out)
    )
    return {"intermediate_steps": [action_out]}

from langgraph.graph import StateGraph, END

graph = StateGraph(AgentState)

graph.add_node("oracle", run_oracle)
graph.add_node("fetch_snowflake_data", run_tool)
graph.add_node("rag_search_filter", run_tool)
graph.add_node("rag_search", run_tool)
graph.add_node("web_search", run_tool)
graph.add_node("final_answer", run_tool)

graph.set_entry_point("oracle")

graph.add_conditional_edges(
    source="oracle",  # where in graph to start
    path=router,  # function to determine which node is called
)

# create edges from each tool back to the oracle
for tool_obj in tools:
    if tool_obj.name != "final_answer":
        graph.add_edge(tool_obj.name, "oracle")

# if anything goes to final answer, it must then move to END
graph.add_edge("final_answer", END)

runnable = graph.compile()

def build_report(output: dict):
    research_steps = output["research_steps"]
    if type(research_steps) is list:
        research_steps = "\n".join([f"- {r}" for r in research_steps])
    sources = output["sources"]
    if type(sources) is list:
        sources = "\n".join([f"- {s}" for s in sources])
    return f"""
INTRODUCTION
------------
{output["introduction"]}

RESEARCH STEPS
--------------
{research_steps}

REPORT
------
{output["main_body"]}

CONCLUSION
----------
{output["conclusion"]}

SOURCES
-------
{sources}
"""

# Define Pydantic Model for JSON Body Input
class AgentRequest(BaseModel):
    agent_name: str
    query: str
    year: Optional[int] = None
    quarter: Optional[int] = None

@app.post("/use-agent/")
async def use_agent(request: AgentRequest):
    """
    Endpoint to use a specific agent directly without LangGraph.
    """
    agent_name = request.agent_name
    query = request.query
    year = request.year
    quarter = request.quarter

    # Validate agent name
    available_agents = ["fetch_snowflake_data","rag_search", "rag_search_filter", "web_search"]
    if agent_name not in available_agents:
        raise HTTPException(
            status_code=400,
            detail=f"Agent '{agent_name}' not found. Available agents: {available_agents}"
        )

    # Construct tool input dynamically based on agent type
    tool_args = {"query": query}

    if agent_name == "rag_search_filter":
        if not year or not quarter:
            raise HTTPException(status_code=400, detail="Year and quarter are required for RAG Search Filter.")
        tool_args["year"] = year
        tool_args["quarter"] = quarter

    try:
        # Simulate tool execution (Replace with actual function call)
        result = f"Executed {agent_name} with query: {query}, year: {year}, quarter: {quarter}"

        return {
            "agent": agent_name,
            "query": query,
            "filters": tool_args,
            "result": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error invoking agent '{agent_name}': {str(e)}")
import re

def construct_filters_from_query(query: str):
    """
    Extracts year and quarter information from a natural language query.
    
    Example Inputs:
    - "What is the revenue for 2024 Q1?"
    - "Show me the financials of NVIDIA in Q3 2023."
    
    Returns:
    - {"year": 2024, "quarter": 1}
    - {"year": 2023, "quarter": 3}
    - {} (if no filters found)
    """
    filters = {}

    # Regular expressions to find year (2000-2099)
    year_match = re.search(r'\b(20[0-9]{2})\b', query)
    if year_match:
        filters["year"] = int(year_match.group(1))

    # Regular expressions to find quarter (Q1, Q2, Q3, Q4)
    quarter_match = re.search(r'\bQ([1-4])\b', query, re.IGNORECASE)
    if quarter_match:
        filters["quarter"] = int(quarter_match.group(1))

    return filters

class AgentRequest(BaseModel):
    agent_name: str
    query: str
    year: Optional[int] = None
    quarter: Optional[int] = None
@app.post("/use-agent/")
async def use_agent(request: AgentRequest):
    if not request.agent_name or not request.query:
        raise HTTPException(status_code=400, detail="Missing required fields: agent_name and query are required.")
    
    # Validate agent name
    available_agents = ["fetch_snowflake_data","rag_search", "rag_search_filter", "web_search", "snowflake_agent"]
    if request.agent_name not in available_agents:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid agent '{request.agent_name}'. Choose from {available_agents}."
        )

    # Construct tool input dynamically
    tool_args = {"query": request.query}
    if request.agent_name == "rag_search_filter" and (not request.year or not request.quarter):
        raise HTTPException(status_code=400, detail="Year and quarter are required for 'rag_search_filter'.")

    try:
        result = f"Executed {request.agent_name} with query: {request.query}, year: {request.year}, quarter: {request.quarter}"
        return {"agent": request.agent_name, "query": request.query, "filters": tool_args, "result": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error invoking agent: {str(e)}")