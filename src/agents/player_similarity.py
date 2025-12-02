import operator
import json # <-- ADDED: Import for parsing tool output
from typing import TypedDict, Annotated, List
# FIX: Use direct Pydantic Field
from pydantic import Field 
from langchain_core.messages import BaseMessage
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# FIX: Use langchain_core for prompts
from langchain_core.prompts import ChatPromptTemplate 
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

# Load environment variables (OPENAI_API_KEY)
load_dotenv()

# --- Configuration ---
CHROMA_PATH = "chroma_db"
# Use a fast, capable model for tool-calling and generation
LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# --- Tool Imports ---
# NOTE: Ensure src/tools/stat_tool.py is correct and in the right path
from src.tools.stat_tool import get_statistical_match_tool


# --- 1. Define the LangGraph State ---
class PlayerSimilarityState(TypedDict):
    """
    Represents the data flow between all nodes in the graph.
    """
    query: str
    # FIX: This now holds the PARSED list of dictionaries, not a string list
    similarity_results: Annotated[List[dict], Field(description="Results from the statistical tool.")]
    tool_calls: List[BaseMessage]
    tool_output: str # Raw string output from the tool
    contextual_reports: Annotated[List[str], Field(description="Documents retrieved from the vector store.")]
    final_answer: str


# --- 2. Define the RAG Retrieval Node ---

def rag_retriever(state: PlayerSimilarityState):
    """
    Retrieves contextual reports from the Chroma vector store based on the user's query 
     AND parses the statistical tool output into a usable Python object.
    """
    print("---NODE: RAG Retriever---")
    query = state["query"]

    # --- RAG Retrieval Logic (Restored) ---
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

    docs = vector_store.similarity_search(query, k=4)
    
    # DEFINE contextual_reports
    contextual_reports = [doc.page_content for doc in docs] 
    print(f"Retrieved {len(contextual_reports)} contextual reports.")
    # -------------------------------------

    # --- Tool Output Parsing Logic (New/Corrected) ---
    # Get the raw string output from the tool executor
    raw_tool_output = state.get("tool_output", "[]") 
    
    # Attempt to parse the string output from the tool into a Python list/dict
    parsed_results = []
    
    # Check if the output is a string that looks like a list or dict (tool output)
    if isinstance(raw_tool_output, str) and (raw_tool_output.startswith('[') or raw_tool_output.startswith('{')):
        try:
            # Safely replace single quotes with double quotes for JSON compliance
            # Then load the JSON string into a Python object (list of dicts)
            parsed_results = json.loads(raw_tool_output.replace("'", '"'))
        except json.JSONDecodeError as e:
            # If parsing fails, store the error message
            print(f"JSON Parsing Error: {e}")
            parsed_results = [{"error": raw_tool_output}]
    elif "No tool call was initiated" not in raw_tool_output:
        # If the output isn't parsable JSON, but is a non-empty string, assume it's an error message.
        parsed_results = [{"error": raw_tool_output}]

    return {"contextual_reports": contextual_reports, "similarity_results": parsed_results} # <-- Returns the parsed list/error dict


# --- 3. Define the StatFinder Node (Tool Calling LLM) ---

tool_llm = LLM.bind_tools([get_statistical_match_tool])

def stat_finder(state: PlayerSimilarityState):
    """
    Tool-Calling LLM node. It uses the statistical tool if the query implies
    a numerical comparison or similarity search for a specific player.
    """
    print("---NODE: Stat Finder (Tool Caller)---")
    query = state["query"]
    
    response = tool_llm.invoke(query)
    
    # Check if the response contains tool calls 
    if response.tool_calls:
        # Access the name safely for printing
        tool_name = response.tool_calls[0]['name'] if isinstance(response.tool_calls[0], dict) else response.tool_calls[0].name
        print(f"LLM decided to call tool: {tool_name}")
        # Return the BaseMessage response object itself in the list
        return {"tool_calls": [response]} 
    else:
        print("LLM decided to proceed without tool call.")
        return {"tool_calls": []} 


# --- 4. Define Tool Executor Node ---

def tool_executor(state: PlayerSimilarityState):
    """
    Executes the tool call requested by the StatFinder node.
    """
    print("---NODE: Tool Executor---")
    tool_calls_messages = state["tool_calls"]
    
    if not tool_calls_messages:
        return {"tool_output": "No tool call was initiated."}

    # Extract the actual list of tool calls from the BaseMessage content
    tool_calls = tool_calls_messages[0].tool_calls
    
    if not tool_calls:
        return {"tool_output": "LLM response did not contain valid tool calls."}
        
    tool_call = tool_calls[0]
    
    # Access the name and arguments correctly 
    function_name = tool_call['name']
    tool_args = tool_call['args']

    # Execute the statistical match tool logic
    if function_name == get_statistical_match_tool.name:
        tool_result = get_statistical_match_tool.invoke(tool_args)
    else:
        tool_result = f"Error: Unknown tool '{function_name}'"

    print("Tool execution complete.")
    return {"tool_output": tool_result}


# --- 5. Define the Final Response Generator Node ---

def final_response_generator(state: PlayerSimilarityState):
    """
    Synthesizes the final response using results from the tool and RAG.
    """
    print("---NODE: Final Response Generator---")
    query = state["query"]
    # similarity_results is now a list of dicts or a list containing an error dict
    stat_results = state.get("similarity_results", [])
    reports = state["contextual_reports"]
    
    # Format the statistical results nicely for the prompt
    stat_output_str = ""
    if stat_results and isinstance(stat_results, list) and 'error' not in stat_results[0]:
        stat_output_str = "Top Statistical Matches:\n" + "\n".join([
            f"- {d['player']} (Similarity Score: {d['similarity_score']:.4f})" 
            for d in stat_results
        ])
    elif stat_results and isinstance(stat_results, list) and 'error' in stat_results[0]:
        stat_output_str = f"Error in statistical tool: {stat_results[0]['error']}"
    else:
        stat_output_str = "No statistical comparison was requested or available."
    
    # Define the prompt template for the final summary
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are the Football Similarity Engine, a highly professional AI scout. "
                "Your task is to answer the user's query by synthesizing information from "
                "the Statistical Tool results and the Contextual Reports provided. "
                "Structure your answer professionally, focusing on direct comparisons "
                "and actionable insights. "
                "\n\n--- INSTRUCTIONS ---"
                "\n1. Clearly state the statistical match results (if available). If there is an error in the tool, report it concisely."
                "\n2. Use the Contextual Reports to explain the 'why' behind the match or provide qualitative context for the target player."

    "[Image of the football field showing player positions] can be useful."
            ),
            ("human", "User Query: {query}\n\nStatistical Tool Output: {stat_results}\n\nContextual Reports:\n{reports}"),
        ]
    )
    
    # Compile and Invoke the chain
    chain = prompt | LLM 
    final_answer = chain.invoke({
        "query": query,
        "stat_results": stat_output_str,
        "reports": "\n---\n".join(reports)
    })
    
    print("Final response generated.")
    return {"final_answer": final_answer.content}


# --- 6. Define the Router Node ---

def router(state: PlayerSimilarityState) -> str:
    """
    This function routes the execution path based on whether the StatFinder 
    LLM decided to call a tool or not.
    """
    print("---NODE: Router---")
    tool_calls = state.get("tool_calls", [])
    
    if tool_calls:
        print("Routing to: execute_tool")
        return "execute_tool"
    else:
        print("Routing to: rag_search")
        return "rag_search"

# --- 7. Build the Graph ---

def build_graph():
    """Builds and compiles the LangGraph structure."""
    
    workflow = StateGraph(PlayerSimilarityState)

    # 1. Add Nodes
    workflow.add_node("stat_finder", stat_finder)
    workflow.add_node("rag_retriever", rag_retriever)
    workflow.add_node("tool_executor", tool_executor)
    workflow.add_node("final_generator", final_response_generator)

    # 2. Set the Entry Point
    workflow.set_entry_point("stat_finder")

    # 3. Add Edges and Conditional Edges
    
    # StatFinder output determines the next step: Tool or RAG
    workflow.add_conditional_edges(
        "stat_finder",  # Source node
        router,         # Router function
        {               # Possible destination nodes
            "execute_tool": "tool_executor",
            "rag_search": "rag_retriever",
        },
    )

    workflow.add_edge("tool_executor", "rag_retriever")
    workflow.add_edge("rag_retriever", "final_generator")
    workflow.add_edge("final_generator", END)

    # Compile the graph
    app = workflow.compile()
    print("Graph compiled successfully!")
    return app

# --- 8. Example Usage (Test the Full Graph) ---

if __name__ == "__main__":
    app = build_graph()
    
    # --- Test Case 1: Requires Tool Call (Statistical Comparison) ---
    print("\n\n--- RUNNING TEST CASE 1 (Tool Call Expected) ---")
    query_1 = "Find the three most statistically similar players to Mohamed Salah based on all available data."
    
    # Note: Initial state values must match the TypedDict schema
    result_1 = app.invoke({
        "query": query_1, 
        "similarity_results": [], 
        "tool_output": "", 
        "tool_calls": [], 
        "contextual_reports": []
    })
    
    print("\n\nFINAL RESULT 1:")
    print("--------------------------------------------------")
    print(result_1['final_answer'])

    # --- Test Case 2: Pure RAG (Contextual/Qualitative) ---
    print("\n\n--- RUNNING TEST CASE 2 (Pure RAG Expected) ---")
    query_2 = "What are the general scouting observations about Virgil van Dijk's leadership and technical ability?"

    result_2 = app.invoke({
        "query": query_2, 
        "similarity_results": [], 
        "tool_output": "", 
        "tool_calls": [], 
        "contextual_reports": []
    })
    
    print("\n\nFINAL RESULT 2:")
    print("--------------------------------------------------")
    print(result_2['final_answer'])