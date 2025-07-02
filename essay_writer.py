import streamlit as st
import os
from dotenv import load_dotenv, find_dotenv
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage, ChatMessage
from langgraph.graph import StateGraph, END
from langchain_openai import AzureChatOpenAI
from typing import TypedDict, List, Annotated
from langchain_core.pydantic_v1 import BaseModel
from tavily import TavilyClient
from langgraph.checkpoint.memory import MemorySaver
import time

# Load environment variables
load_dotenv(find_dotenv(), override=True)

# Streamlit page configuration
st.set_page_config(
    page_title="Essay Writer AI",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'execution_steps' not in st.session_state:
    st.session_state.execution_steps = []
if 'essay_generated' not in st.session_state:
    st.session_state.essay_generated = False
if 'final_essay' not in st.session_state:
    st.session_state.final_essay = ""

# Agent State and Prompts (Original Code)
class AgentState(TypedDict):
    task: str
    plan: str
    draft: str
    critique: str
    content: List[str]
    revision_number: int
    max_revisions: int

# Original prompts
PLAN_PROMPT = '''You are an expert writer tasked with writing a high level outline of an essay.
Write such an outline for the user provided topic.
Give an outline of the essay along with any relevant notes or instructions for the sections.'''

RESEARCH_PLAN_PROMPT = '''You are a researcher charged with providing information that can be used when writing the following essay.
Generate a list of search queries that will gather any relevant information. Only generate 3 queries max.'''

WRITER_PROMPT = '''You are an essay assistant tasked with writing excellent 5-paragraph essays.
Generate the best essay possible for the user's request and the initial outline.
If the user provides critique, respond with a revised version of your previous attempts.
Utilize all the information below as needed:

------

{content}'''

REFLECTION_PROMPT = '''You are a teacher grading an essay submission.
Generate critique and recommendations for the user's submission.
Provide detailed recommendations, including requests for length, depth, style, etc.'''

RESEARCH_CRITIQUE_PROMPT = '''You are a researcher charged with providing information that can \
be used when making any requested revisions (as outlined below).
Generate a list of search queries that will gather any relevant information. Only generate 3 queries max.'''

class Queries(BaseModel):
    queries: List[str]

# Initialize components
@st.cache_resource
def initialize_components():
    try:
        model = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME_4"),
            azure_endpoint=os.getenv("AZURE_ENDPOINT"),
            api_version='2024-12-01-preview',
            temperature=0,
        )
        tavily = TavilyClient(api_key=os.environ.get('TAVILY_API_KEY'))
        return model, tavily
    except Exception as e:
        st.error(f"Error initializing components: {str(e)}")
        return None, None

model, tavily = initialize_components()

# Node functions with logging (Original logic preserved)
def add_execution_step(step_name, details=""):
    timestamp = time.strftime("%H:%M:%S")
    st.session_state.execution_steps.append({
        'timestamp': timestamp,
        'step': step_name,
        'details': details
    })

def plan_node(state: AgentState):
    add_execution_step("📋 Planning", "Creating essay outline...")
    messages = [
        SystemMessage(content=PLAN_PROMPT),
        HumanMessage(content=state['task'])
    ]
    response = model.invoke(messages)
    add_execution_step("✅ Planning Complete", f"Outline created ({len(response.content)} chars)")
    return {"plan": response.content}

def research_plan_node(state: AgentState):
    add_execution_step("🔍 Research Planning", "Generating search queries...")
    queries = model.with_structured_output(Queries).invoke([
        SystemMessage(content=RESEARCH_PLAN_PROMPT),
        HumanMessage(content=state['task'])
    ])
    content = state.get('content', [])
    
    add_execution_step("📊 Researching", f"Executing {len(queries.queries)} search queries...")
    for q in queries.queries:
        response = tavily.search(query=q, max_results=2)
        for r in response['results']:
            content.append(r['content'])
    
    add_execution_step("✅ Research Complete", f"Gathered {len(content)} research sources")
    return {"content": content}

def generation_node(state: AgentState):
    revision_num = state.get("revision_number", 0) + 1
    add_execution_step("✍️ Generating Essay", f"Creating draft (revision {revision_num})...")
    
    content = "\n\n".join(state['content'] or [])
    user_message = HumanMessage(
        content=f"{state['task']}\n\nHere is my plan:\n\n{state['plan']}")
    
    messages = [
        SystemMessage(content=WRITER_PROMPT.format(content=content)),
        user_message
    ]
    
    response = model.invoke(messages)
    add_execution_step("✅ Essay Generated", f"Draft {revision_num} complete ({len(response.content)} chars)")
    
    return {
        "draft": response.content,
        "revision_number": revision_num
    }

def reflection_node(state: AgentState):
    add_execution_step("🤔 Reflecting", "Analyzing essay for improvements...")
    messages = [
        SystemMessage(content=REFLECTION_PROMPT),
        HumanMessage(content=state['draft'])
    ]
    response = model.invoke(messages)
    add_execution_step("✅ Reflection Complete", "Critique and recommendations generated")
    return {"critique": response.content}

def research_critique_node(state: AgentState):
    add_execution_step("🔍 Research Critique", "Researching based on critique...")
    queries = model.with_structured_output(Queries).invoke([
        SystemMessage(content=RESEARCH_CRITIQUE_PROMPT),
        HumanMessage(content=state['critique'])
    ])
    content = state['content'] or []
    for q in queries.queries:
        response = tavily.search(query=q, max_results=2)
        for r in response['results']:
            content.append(r['content'])
    add_execution_step("✅ Critique Research Complete", f"Additional research gathered")
    return {'content': content}

def should_continue(state):
    if state['revision_number'] > state['max_revisions']:
        add_execution_step("🏁 Process Complete", f"Max revisions ({state['max_revisions']}) reached")
        return END
    add_execution_step("🔄 Continuing", "Moving to reflection phase...")
    return 'reflect'

# Build graph (Original structure preserved)
@st.cache_resource
def build_graph():
    if model is None or tavily is None:
        return None
        
    builder = StateGraph(AgentState)
    
    # Adding nodes to the graph
    builder.add_node('planner', plan_node)
    builder.add_node('generate', generation_node)
    builder.add_node('reflect', reflection_node)
    builder.add_node('research_plan', research_plan_node)
    builder.add_node('research_critique', research_critique_node)
    
    # Setting the entry point of the state graph
    builder.set_entry_point('planner')
    
    # Adding the conditional edge
    builder.add_conditional_edges(
        'generate',
        should_continue,
        {END: END, 'reflect': 'reflect'}
    )
    
    # Adding regular edges
    builder.add_edge('planner', 'research_plan')
    builder.add_edge('research_plan', 'generate')
    builder.add_edge('reflect', 'research_critique')
    builder.add_edge('research_critique', 'generate')
    
    memory = MemorySaver()
    return builder.compile(checkpointer=memory)

graph = build_graph()

# Streamlit UI
st.title("✍️ Essay Writer AI")
st.markdown("### AI-Powered Essay Writing with Multi-Agent System")

# Sidebar for execution steps
with st.sidebar:
    st.header("🔄 Execution Steps")
    
    if st.session_state.execution_steps:
        for i, step in enumerate(st.session_state.execution_steps):
            with st.container():
                st.markdown(f"**{step['timestamp']}** - {step['step']}")
                if step['details']:
                    st.caption(step['details'])
                st.markdown("---")
    else:
        st.info("No execution steps yet. Start by generating an essay!")
    
    if st.button("🗑️ Clear Steps"):
        st.session_state.execution_steps = []
        st.rerun()

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📝 Input")
    
    # User input form
    with st.form("essay_form"):
        task = st.text_area(
            "Essay Topic",
            placeholder="Enter your essay topic here...",
            height=100,
            help="Describe what you want the essay to be about"
        )
        
        max_revisions = st.slider(
            "Maximum Revisions",
            min_value=1,
            max_value=5,
            value=2,
            help="How many times should the essay be revised?"
        )
        
        submitted = st.form_submit_button("🚀 Generate Essay", type="primary")
        
        if submitted and task:
            if graph is None:
                st.error("❌ Error: Could not initialize the essay writer. Please check your API keys.")
            else:
                # Clear previous steps and results
                st.session_state.execution_steps = []
                st.session_state.essay_generated = False
                st.session_state.final_essay = ""
                
                # Show progress
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    thread = {'configurable': {'thread_id': str(time.time())}}
                    prompt = {
                        'task': task,
                        'max_revisions': max_revisions,
                        'revision_number': 0,
                    }
                    
                    status_text.text("Starting essay generation...")
                    progress_bar.progress(10)
                    
                    # Run the graph
                    events = list(graph.stream(prompt, thread))
                    
                    # Update progress
                    total_events = len(events)
                    for i, event in enumerate(events):
                        progress = 10 + (80 * (i + 1) / total_events)
                        progress_bar.progress(int(progress))
                        status_text.text(f"Processing step {i+1}/{total_events}...")
                        time.sleep(0.1)  # Small delay for visual effect
                    
                    # Get the final essay
                    final_event = events[-1] if events else {}
                    if 'generate' in final_event:
                        st.session_state.final_essay = final_event['generate']['draft']
                        st.session_state.essay_generated = True
                        progress_bar.progress(100)
                        status_text.text("✅ Essay generation complete!")
                    else:
                        st.error("❌ Could not generate essay. Please try again.")
                        
                except Exception as e:
                    st.error(f"❌ Error generating essay: {str(e)}")
                    progress_bar.empty()
                    status_text.empty()

with col2:
    st.header("📄 Output")
    
    if st.session_state.essay_generated and st.session_state.final_essay:
        st.success("✅ Essay Generated Successfully!")
        
        # Show word count
        word_count = len(st.session_state.final_essay.split())
        st.metric("Word Count", word_count)
        
        # Display the essay
        st.markdown("### Final Essay")
        st.markdown(st.session_state.final_essay)
        
        # Download button
        st.download_button(
            label="📥 Download Essay",
            data=st.session_state.final_essay,
            file_name="generated_essay.txt",
            mime="text/plain"
        )
        
    elif not st.session_state.essay_generated:
        st.info("👈 Enter a topic and click 'Generate Essay' to see your essay here!")
    else:
        st.warning("⚠️ Essay generation failed. Please try again.")

# Footer
st.markdown("---")
st.markdown("**Note**: Make sure you have set up your Azure OpenAI and Tavily API keys in your environment variables.")

# Environment variables check
with st.expander("🔧 Environment Setup"):
    azure_key = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME_4")
    azure_endpoint = os.getenv("AZURE_ENDPOINT")
    tavily_key = os.getenv("TAVILY_API_KEY")
    
    st.write("**Required Environment Variables:**")
    st.write(f"- AZURE_OPENAI_DEPLOYMENT_NAME_4: {'✅ Set' if azure_key else '❌ Not Set'}")
    st.write(f"- AZURE_ENDPOINT: {'✅ Set' if azure_endpoint else '❌ Not Set'}")
    st.write(f"- TAVILY_API_KEY: {'✅ Set' if tavily_key else '❌ Not Set'}")
    
    if not all([azure_key, azure_endpoint, tavily_key]):
        st.warning("⚠️ Some environment variables are missing. The app may not work correctly.")