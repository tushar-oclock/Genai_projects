import os
import streamlit as st
from langchain_openai import ChatOpenAI
from crewai_tools import PDFSearchTool
from langchain_community.tools.tavily_search import TavilySearchResults
from crewai_tools import tool
from crewai import Crew, Task, Agent
import requests

# Set API Keys
os.environ["GROQ_API_KEY"] = "---"
os.environ['TAVILY_API_KEY'] = '----'

# Initialize LLM
llm = ChatOpenAI(
    openai_api_base="https://api.groq.com/openai/v1",
    openai_api_key=os.environ['GROQ_API_KEY'],
    model_name="llama3-8b-8192",
    temperature=0.1,
    max_tokens=1000,
)

# Download and process PDF
pdf_url = "https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf"
response = requests.get(pdf_url)
with open("attention_is_all_you_need.pdf", "wb") as file:
    file.write(response.content)

rag_tool = PDFSearchTool(pdf='attention_is_all_you_need.pdf',
    config=dict(
        llm=dict(
            provider="groq",
            config=dict(
                model="llama3-8b-8192",
            ),
        ),
        embedder=dict(
            provider="huggingface",
            config=dict(
                model="BAAI/bge-small-en-v1.5",
            ),
        ),
    )
)

web_search_tool = TavilySearchResults(k=3)

@tool
def router_tool(question):
    if 'self-attention' in question:
        return 'vectorstore'
    else:
        return 'web_search'

# Define Agents
Router_Agent = Agent(role='Router', goal='Route user question to a vectorstore or web search', verbose=True, allow_delegation=False, llm=llm)
Retriever_Agent = Agent(role="Retriever", goal="Use the information retrieved from the vectorstore to answer the question", verbose=True, allow_delegation=False, llm=llm)
Grader_agent = Agent(role='Answer Grader', goal='Filter out erroneous retrievals', verbose=True, allow_delegation=False, llm=llm)
hallucination_grader = Agent(role="Hallucination Grader", goal="Filter out hallucination", verbose=True, allow_delegation=False, llm=llm)
answer_grader = Agent(role="Answer Grader", goal="Filter out hallucination from the answer.", verbose=True, allow_delegation=False, llm=llm)

# Define Tasks
router_task = Task(description=("Analyse the keywords in the question {question} and decide whether it is eligible for a vectorstore search or a web search."), agent=Router_Agent, tools=[router_tool])
retriever_task = Task(description=("Retrieve information based on router task decision."), agent=Retriever_Agent, context=[router_task])
grader_task = Task(description=("Evaluate retrieved content relevance."), agent=Grader_agent, context=[retriever_task])
hallucination_task = Task(description=("Check if the answer is factually supported."), agent=hallucination_grader, context=[grader_task])
answer_task = Task(description=("If hallucination task approves, return a concise answer; otherwise, perform web search."), context=[hallucination_task], agent=answer_grader)

rag_crew = Crew(agents=[Router_Agent, Retriever_Agent, Grader_agent, hallucination_grader, answer_grader], tasks=[router_task, retriever_task, grader_task, hallucination_task, answer_task], verbose=True)

# Streamlit UI
st.set_page_config(page_title="RAG-based Q&A", layout="wide")
st.title("📖 RAG-based Question Answering System")
st.markdown("Type your question below and get AI-powered answers!")

question = st.text_input("Enter your question:")
if st.button("Get Answer"):
    if question:
        with st.spinner("Fetching the best answer..."):
            inputs = {"question": question}
            result = rag_crew.kickoff(inputs=inputs)
        st.subheader("Answer:")
        st.write(result)
    else:
        st.warning("Please enter a question!")

st.markdown("---")
st.markdown("**Built with LangChain, CrewAI, and Streamlit**")
