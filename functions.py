import operator
from config import config
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from utils import bcolors, get_tavily_search , get_duckduckgo_search, get_res
from langgraph.graph import MessagesState # type: ignore
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage # type: ignore
from nodes import GraderNode , GenerationNode
from langchain_groq import ChatGroq  # type: ignore #*********************************************************************
from dotenv import load_dotenv # type: ignore
from langchain.schema import Document  
import numpy as np
import pickle
from typing import Annotated, Literal
from pydantic import BaseModel, Field # type: ignore
import faiss
import re
from sentence_transformers import SentenceTransformer
import joblib
import time 

embedder = SentenceTransformer('all-MiniLM-L6-v2')
clf = joblib.load("intent_classifier.pkl")

def classify_intent_using_ml(query):
    query_lower = query.lower()
    if re.search(r"\b(hello|marhaba|hi|good|thank|joke|poem|fun|Keefak|peace upon you|Morning|Evening)\b", query_lower):
        return "no_rag"

    query_embedding = embedder.encode([query])[0]
    pred = clf.predict([query_embedding])[0]
    return "yes" if pred == 1 else "no"


load_dotenv()

os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")

class State(MessagesState):
    summary: str
    tavily_docs: str
    duckduck_docs: str
    documents: list
    question: str
    generation: str

class ClassifyQuery(BaseModel):
    """Binary score for intent classification."""
    binary_score: Literal["yes", "no"] = Field(
        description="User query intent needs to use RAG or not, 'yes' or 'no'"
    )

model_name = config['model_name']
classifier_name = config['classifier_name']
top_k = config['top_k']

embeddings_model_name = config['embedding_model_name']
model_kwargs = {'device': config['device']}  
encode_kwargs = {'normalize_embeddings': config['normalize_embeddings']}

embeddings = HuggingFaceEmbeddings(
    model_name=embeddings_model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
#vector_db = FAISS.load_local("my_vector_db", embeddings,allow_dangerous_deserialization=True)
# Load FAISS index and documents
index = faiss.read_index("my_vector_db.index")
with open("my_vector_db.pkl", "rb") as f:
    final_chunks = pickle.load(f)

generation_node = GenerationNode(model_name)
grader_node = GraderNode(model_name)
model = ChatGroq(model_name=model_name, temperature=0.2)
intent_classifier_model = ChatGroq(model_name=classifier_name, temperature=0)
intent_classifier_model = intent_classifier_model.with_structured_output(ClassifyQuery)


def intent_classifier(state: State , use_ml = True):
    print(bcolors.OKBLUE + 'Running the intent classifier..' + bcolors.ENDC)
    question = state.get("question",'')
    summary = state.get('summary','')

    rag_prompt = f'''You received the following query: "{question}". Determine if this query requires retrieving historical or cultural information from the database.

    - Answer "yes" if the query asks for detailed historical or cultural data.
    - Answer "no" if the query is casual, general, or does not require such data (e.g., greetings, jokes, or simple facts).

    Examples that should return "no":
    - What is the capital of Syria?
    - Hello.
    - Tell me a joke about Syria.
    - What is a delicious dish in Syria?

    Examples that should return "yes":
    - What cultural significance did wine hold in the ancient Near East, and how did it impact the daily lives of people in the region?
    - What was the total number of locations in the region after the additions during the Greco-Roman period?
    - How did the sultan's punitive expedition affect the villages in the region where the robbery took place?

    Return only a binary score "yes" or "no" with no additional text.'''
    no_rag = False
    if use_ml == True:
        pred = classify_intent_using_ml(question)
        if pred.lower() == 'yes':
            no_rag = False
        else:
            no_rag = True
    else:
        response = intent_classifier_model.invoke([SystemMessage(content=rag_prompt)])
        no_rag = False
        0
        if response.binary_score.lower() == 'yes':
            no_rag = False
        else:
            no_rag = True
    return {'question': question , 'messages': question, 'no_rag': no_rag}

# def retrieve(state: State):

#     print(bcolors.OKBLUE + 'Retrieving..' + bcolors.ENDC)

#     query = state["messages"][-1].content
    
#     docs = vector_db.similarity_search(query, k=top_k)

#     #context = "\n\n".join([doc.page_content for doc in docs])
#     return {'documents':docs}
def retrieve(state: State):
    print(bcolors.OKBLUE + 'Retrieving..' + bcolors.ENDC)
    query = state["messages"][-1].content
    query_embedding = np.array([embeddings.embed_query(query)], dtype=np.float32)
    _, indices = index.search(query_embedding, k=top_k)
    docs = [final_chunks[i] for i in indices[0] if i < len(final_chunks)]
    
    return {'documents': docs}



def summarization_router(state: State):
    messages = state['messages']
    last_message = state["messages"][-1]
    
    if len(messages) > 5:
        return 'summarize_conversation'
    return 'END'
def no_rag_router(state: State):
    no_rag = state['no_rag']
    if no_rag == False:
        return 'retrieve'
    else:
        return 'generate'
def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """
    search = state["search"]
    #no_rag = state['no_rag']
    #if no_rag == True:
    #    return 'generate'
    if search == "Yes":
        return "search"
    else:
        return "generate"
    
def decide_to_generateV2(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """
    search = state["search"]
    #no_rag = state['no_rag']
    #if no_rag == True:
    #    return 'generate'
    if search == "Yes":
        return ['tavily_search','duckduckgo_search']
    else:
        return "generate"
         
def summarize_conversation(state: State):
    print(bcolors.OKBLUE + 'Summarizing...' + bcolors.ENDC)

    summary = state.get("summary", "")
    if summary:
        
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
        
    else:
        summary_message = "Create a summary of the conversation above:"

    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = model.invoke(messages)
    
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"summary": response.content, "messages": delete_messages}

def generate(state: State):
    
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print(bcolors.OKBLUE + 'Generating...' + bcolors.ENDC)
    question = state["question"]
    documents = state.get("documents","")
    print(documents)
    summary = state.get("summary", "")
    if summary:
        
        system_message = f"Summary of conversation earlier: {summary}"

        messages = [SystemMessage(content=system_message)] + state["messages"]
    
    else:
        messages = state["messages"]
    if isinstance(documents,list):
        documents.extend(Document(page_content=state.get('tavily_docs','')))
        documents.extend(Document(page_content=state.get('duckduck_docs','')))
    
    generation = generation_node.generate(question , documents, messages)
    documents = []
    return {
        "documents": documents,
        "question": question,
        "generation": generation,
        "messages": generation
    }    

def grade_documents(state: State):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """
    print(bcolors.OKBLUE + 'Grading...' + bcolors.ENDC)
    question = state.get("question", "")
    documents = state.get("documents", "")
    filtered_docs = []
    search = "No"
    for d in documents:
        print(bcolors.WARNING + 'Grading the document content: \n' + str(d.page_content) +  bcolors.ENDC)
        grade = grader_node.grade(
            question = question,  doc_txt = d.page_content
        )
        if grade == "yes":
            filtered_docs.append(d)
        else:
            print(bcolors.FAIL + 'not related' + bcolors.ENDC)
            search = "Yes"
            continue
    return {
        "documents": filtered_docs,
        "question": question,
        "search": search,
    }
def web_search(state):
    """
    Web search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """
    print(bcolors.OKBLUE + 'Searching...' + bcolors.ENDC)
    question = state["question"]
    documents = state.get("documents", [])
    web_results = get_res(question,model_name)
    print(bcolors.OKBLUE + 'Returned web results: ' + bcolors.ENDC)
    print(bcolors.WARNING + web_results + bcolors.ENDC)
    documents.extend(
            Document(page_content=web_results)
    )
    return {"documents": documents, "question": question}


def tavily_search(state):

    print(bcolors.OKBLUE + 'Searching using Tavily...' + bcolors.ENDC)
    question = state["question"]
    web_results = get_tavily_search(question)
    print(bcolors.OKBLUE + 'Returned Tavily results: ' + bcolors.ENDC)
    print(bcolors.WARNING + web_results + bcolors.ENDC)
    return {"tavily_docs": web_results, "question": question}

def duckduckgo_search(state):

    print(bcolors.OKBLUE + 'Searching using duckdcukgo...' + bcolors.ENDC)
    question = state["question"]
    web_results = get_duckduckgo_search(question)
    print(bcolors.OKBLUE + 'Returned duckdcukgo results: ' + bcolors.ENDC)
    print(bcolors.WARNING + web_results + bcolors.ENDC)
    return {"duckduck_docs": web_results}

##########################################################################################


##########################################################################################

