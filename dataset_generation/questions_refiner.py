import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import random
import re
from langchain_groq import ChatGroq  # type: ignore
from langchain_core.messages import SystemMessage  # type: ignore
from dotenv import load_dotenv
import time
from tqdm import tqdm
from utils import remove_think_content
load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

DATA_DIR = "/home/modar/Desktop/AI_chatbot/Q&A_data/need_rag/"
OUTPUT_DIR = "/home/modar/Desktop/AI_chatbot/Q&A_data/need_rag_refined/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PARAPHRASE_PROMPT = """
    I am refining a dataset of questions that need retrieval-augmented generation (RAG).
    Your task is to rephrase the given questions to make them sound more natural and human-like,
    while still ensuring they require RAG. Use variations like:
    - Tell me...
    - Wow, could you tell me more about...
    - I want to know about...
    - What do you think about...
    - Can you explain...
    - Do you have any insights on...
    - Give me details on...
    - I'm curious about...
    - Expand on...
    - How would you describe...
    - And leave some question as they are to maintain all cases (user may ask the question directly).
    
    Ensure the meaning remains the same, and keep responses in the following strict format:
    1- [Refined Question]
    2- [Refined Question]
    ...
    DO NOT INCLUDE ANYTHING ELSE.
    
    Here are the original questions:
    {questions}
"""

model = ChatGroq(model_name="llama3-8b-8192", temperature=0.5)


def refine_questions(file_path, output_path,think=False):
    df = pd.read_csv(file_path)
    questions = "\n".join([f"{i+1}- {q}" for i, q in enumerate(df['question'].tolist())])
    
    input_prompt = PARAPHRASE_PROMPT.format(questions=questions)
    result = model.invoke([SystemMessage(content=input_prompt)]).content
    if think == True:
        result = remove_think_content(result)
    pattern = r"\d+-\s*"
    refined_questions = re.split(pattern, result)
    refined_questions = [q.strip() for q in refined_questions if q.strip()]
    
    refined_df = pd.DataFrame({'question': refined_questions, 'label': ['rag'] * len(refined_questions)})
    refined_df.to_csv(output_path, index=False)
    time.sleep(1) 

for file in tqdm(os.listdir(DATA_DIR)):
    if file.endswith(".csv"):
        input_path = os.path.join(DATA_DIR, file)
        output_path = os.path.join(OUTPUT_DIR, file)
        print(f"Refining: {file}")
        refine_questions(input_path, output_path)

print("Refinement complete!")




