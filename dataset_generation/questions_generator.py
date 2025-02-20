import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import config
from utils import clean_pdf_text
import pymupdf4llm # type: ignore
from langchain.text_splitter import RecursiveCharacterTextSplitter # type: ignore
from langchain.schema import Document   # type: ignore
import pickle
from utils import bcolors , remove_think_content
from langgraph.graph import MessagesState # type: ignore
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage , RemoveMessage # type: ignore
from pydantic import BaseModel, Field # type: ignore
from langchain_groq import ChatGroq  # type: ignore #*********************************************************************
from typing import Literal 
import os
import pytesseract
from pdf2image import convert_from_path
from tqdm import tqdm
from dotenv import load_dotenv
import re
import pandas as pd
import warnings
import time
warnings.filterwarnings("ignore", category=UserWarning)  # Suppress user warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)  # Suppress deprecation warnings

general_QA_prompt = '''
I am building a dataset for an intent classifier with two classes:  
1️⃣ **"need rag"** (questions that require retrieving information from documents)  
2️⃣ **"no need rag"** (questions that can be answered directly by a general chatbot).  

Your task is to generate **"no need rag"** questions, specifically about **Syrian culture, history, and general knowledge**.  

### **Instructions:**  
You must generate **diverse** sets of **simple** questions and **casual conversational phrases** related to Syrian culture, history, and general interactions, that can be answered **without retrieval**. Avoid queries that require document or fact-based retrieval.

#### **1️⃣ Question Types**  
- **General Knowledge Questions** (short and simple, focused on common knowledge)  
  - Example: "Who was the first ruler of Aleppo?"  
  - Example: "When was the Umayyad Mosque built?"  
- **Conversational & Interactive Queries**  
  - Example: "Tell me something interesting about Syria."  
  - Example: "What do you think about ancient Syrian architecture?"  
- **Casual Phrases in Chat Flow**  
  - Example: "Hello!"  
  - Example: "Good morning, how are you?"  
  - Example: "Wow, that’s interesting!"  
  - Example: "Thank you!"  
  - Example: "What’s your favorite Syrian food?"  

#### **2️⃣ Guidelines for Questions:**  
✅ Keep them **simple**, **conversational**, and **easy to answer** without relying on documents or specific references.  
✅ **DO NOT** ask questions requiring obscure, highly specific, or book-dependent knowledge.  
✅ Avoid using **academic or highly historical questions** that necessitate retrieval.  
✅ Ensure **variety** in sentence structure (e.g., What, How, When, Tell me, Describe, Greetings, Expressing interests).  
✅ Mix between **long and short** questions.  
✅ **DO NOT** ask about the **Syrian Civil War**.  

#### **3️⃣ Conversational Phrases to Avoid:**  
To make sure there is no confusion, **avoid** questions or greetings that could be misinterpreted as needing document-based answers:
- **"Tell me"** and similar constructions should focus on **simple** information, not detailed facts (e.g., "Tell me about Syria" is fine; "Tell me about the collapse of the Aleppo economy in the 15th century" is not).  
- **Casual greetings** and conversational phrases like **"Good morning"**, **"Hello!"**, **"How are you?"**, **"What’s up?"**, or **"Tell me a joke"** should be included as **no need rag**.  

### **Response Format (Strictly Follow This Format AND ONLY THIS Format, DO NOT REPLY WITH SOMETHING ELSE)**  
1- [Question]  
2- [Question]  
3- [Question]  
...

**DO NOT** include explanations, extra formatting, or any other text.  

'''
general_QA_prompt_rephrase = '''
I am building a dataset for an **intent classifier** with two classes:  
1️⃣ **"need rag"** → Questions that require retrieving information from documents.  
2️⃣ **"no need rag"** → Questions that can be answered **directly** by a general chatbot.  

You will receive an existing dataset and must **augment it** by generating **new** questions or rephrasing existing ones while strictly following the instructions below.  

---
## 🔹 **Task: Generate "No Need RAG" Questions**  
You must generate **diverse**, **simple**, and **conversational** questions related to **Syrian culture, history, and general knowledge** that **DO NOT** require document retrieval.  

---
## 🔹 **Question Types to Generate**  
✅ **General Knowledge (Directly Answerable)**  
- Example: *"Who was the first ruler of Aleppo?"*  
- Example: *"When was the Umayyad Mosque built?"*  

✅ **Casual & Conversational Queries**  
- Example: *"Tell me something interesting about Syria."*  
- Example: *"What do you think about Syrian architecture?"*  

✅ **Short Chatbot Phrases**  
- Example: *"Hello!"*  
- Example: *"What’s your favorite Syrian dish?"*  
- Example: *"Wow, that’s amazing!"*  

---
## 🔹 **Guidelines**  
✔ **Keep it simple** → No complex, academic, or fact-heavy questions.  
✔ **Ensure variety** → Mix **What, How, When, Tell me, Describe** questions with **greetings, expressions, and casual prompts**.  
✔ **No dependency on external documents** → Questions should be **answerable immediately**.  
✔ **DO NOT ask about the Syrian Civil War**.  
✔ **Mix long & short questions** to improve dataset diversity.  

---
## 🔹 **Avoid These Mistakes**  
❌ **No document-based or research-heavy questions**.  
- 🚫 *"Tell me about the collapse of Aleppo’s economy in the 15th century."*  
- 🚫 *"What were the key causes of the Mamluk-Ottoman conflicts in Syria?"*  

❌ **No retrieval-dependent historical or political questions**.  
- 🚫 *"Who wrote the oldest book on Syrian history?"*  
- 🚫 *"How did trade routes affect Syria in the 12th century?"*  

❌ **No overly vague or unanswerable questions**.  
- 🚫 *"What is the meaning of life?"*  
- 🚫 *"Can you tell me everything about Syria?"*  

---
## 🔹 **Response Format (STRICTLY FOLLOW THIS FORMAT)**  
1- [Question]  
2- [Question]  
3- [Question]  
…  

⚠ **DO NOT** include explanations, bullet points, extra formatting, or any text outside the required format. 


---

🔹 **Why This is Better?**  
✅ **Highly constrained yet flexible** → Ensures high-quality data generation.  
✅ **Prevents unwanted outputs** → Limits irrelevant or retrieval-based questions.  
✅ **Optimized for NLP models** → Clear task definition, examples, and strict output structure.  

The file data: {data}
'''

QA_genration_prompt = '''
    Generate a list of high-quality questions based on the given text to train a RAG system.
    - The questions should be designed as if asked to a model who cannot see the text, so each question should be clear and self-contained, every term should be clear and mentioning ambiguos terms.
    - Avoid referencing specific entities, authors, or sections of the text.
    - The questions should be varied in type: some should require short, specific answers, and others should require more detailed responses.
    - Keep the questions concise and focused, ensuring they are suitable for assessing comprehension and critical thinking.
    - The questions should only and only focus on heritage, history and culture.
    - Don't write something like 'according to the text' in any of the questions, there is not text seen by the model we are testing, it only sees your questions separately from each other.
    ANSWER WITH THIS FORMAAT AND ONLY THIS FORMAT:
    1- [Question]
    2- [Question]
    ...
    DO NOT WRITE ANYTHING IN THE RESPONSE OTHER THAN THE MENTIONED FORMAT.
    ### **Text to generate questions on:**
    {text}
'''

print(bcolors.WARNING + 'Converting the pdf file to md...\n' + bcolors.ENDC)
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

def generate_dataset(rag=True,think=False,rephrase=False):

    model = ChatGroq(model_name=config['model_name'], temperature=0.3)

    if rag == True:
        books_path = '/home/modar/Desktop/'
        books = ["ancient-syria.pdf" , "History_of_syria.pdf" ]
        text_name = 'Final_text.txt'
        if not os.path.exists(books_path+text_name):
            images1 = convert_from_path(f'{books_path}{books[0]}', first_page=26, last_page=338)
            print('f1')
            images2 = convert_from_path(f'{books_path}{books[1]}', first_page=24, last_page=300)
            print('f2')
            images3 = convert_from_path(f'{books_path}{books[1]}', first_page=301, last_page=450)
            print('f3')
            images4 = convert_from_path(f'{books_path}{books[1]}', first_page=451, last_page=730)
            print('f4')
            images = images1 + images2 + images3 + images4
            text = ''

            for img in tqdm(images):
                text += pytesseract.image_to_string(img)

            clean_text = clean_pdf_text(text)
            
            text_file = open(books_path + text_name, "w")
            text_file.write(str(clean_text))
            text_file.close()
        else:
            text_file = open(books_path + text_name, "r")
            clean_text = text_file.read()
            text_file.close()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,#384,
            chunk_overlap=128,#100,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        text_chunks = text_splitter.split_text(clean_text)
        final_chunks = [Document(page_content=text) if isinstance(text, str) else text for text in text_chunks]

        n = int(len(final_chunks) / 18) 
       
        for i in tqdm(range(n)):
            input = QA_genration_prompt.format(text=final_chunks[i*18 + 89])
            result = model.invoke([SystemMessage(content=input)]).content
            if think == True:
                result = remove_think_content(result)
            pattern = r"\d+\-\s*"  
            data = re.split(pattern, result)
            data = [q.strip() for q in data if q.strip()]
            labels = ['rag' for _ in range(len(data))]
            file = pd.DataFrame({'question':data,'label':labels})
            file.to_csv(f'/home/modar/Desktop/AI_chatbot/Q&A_data/need_rag/data_{i+89}.csv')
            time.sleep(2)
    else:
        n = 63
        if rephrase == False:
            for i in tqdm(range(n)):
                input = general_QA_prompt
                result = model.invoke([SystemMessage(content=input)]).content
                pattern = r"\d+\-\s*"  
                if think == True:
                    result = remove_think_content(result)
                data = re.split(pattern, result)
                data = [q.strip() for q in data if q.strip()]
                labels = ['no_rag' for _ in range(len(data))]
                file = pd.DataFrame({'question':data,'label':labels})
                file.to_csv(f'/home/modar/Desktop/AI_chatbot/Q&A_data/no_need_rag/data_{i}.csv')
                time.sleep(2.5)
        else:
            for i , file in tqdm(enumerate(os.listdir('/home/modar/Desktop/AI_chatbot/Q&A_data/no_need_rag'))):
                df = pd.read_csv('/home/modar/Desktop/AI_chatbot/Q&A_data/no_need_rag/' + file)
                questions = "\n".join([f"{i+1}- {q}" for i, q in enumerate(df['question'].tolist())])
                input = general_QA_prompt_rephrase.format(data=questions)

                result = model.invoke([SystemMessage(content=input)]).content
                pattern = r"\d+\-\s*"  
                if think == True:
                    result = remove_think_content(result)
                data = re.split(pattern, result)
                data = [q.strip() for q in data if q.strip()]
                labels = ['no_rag' for _ in range(len(data))]
                file = pd.DataFrame({'question':data,'label':labels})
                file.to_csv(f'/home/modar/Desktop/AI_chatbot/Q&A_data/no_need_rag/data_{n+i}.csv')
                time.sleep(2.5)

generate_dataset(rag=False,think=True,rephrase=True)