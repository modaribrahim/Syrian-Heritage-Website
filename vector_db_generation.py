import faiss
import numpy as np
import time
import pickle
from langchain.embeddings import HuggingFaceEmbeddings  # type: ignore
from config import config
from utils import clean_pdf_text, bcolors
import pymupdf4llm  # type: ignore
from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore
from langchain.schema import Document  # type: ignore
import pytesseract
from pdf2image import convert_from_path
from tqdm import tqdm

embeddings_model_name = config['embedding_model_name']
model_kwargs = {'device': config['device']}
encode_kwargs = {'normalize_embeddings': config['normalize_embeddings']}

print(bcolors.WARNING + 'Initializing embedding model...' + bcolors.ENDC)

embeddings = HuggingFaceEmbeddings(
    model_name=embeddings_model_name,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
print(bcolors.WARNING + 'Converting the pdf file to md...' + bcolors.ENDC)
books_path = '/home/modar/Desktop/'
books = ["ancient-syria.pdf" , "History_of_syria.pdf" ]

images1 = convert_from_path(f'{books_path}{books[0]}', first_page=26, last_page=338)
images2 = convert_from_path(f'{books_path}{books[1]}', first_page=24, last_page=300)
images3 = convert_from_path(f'{books_path}{books[1]}', first_page=301, last_page=450)
images4 = convert_from_path(f'{books_path}{books[1]}', first_page=451, last_page=730)
images = images1 + images2 + images3 + images4
text = ''

for img in tqdm(images):
    text += pytesseract.image_to_string(img)

clean_text = clean_pdf_text(text)

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=512,#384,
#     chunk_overlap=128,#100,
#     length_function=len,
#     separators=["\n\n", "\n", ". ", " ", ""]
# )
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=256,  #
    chunk_overlap=100,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""]
)

text_chunks = text_splitter.split_text(clean_text)
final_chunks = [Document(page_content=text) if isinstance(text, str) else text for text in text_chunks]

print(bcolors.WARNING + 'Generating the vector db...' + bcolors.ENDC)

text_embeddings = np.array([embeddings.embed_query(doc.page_content) for doc in final_chunks], dtype=np.float32)
n, d = text_embeddings.shape  
index = faiss.IndexFlatIP(d)  
index.add(text_embeddings)

faiss.write_index(index, "my_vector_db.index")

with open("my_vector_db.pkl", "wb") as f:
    pickle.dump(final_chunks, f)

print(bcolors.OKGREEN + 'Operation is done successfully' + bcolors.ENDC)




