from typing import TypedDict

class Config(TypedDict):
    api_key: str
    model_name: str
    embedding_model_name: str
    device: str
    normalize_embeddings: bool

config: Config = {
'model_name': 'llama-3.3-70b-versatile',#'deepseek-r1-distill-llama-70b',#'llama3-70b-8192',#,'deepseek-r1-distill-llama-70b'#,
'embedding_model_name': 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1',#'sentence-transformers/all-MiniLM-L12-v2',
'device': 'cpu',
'normalize_embeddings': True,
'top_k': 5,
'classifier_name':'qwen-2.5-32b'
}






