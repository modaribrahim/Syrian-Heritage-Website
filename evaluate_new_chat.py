import faiss
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pymupdf4llm
from config import config
from utils import clean_markdown, bcolors
import os

os.makedirs("plots", exist_ok=True)

embedding_models = [
    'sentence-transformers/all-MiniLM-L6-v2',
    'sentence-transformers/all-MiniLM-L12-v2'
]
chunk_sizes = [384, 512]
chunk_overlaps = [100, 128]
top_k_values = [1 , 2 , 3 , 4 , 5]

questions = [
    'What did Odenathus do when he marched upon Emesa?',
    'Who were given joint command of Macedon and mainland Greece after Alexander’s death, and what were their roles?',
    'How did Menelaos come to replace Jason as high priest, and what role did bribery play in that process?'
    'What events led to Syria becoming a Roman province, and how did Pompey accomplish this? And How did Caracalla attempt to unite the Roman and Parthian empires, and what was the outcome?',
    ''
]
ground_truth = [
    'Odenathus was not yet done. Returning to Syria, he marched upon the city of Emesa, where a pretender to the imperial throne, Quietus, had taken up residence. He assembled his armed hordes outside Emesa’s walls, and demanded the city’s surrender.',
    'Antipater, Alexander’s chief representative in Europe, and Craterus, his highest-ranking military officer, were given joint command of Macedon and the rest of mainland Greece',
    'Three years later, Jason was out of office and on the run. This happened after he had sent one of his subordinates, Menelaos, to Antiochus with funds to renew his bribe. Menelaos took the opportunity to outbid his master with a bigger bribe and was duly appointed high priest in his place.'
    '''Several years before Syria became a Roman province in 64 BC, the Roman commander Pompey had been wintering with his troops in Cilicia (67–66 BC) after successfully eradicating piracy from the Mediterranean and Black Seas. The People of Rome then assigned him a new task—to settle the political and military affairs of the eastern lands. He first dealt with the troublesome kingdoms of Pontus and Armenia in northern and eastern Anatolia through a combination of force and intimidation. Then, turning his attention to Syria, he entered its chief city, Antioch, without resistance, easily swept aside the last remnants of the Seleucid empire, and declared Syria a Roman province.
    Caracalla sought to merge the Roman and Parthian empires diplomatically by proposing marriage to a daughter of the Parthian king, Artabanus IV, similar to how Alexander the Great had married a daughter of Darius III to unite the eastern and western worlds. According to Cassius Dio, Artabanus rejected the proposal, prompting Caracalla to resume his offensive and raid Media. However, Herodian provides a more detailed account, stating that Caracalla persisted in his proposal, eventually convincing Artabanus through eloquence and gifts. But this was a deception—when the Parthians gathered for the wedding celebration, Caracalla signaled his troops to attack, leading to a massacre. Artabanus narrowly escaped with the help of his bodyguard. After the treacherous attack, Caracalla marched across Parthian territory, looting and plundering until his soldiers were exhausted, before finally returning to Mesopotamia.
    ''',

]

def generate_vector_db(embedding_model_name, chunk_size, chunk_overlap):
    db_name = f"vector_db_{embedding_model_name.split('/')[-1]}_{chunk_size}_{chunk_overlap}"
    index_filename = f"{db_name}.index"
    pkl_filename = f"{db_name}.pkl"

    if os.path.exists(index_filename) and os.path.exists(pkl_filename):
        print(f"Vector DB '{db_name}' already exists. Using existing files.")
        return db_name

    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={'device': config['device']},
        encode_kwargs={'normalize_embeddings': config['normalize_embeddings']}
    )

    md_text = pymupdf4llm.to_markdown("/home/modar/Desktop/ancient-syria.pdf")
    cleaned_md = clean_markdown(md_text)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    text_chunks = text_splitter.split_text(cleaned_md)
    final_chunks = [Document(page_content=text) if isinstance(text, str) else text for text in text_chunks]

    text_embeddings = np.array([embeddings.embed_query(doc.page_content) for doc in final_chunks], dtype=np.float32)
    n, d = text_embeddings.shape
    index = faiss.IndexFlatIP(d)
    index.add(text_embeddings)

    faiss.write_index(index, index_filename)
    with open(pkl_filename, "wb") as f:
        pickle.dump(final_chunks, f)

    print(f"Created new Vector DB '{db_name}'.")
    return db_name

def evaluate_model(db_name, embedding_model_name, top_k):
    index = faiss.read_index(f"{db_name}.index")
    with open(f"{db_name}.pkl", "rb") as f:
        text_chunks = pickle.load(f)

    model = SentenceTransformer(embedding_model_name)
    query_embeddings = model.encode(questions, convert_to_numpy=True)
    ground_truth_embeddings = model.encode(ground_truth, convert_to_numpy=True)

    n = index.ntotal
    corpus_embeddings = np.array([index.reconstruct(i) for i in range(n)])

    per_question_results = {}
    all_precisions = []
    all_recalls = []

    for i, query_embedding in enumerate(query_embeddings):
        gt_embedding = ground_truth_embeddings[i]
        sim_corpus = cosine_similarity(corpus_embeddings, [gt_embedding]).flatten()
        relevant_flags_full = sim_corpus >= 0.8
        total_relevant = np.sum(relevant_flags_full)

        distances, indices = index.search(query_embedding.reshape(1, -1), top_k)
        retrieved_relevant_flags = []
        for idx in indices[0]:
            sim_val = cosine_similarity([corpus_embeddings[idx]], [gt_embedding])[0][0]
            retrieved_relevant_flags.append(sim_val >= 0.8)
        retrieved_relevant_flags = np.array(retrieved_relevant_flags)

        precisions = []
        recalls = []
        for k in range(1, top_k+1):
            rel_in_top_k = np.sum(retrieved_relevant_flags[:k])
            precision_at_k = rel_in_top_k / k
            recall_at_k = rel_in_top_k / total_relevant if total_relevant > 0 else 0
            precisions.append(precision_at_k)
            recalls.append(recall_at_k)

        df_metrics = pd.DataFrame({
            "k": np.arange(1, top_k+1),
            "precision_at_k": precisions,
            "recall_at_k": recalls
        })
        per_question_results[questions[i]] = df_metrics
        all_precisions.append(precisions)
        all_recalls.append(recalls)

    avg_precisions = np.mean(all_precisions, axis=0)
    avg_recalls = np.mean(all_recalls, axis=0)
    df_avg = pd.DataFrame({
        "k": np.arange(1, top_k+1),
        "precision_at_k": avg_precisions,
        "recall_at_k": avg_recalls
    })

    return per_question_results, df_avg

def main():
    results = []
    for embedding_model in embedding_models:
        for chunk_size in chunk_sizes:
            for chunk_overlap in chunk_overlaps:
                db_name = generate_vector_db(embedding_model, chunk_size, chunk_overlap)
                for top_k in top_k_values:
                    per_question_results, df_avg = evaluate_model(db_name, embedding_model, top_k)
                    results.append({
                        "embedding_model": embedding_model,
                        "chunk_size": chunk_size,
                        "chunk_overlap": chunk_overlap,
                        "top_k": top_k,
                        "per_question_results": per_question_results,
                        "df_avg": df_avg
                    })

    for result in results:
        embedding_model = result["embedding_model"]
        chunk_size = result["chunk_size"]
        chunk_overlap = result["chunk_overlap"]
        top_k = result["top_k"]
        per_question_results = result["per_question_results"]
        df_avg = result["df_avg"]

        for question, df in per_question_results.items():
            plt.figure(figsize=(10, 5))
            plt.plot(df['k'], df['precision_at_k'], marker='o', label='Precision@K')
            plt.plot(df['k'], df['recall_at_k'], marker='o', label='Recall@K')
            plt.title(f'Precision and Recall@K for Query:\n"{question}"\nModel: {embedding_model}, Chunk Size: {chunk_size}, Overlap: {chunk_overlap}')
            plt.xlabel('K')
            plt.ylabel('Score')
            plt.legend()
            plt.grid(True)
            sanitized_question = "".join(c for c in question if c.isalnum())
            plt.savefig(f'plots/{embedding_model.split("/")[-1]}_{chunk_size}_{chunk_overlap}_{top_k}_{sanitized_question}.png')
            plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(df_avg['k'], df_avg['precision_at_k'], marker='o', label='Average Precision@K')
        plt.plot(df_avg['k'], df_avg['recall_at_k'], marker='o', label='Average Recall@K')
        plt.title(f'Average Precision and Recall@K\nModel: {embedding_model}, Chunk Size: {chunk_size}, Overlap: {chunk_overlap}')
        plt.xlabel('K')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'plots/{embedding_model.split("/")[-1]}_{chunk_size}_{chunk_overlap}_{top_k}_average.png')
        plt.close()

if __name__ == "__main__":
    main()
