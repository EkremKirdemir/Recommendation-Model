import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from pymongo import MongoClient
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import logging

logging.basicConfig(level=logging.INFO, filename='processing_log.log', filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s')

uri = "mongodb://localhost:27017/"
client = MongoClient(uri)
db = client['dataset']
collection = db['article']
tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    filtered_text = ' '.join([word for word in tokens if word.lower() not in stop_words])
    return filtered_text

def get_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def update_document(doc_id, embedding):
    collection.update_one({'_id': doc_id}, {'$set': {'embedding': embedding.tolist()}})

def process_documents():
    cursor = collection.find({})
    try:
        for doc in cursor:
            try:
                title = remove_stopwords(doc.get('title', ''))
                abstract = remove_stopwords(doc.get('abstract', ''))
                keywords = doc.get('keywords', [])

                embeddings = []
                if title:
                    embeddings.append(get_embeddings(title))
                if abstract:
                    embeddings.append(get_embeddings(abstract))
                for keyword in keywords:
                    filtered_keyword = remove_stopwords(keyword)
                    if filtered_keyword:
                        embeddings.append(get_embeddings(filtered_keyword))

                if embeddings:
                    average_embedding = np.mean(embeddings, axis=0)
                    update_document(doc['_id'], average_embedding)
                    logging.info(f"Updated document {doc['_id']} with new embedding.")
            except Exception as e:
                logging.error(f"Error processing document {doc['_id']}: {str(e)}")
    finally:
        cursor.close()
        logging.info("Processing complete.")

if __name__ == "__main__":
    process_documents()
