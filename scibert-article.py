import os
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from pymongo import MongoClient
from bson.objectid import ObjectId
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import logging

logging.basicConfig(level=logging.INFO, filename='processing_log.log', filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s')

nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

uri = "mongodb://localhost:27017/"
client = MongoClient(uri)
db = client['dataset']
collection = db['article']

tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')

def remove_stopwords(text):
    tokenizer_nltk = RegexpTokenizer(r'\w+')
    word_tokens = tokenizer_nltk.tokenize(text.lower())
    filtered_text = ' '.join([word for word in word_tokens if word not in stop_words])
    return filtered_text

def get_embeddings(text):
    tokens = tokenizer.tokenize(text)
    max_length = 512 - 2
    token_chunks = [tokens[i:i + max_length] for i in range(0, len(tokens), max_length)]
    embeddings = []

    for chunk in token_chunks:
        chunk_text = tokenizer.convert_tokens_to_string(chunk)
        inputs = tokenizer(chunk_text, return_tensors="pt", padding="max_length", max_length=512, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        chunk_embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        embeddings.append(chunk_embeddings)

    return embeddings

def update_document(doc_id, filtered_text, embeddings):
    average_embedding = np.mean(embeddings, axis=0)
    collection.update_one({'_id': ObjectId(doc_id)}, {'$set': {'filtered_article': filtered_text, 'embeddings': [emb.tolist() for emb in embeddings], 'average_embedding': average_embedding.tolist()}})

def process_articles():
    cursor = collection.find({}, no_cursor_timeout=True)
    try:
        for doc in cursor:
            try:
                if 'average_embedding' in doc:
                    logging.info(f"Skipping document {doc['_id']} as it already has an average embedding.")
                    continue
                article_text = doc['article']
                filtered_text = remove_stopwords(article_text)
                embeddings = get_embeddings(filtered_text)
                update_document(doc['_id'], filtered_text, embeddings)
                logging.info(f"Processed document {doc['_id']} successfully.")
            except Exception as e:
                logging.error(f"Error processing document {doc['_id']}: {str(e)}")
    finally:
        cursor.close()
if __name__ == "__main__":
    process_articles()
    logging.info("Processing complete.")
