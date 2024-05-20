import os
from pymongo import MongoClient

def rename_key_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.key'):
            base = os.path.splitext(filename)[0]
            os.rename(os.path.join(directory, filename), os.path.join(directory, base + '.txt'))

uri = "mongodb://localhost:27017/"
client = MongoClient(uri)
db = client['dataset']
collection = db['article']

articles_dir = '/Users/kirdemir/Desktop/recomodel/Krapivin2009/docsutf8'
keywords_dir = '/Users/kirdemir/Desktop/recomodel/Krapivin2009/keys'


rename_key_files(keywords_dir)


for filename in os.listdir(articles_dir):
    if filename.endswith('.txt'):
        article_path = os.path.join(articles_dir, filename)
        with open(article_path, 'r', encoding='utf-8') as file:
            article_content = file.read()

        keywords_filename = os.path.splitext(filename)[0] + '.txt'
        keywords_path = os.path.join(keywords_dir, keywords_filename)
        if os.path.exists(keywords_path):
            with open(keywords_path, 'r', encoding='utf-8') as file:
                keywords_content = file.read().split()

            document = {
                'filename': filename,
                'article': article_content,
                'keywords': keywords_content
            }

            collection.insert_one(document)

print("Data insertion complete.")
