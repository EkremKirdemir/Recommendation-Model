from pymongo import MongoClient
from pymongo.errors import PyMongoError

uri = "mongodb://localhost:27017/"
client = MongoClient(uri)
db = client['dataset']


source_collection = db['article']
destination_collection = db['keywords']

def gather_unique_keywords():
    keywords_set = set()

    try:
        documents = source_collection.find()
        print("Fetched documents from source collection")

        for doc in documents:
            if 'keywords' in doc:
                keywords = doc['keywords']
                if isinstance(keywords, list):
                    keywords_set.update(keywords)
                else:
                    print(f"Document with _id {doc['_id']} has 'keywords' not as a list: {keywords}")
            else:
                print(f"Document with _id {doc['_id']} has no 'keywords' field")

        unique_keywords_list = list(keywords_set)
        print("Unique keywords gathered:", unique_keywords_list)

        result = destination_collection.update_one(
            {'_id': 'unique_keywords'},
            {'$set': {'keywords': unique_keywords_list}},
            upsert=True 
        )
        print("Update result:", result.raw_result)

    except PyMongoError as e:
        print("An error occurred:", e)

if __name__ == "__main__":
    gather_unique_keywords()
    print("Unique keywords have been gathered and stored successfully.")
