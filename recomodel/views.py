import subprocess
from django.urls import reverse
from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.views.decorators.http import require_http_methods
from pymongo import MongoClient
from django.core.paginator import Paginator
from django.shortcuts import render, get_object_or_404
from bson import ObjectId
from django.shortcuts import redirect
from pymongo import MongoClient
from django.contrib import messages
import re
from django.contrib.auth.hashers import make_password, check_password
from .forms import KeywordSelectionForm
from bson import ObjectId
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

uri = "mongodb://localhost:27017/"
client = MongoClient(uri)
db = client['dataset']
keywords_collection = db['keywords']
articles_collection = db['article']
users_collection = db['users'] 
tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')

def get_keywords():
    document = keywords_collection.find_one({'_id': 'unique_keywords'})
    if document:
        return document.get('keywords', [])
    return []
def calculate_precision_recall_all_users():
    users = users_collection.find()
    total_tp = total_fp = total_fn = 0

    for user in users:
        if 'feedback' not in user:
            continue

        feedback = user['feedback']
        true_positive = set(feedback.get('true_positive', []))
        false_positive = set(feedback.get('false_positive', []))
        false_negative = set(feedback.get('false_negative', []))

        total_tp += len(true_positive)
        total_fp += len(false_positive)
        total_fn += len(false_negative)

    total_positive = total_tp + total_fp
    total_relevant = total_tp + total_fn

    precision = total_tp / total_positive if total_positive else 0
    recall = total_tp / total_relevant if total_relevant else 0

    return precision, recall


def profile(request):
    user_id = request.session.get('user_id')
    if not user_id:
        messages.error(request, "You must be logged in to access this page.")
        return redirect('login')

    precision, recall = calculate_precision_recall_all_users()
    user_profile = users_collection.find_one({'_id': ObjectId(user_id)})
    user_keywords = user_profile.get('keywords', []) if user_profile else []

    if request.method == 'POST':
        selected_keywords = request.POST.getlist('keywords')
        if len(selected_keywords) != 5:
            messages.error(request, "You must select exactly 5 keywords.")
        else:
            users_collection.update_one(
                {'_id': ObjectId(user_id)},
                {'$set': {'keywords': selected_keywords}},
                upsert=True
            )
            messages.success(request, "Your keywords have been updated.")
            return redirect('profile')

    keywords = get_keywords()
    return render(request, 'profile.html', {'keywords': keywords, 'user_keywords': user_keywords, 'precision': precision, 'recall': recall})

def get_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()


def extract_title(article_text):
    match = re.search(r'--T\s*(.*?)\s*--A', article_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return 'Untitled'

def calculate_user_embedding(user_profile):
    interest_keywords = user_profile.get('keywords', [])
    interest_embeddings = [get_embeddings(keyword) for keyword in interest_keywords]
    visited_embeddings = [np.array(e) for e in user_profile.get('visited_embeddings', [])] if 'visited_embeddings' in user_profile else []

    all_embeddings = interest_embeddings + visited_embeddings
    if all_embeddings:
        return np.mean(all_embeddings, axis=0)
    return None

def recommend_articles(user_embedding, articles_collection, top_n=5):
    articles = list(articles_collection.find({'embedding': {'$exists': True}}))
    if not articles:
        return []

    article_ids = [str(article['_id']) for article in articles]
    article_embeddings = [np.array(article['embedding']) for article in articles]

    similarities = cosine_similarity([user_embedding], article_embeddings)[0]
    top_indices = np.argsort(similarities)[-top_n:][::-1]

    recommended_articles = [(article_ids[i], extract_title(articles[i]['article'])) for i in top_indices]
    return recommended_articles

def search(request):
    user_id = request.session.get('user_id')
    user_profile = users_collection.find_one({'_id': ObjectId(user_id)}) if user_id else None

    if not user_id:
        messages.error(request, "You must be logged in to access this page.")
        return redirect('login')

    query = request.GET.get('query', '')
    if query:
        return redirect('search_results', query=query)

    if user_profile:
        user_embedding = calculate_user_embedding(user_profile)
        if user_embedding is not None:
            recommendations = recommend_articles(user_embedding, articles_collection)
        else:
            messages.error(request, "You must establish your interests first.")
            print('ateşe düştüm vaaah')
            return redirect('profile')
    else:
        recommendations = []

    return render(request, 'search.html', {'recommendations': recommendations})


def feedback(request, article_id, feedback_type):
    user_id = request.session.get('user_id')
    if not user_id:
        messages.error(request, "You must be logged in to provide feedback.")
        return redirect('login')

    user_profile = users_collection.find_one({'_id': ObjectId(user_id)})
    if not user_profile:
        messages.error(request, "User not found.")
        return redirect('search')

    feedback_types = ['true_positive', 'true_negative', 'false_positive', 'false_negative']
    if feedback_type not in feedback_types:
        messages.error(request, "Invalid feedback type.")
        return redirect('search')
    
    opposite_feedback_types = [ft for ft in feedback_types if ft != feedback_type]
    update_operations = {f'$pull': {f'feedback.{ft}': article_id} for ft in opposite_feedback_types}
    users_collection.update_one({'_id': ObjectId(user_id)}, update_operations)

    users_collection.update_one(
        {'_id': ObjectId(user_id)},
        {'$addToSet': {f'feedback.{feedback_type}': article_id}}
    )

    messages.success(request, "Thank you for your feedback!")
    return redirect('search')


def search_results(request, query):
    search_results = articles_collection.find({'article': {'$regex': f'--T\s*.*{query}.*\s*--A', '$options': 'i'}})
    search_results = [
        {
            'id_str': str(article['_id']),
            'paper_name': extract_title(article['article']),
            'paper_date': article.get('date', 'Unknown'),
            'paper_authors': article.get('authors', 'Unknown'),
            'paper_citations': article.get('citations', 0),
            'paper_abstract': article.get('abstract', 'No abstract available')
        } for article in search_results
    ]

    paginator = Paginator(search_results, 10)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    return render(request, 'search_results.html', {'page_obj': page_obj, 'query': query})

def extract_abstract(article_text):
    match = re.search(r'--A\s*(.*?)\s*--B', article_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return 'No abstract available'

def get_article_embedding(article_id):
    article = articles_collection.find_one({'_id': ObjectId(article_id)})
    if article and 'embedding' in article:
        return np.array(article['embedding'])
    return None

def paper_detail(request, article_id):
    user_id = request.session.get('user_id')
    if user_id:
        article_embedding = get_article_embedding(article_id)
        if article_embedding is not None:
            users_collection.update_one(
                {'_id': ObjectId(user_id)},
                {'$addToSet': {'visited_embeddings': article_embedding.tolist()}}
            )

    article = articles_collection.find_one({'_id': ObjectId(article_id)})
    article_text = article['article']
    document = {
        'title': extract_title(article_text),
        'abstract': extract_abstract(article_text),
        'keywords': article['keywords']
    }

    return render(request, 'paper_detail.html', {'document': document})

@require_http_methods(["GET"])
def home(request):
    

    return render(request, 'home.html', {
        'request': request
    })

def user_login(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = users_collection.find_one({'username': username})
        if user and check_password(password, user['password']):
            request.session['user_id'] = str(user['_id'])
            request.session['username'] = username
            messages.success(request, "You are now logged in.")
            return redirect('home')
        else:
            messages.error(request, "Invalid username or password.")
    return render(request, 'home.html')

def user_logout(request):
    try:
        del request.session['user_id']
        del request.session['username']
        request.session.modified = True
    except KeyError:
        pass
    messages.success(request, "You have been logged out.")
    return redirect('login')

def user_register(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        if users_collection.find_one({'username': username}):
            messages.error(request, "Username already exists.")
        else:
            hashed_password = make_password(password)
            users_collection.insert_one({'username': username, 'password': hashed_password})
            messages.success(request, "Your account has been created successfully!")
            return redirect('login')
    return render(request, 'home.html')