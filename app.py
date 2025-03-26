from flask import Flask, request, jsonify, session
from flask_cors import CORS
import faiss
import numpy as np
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from huggingface_hub import InferenceClient
from pymongo import MongoClient
from datetime import datetime, timedelta
import atexit
import re
import threading 

# Lấy biến môi trường
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
HF_API_TOKEN = os.environ["HF_API_TOKEN"]
MONGO_URI = os.environ["MONGO_URI"]

# Khởi tạo Flask app và bật CORS
app = Flask(__name__)
app.secret_key = os.urandom(24)
CORS(app)

# Kết nối MongoDB
mongo_client = MongoClient(MONGO_URI)
db = mongo_client["chat_db"]
documents_collection = db["documents"]

# Khởi tạo model và client
genai_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GEMINI_API_KEY)
client = InferenceClient(api_key=HF_API_TOKEN)

# Dữ liệu mẫu
doc_texts = [
    "your name, who, called = Ope Watson",  
    "age, how old, birth year = 20",  
    "where, location, country = Hue, Vietnam",  
    "hobby, interests, like to do = sing, code, talk, photograph",  
    "job, work, profession, study = Computer Engineering",  
    "university, school, education = HCMUT",  
    "favorite food, like to eat = Bun dau mam tom, Banh deo, Che Hue",  
    "favorite color, color you like = yellow and blue",  
    "language, speak, talk = English, Japanese, Vietnamese",  
    "pet, animal, have pet = dont have",  
    "music, favorite song, like to listen = Tan Gai 101, Cat keo tren Lenin, Em gai, FLy me to the moon",  
    "book, favorite book, like to read = The Story Of A Seagull And The Cat Who Taught Her To Fly Book by Luis Sepúlveda",  
    "movie, film, favorite movie = Princess of Mononoke, Quintessential Quintuplets",  
    "anime, cartoon, favorite anime = Conan, Doraemon, ",  
    "goal, dream, future plan = inventor",  
    "relationship, girlfriend, love life = have 100 girlfriends",  
    "programming, coding, language you use = python, C++, JS",  
    "ai, machine learning, neural network = learning",  
    "exercise, workout, fitness = Callisthenic",  
    "game, video game, play = Minecraft, Brawlstar, AOV",  
]

# Lưu trữ phiên trò chuyện tạm thời (dùng username làm key)
sessions = {}
session_event = threading.Event()

# Biến toàn cục cho FAISS
doc_index = None
doc_embeddings = None
doc_texts_current = None

# Hàm lấy embeddings
def get_embeddings(texts):
    try:
        result = client.feature_extraction(
            texts,
            model="sentence-transformers/all-MiniLM-L6-v2"
        )
        return np.array(result)
    except Exception as e:
        raise Exception(f"Hugging Face API error: {str(e)}")

# Khởi tạo FAISS index
def initialize_index(username):
    global doc_index, doc_embeddings, doc_texts_current
    all_texts = doc_texts.copy()
    all_embeddings = []

    if documents_collection.count_documents({}) > 0:
        stored_docs = list(documents_collection.find({}))
        if stored_docs:
            all_embeddings = [np.array(doc["embedding"]) for doc in stored_docs]
            all_texts = [doc["text"] for doc in stored_docs]
    
    if len(all_embeddings) != len(doc_texts):
        documents_collection.drop()
        all_embeddings = []
        for text in doc_texts:
            embedding = get_embeddings([text])[0]
            documents_collection.insert_one({
                "text": text,
                "embedding": embedding.tolist()
            })
            all_embeddings.append(embedding)

    user_collection = db[username]
    convo_docs = list(user_collection.find({"embedding": {"$exists": True}}))
    for convo in convo_docs:
        summary_sentence = convo["summary_sentence"]
        all_embeddings.append(np.array(convo["embedding"]))
        all_texts.append(summary_sentence)

    doc_texts_current = all_texts
    doc_embeddings = np.array(all_embeddings)

    dimension = doc_embeddings.shape[1]
    doc_index = faiss.IndexFlatL2(dimension)
    doc_index.add(doc_embeddings)

# Hàm tóm tắt và lưu DB
def summarize_and_store(username):
    global doc_index, doc_embeddings, doc_texts_current
    if username not in sessions:
        return
    convo = sessions[username]["convo"]
    user_collection = db[username]
    
    conversation_text = "\n".join([
        m['parts'][0]['text'] for m in convo 
        if m['role'] == 'user' and not re.search(r'\b(what|how|where|when|why|who|which)\b|\?', m['parts'][0]['text'], re.IGNORECASE)
    ])

    prompt = f"""
    "Summarize the information about user in a '{username}, attribute : value' format in a single paragraph separated with a semicolon ';' . Capture separated key facts. Follow the format, do not use asterisks, all in lowercase"
    
    Example: {username},height = 20 ; {username}, age, how old, birth year = 20

    Conversation history:
    {conversation_text}
    """

    summary = genai_model.invoke(prompt).content
    print(f"summary {username}: {summary}")
    summary_sentences = [s.strip() for s in summary.split(";") if s.strip()]
    for sentence in summary_sentences:
        embedding = get_embeddings([sentence])[0]
        user_collection.insert_one({
            "summary_sentence": sentence,
            "embedding": embedding.tolist(),
            "timestamp": datetime.now()
        })

    del sessions[username]
    if not sessions:
        doc_index = None
        doc_embeddings = None
        doc_texts_current = None
        session_event.clear()

# Hàm lưu tất cả session khi server dừng
def save_all_sessions():
    for username in list(sessions.keys()):
        summarize_and_store(username)

atexit.register(save_all_sessions)

# API endpoint GET /summarize (mới)
@app.route('/summarize', methods=['GET'])
def summarize_endpoint():
    if not sessions:
        return jsonify({"message": "No active sessions to summarize"}), 200
    
    now = datetime.now()
    summarized_users = []
    for username in list(sessions.keys()):
        summarize_and_store(username)
        summarized_users.append(username)
    
    print(f"Summarized all sessions at {now}")
    return jsonify({
        "message": f"Summarized {len(summarized_users)} sessions",
        "summarized_users": summarized_users,
        "timestamp": now.strftime('%Y-%m-%d %H:%M:%S')
    }), 200

# Hàm truy xuất tài liệu
def retrieve_docs(query, top_k=1):
    if doc_index is None:
        return ["No FAISS index available, waiting for a new session!"]
    query_embedding = get_embeddings([query])
    distances, indices = doc_index.search(query_embedding, top_k)
    return [doc_texts_current[idx] for idx in indices[0]]

# Hàm sinh câu trả lời
def generate_response(username, query):
    if username not in sessions:
        return "Session expired, please enter your name again!"
    convo = sessions[username]["convo"]
    retrieved_docs = retrieve_docs(query, top_k=3)
    context = "\n".join(retrieved_docs)
    
    convo.append({"role": "user", "parts": [{"text": query}]})
    
    prompt = f"Below is the conversation history between the {username} and Ope Watson (you):\n"
    for message in convo:
        role = message["role"]
        text = message["parts"][0]["text"]
        if role == "user":
            prompt += f"{username}: {text}\n"
        elif role == "assistant":
            prompt += f"Ope Watson: {text}\n"
        elif role == "system":
            prompt += f"system: {text}\n"
    prompt += (
        f"Additional information about Ope Watson consider it as his memory about him and {username}. Ignore if this context is irrelevant: {context}\n"
        f"Now, respond to the {username}'s question: {query}"
    )

    response = genai_model.invoke(prompt).content
    convo.append({"role": "assistant", "parts": [{"text": response}]})
    sessions[username]["last_active"] = datetime.now()
    return response

# API endpoint POST /rag
@app.route('/rag', methods=['POST'])
def rag_endpoint():
    global doc_index
    data = request.get_json()
    username = data.get('username', '')
    query = data.get('query', '')
    if not username or not query:
        return jsonify({"error": "Username và query là bắt buộc!"}), 400
    
    if username not in sessions:
        if doc_index is None:
            initialize_index(username)
        sessions[username] = {
            "convo": [{"role": "system", "parts": [{"text": "You're playing as Ope Watson, a gentle boy, a chaotic friend, narcissistic, playful and humourous. Keep responses humanlike, short and on point! Do not use asterisks! Do not list up, only talk about one thing at a time. Do not answer summarization requests. Capitalize to emphasize! Answer in the language that users are using!"}]}],
            "last_active": datetime.now(),
            "username": username
        }
        session_event.set()

    try:
        response = generate_response(username, query)
        return jsonify({
            "query": query,
            "response": response,
            "retrieved_docs": retrieve_docs(query, top_k=1),
            "session_id": username
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# API endpoint GET /status
@app.route('/status', methods=['GET'])
def status_endpoint():
    return jsonify({
        "status": "Server is running",
        "current_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "active_sessions": len(sessions)
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)