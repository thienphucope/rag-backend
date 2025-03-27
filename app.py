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
ope_watson_collection = db["ope_watson"]

# Khởi tạo model và client
genai_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GEMINI_API_KEY)
client = InferenceClient(api_key=HF_API_TOKEN)

# Lưu trữ phiên trò chuyện tạm thời (dùng username làm key)
sessions = {}  # {username: {"convo": list, "last_active": datetime, "username": str, "faiss": {"index": faiss.Index, "embeddings": np.array, "texts": list}}}
session_event = threading.Event()

# Biến toàn cục cho FAISS của Ope Watson
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
    # Lấy dữ liệu từ bảng ope_watson, sắp xếp theo ID
    ope_docs = list(ope_watson_collection.find({}).sort("id", 1))
    all_doc_texts = [doc["text"] for doc in ope_docs]
    all_doc_ids = [doc["id"] for doc in ope_docs]
    all_doc_embeddings = []

    # Lấy dữ liệu từ documents_collection
    stored_docs = list(documents_collection.find({}).sort("id", 1))
    stored_dict = {}
    
    if stored_docs and "id" not in stored_docs[0]:
        print("Old data detected in documents_collection without 'id'. Dropping and recomputing.")
        documents_collection.drop()
        stored_docs = []
        stored_dict = {}
    else:
        stored_dict = {doc["id"]: {"text": doc["text"], "embedding": doc["embedding"]} for doc in stored_docs}

    # So sánh và cập nhật embedding nếu cần
    needs_recompute = False
    for ope_doc in ope_docs:
        doc_id = ope_doc["id"]
        doc_text = ope_doc["text"]
        if doc_id not in stored_dict:
            embedding = get_embeddings([doc_text])[0]
            documents_collection.insert_one({
                "id": doc_id,
                "text": doc_text,
                "embedding": embedding.tolist()
            })
            all_doc_embeddings.append(embedding)
            needs_recompute = True
        elif stored_dict[doc_id]["text"] != doc_text:
            embedding = get_embeddings([doc_text])[0]
            documents_collection.update_one(
                {"id": doc_id},
                {"$set": {"text": doc_text, "embedding": embedding.tolist()}}
            )
            all_doc_embeddings.append(embedding)
            needs_recompute = True
        else:
            all_doc_embeddings.append(np.array(stored_dict[doc_id]["embedding"]))

    if needs_recompute:
        print(f"Recomputed embeddings for {sum(1 for x in ope_docs if x['id'] not in stored_dict or stored_dict[x['id']]['text'] != x['text'])} changed/new records.")
    elif not stored_docs:
        documents_collection.drop()
        all_doc_embeddings = []
        for doc in ope_docs:
            embedding = get_embeddings([doc["text"]])[0]
            documents_collection.insert_one({
                "id": doc["id"],
                "text": doc["text"],
                "embedding": embedding.tolist()
            })
            all_doc_embeddings.append(embedding)

    doc_texts_current = all_doc_texts
    doc_embeddings = np.array(all_doc_embeddings)

    dimension = doc_embeddings.shape[1]
    doc_index = faiss.IndexFlatL2(dimension)
    doc_index.add(doc_embeddings)

    # Khởi tạo FAISS index cho user và lưu vào sessions
    user_collection = db[username]
    convo_docs = list(user_collection.find({"embedding": {"$exists": True}}))
    all_user_texts = [convo["summary_sentence"] for convo in convo_docs]
    all_user_embeddings = [np.array(convo["embedding"]) for convo in convo_docs]

    if all_user_embeddings:
        user_embeddings = np.array(all_user_embeddings)
        user_index = faiss.IndexFlatL2(dimension)
        user_index.add(user_embeddings)
        sessions[username]["faiss"] = {
            "index": user_index,
            "embeddings": user_embeddings,
            "texts": all_user_texts
        }
    else:
        sessions[username]["faiss"] = None

# Hàm tóm tắt và lưu DB
def summarize_and_store(username):
    # if username not in sessions:
    #     return
    # convo = sessions[username]["convo"]
    # user_collection = db[username]
    
    # conversation_text = "\n".join([
    #     m['parts'][0]['text'] for m in convo 
    #     if m['role'] == 'user' and not re.search(r'\b(what|how|where|when|why|who|which)\b|\?', m['parts'][0]['text'], re.IGNORECASE)
    # ])

    # prompt = f"""
    # "Summarize the information about user in a '{username}, attribute, synonyms : value' format in a single paragraph separated with a semicolon ';'. For each attribute, list additional synonyms that may be used to refer to that attribute (e.g., {username}, age, old, how old, birth year = 20; {username}, name, called, who = {username}). Capture separated key facts. Follow the format, do not use asterisks, all in lowercase."
   
    # Conversation history:
    # {conversation_text}
    # """

    # summary = genai_model.invoke(prompt).content
    # print(f"summary {username}: {summary}")
    # summary_sentences = [s.strip() for s in summary.split(";") if s.strip()]
    # for sentence in summary_sentences:
    #     embedding = get_embeddings([sentence])[0]
    #     user_collection.insert_one({
    #         "summary_sentence": sentence,
    #         "embedding": embedding.tolist(),
    #         "timestamp": datetime.now()
    #     })

    # Xóa session của user (bao gồm FAISS index)
    del sessions[username]
    if not sessions:
        global doc_index, doc_embeddings, doc_texts_current
        doc_index = None
        doc_embeddings = None
        doc_texts_current = None
        session_event.clear()

# Hàm lưu tất cả session khi server dừng
def save_all_sessions():
    for username in list(sessions.keys()):
        summarize_and_store(username)

atexit.register(save_all_sessions)

# API endpoint GET /summarize
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
def retrieve_docs(username, query, top_k=1):
    query_embedding = get_embeddings([query])
    
    ope_docs = []
    user_docs = []

    if doc_index is not None:
        distances, indices = doc_index.search(query_embedding, top_k)
        ope_docs = [doc_texts_current[idx] for idx in indices[0]]
    else:
        ope_docs.append("No document index available for Ope Watson!")

    user_faiss = sessions[username].get("faiss")
    if user_faiss and user_faiss["index"] is not None:
        distances, indices = user_faiss["index"].search(query_embedding, top_k)
        user_docs = [user_faiss["texts"][idx] for idx in indices[0]]
    else:
        user_docs.append("No user index available!")

    return ope_docs, user_docs

# Hàm sinh câu trả lời
def generate_response(username, query):
    if username not in sessions:
        return "Session expired, please enter your name again!"
    convo = sessions[username]["convo"]
    ope_docs, user_docs = retrieve_docs(username, query, top_k=1)
    ope_context = "\n".join(ope_docs)
    user_context = "\n".join(user_docs)
    print(f"Ope context: {ope_context}")
    print(f"User context: {user_context}")
    convo.append({"role": "user", "parts": [{"text": query}]})
    
    prompt = f"Below is the conversation history between {username} and Ope Watson (yourself):\n"
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
        f"Additional information about Ope Watson (format: 'Ope Watson, attributes = info'): {ope_context}\n"
        f"Additional information about {username} (format: '{username}, attributes = info'): {user_context}\n"
        f"Now, respond to {username}'s question: {query}"
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
            sessions[username] = {
                "convo": [{"role": "system", "parts": [{"text": "You're playing as Ope Watson, a gentle boy, a chaotic friend, narcissistic, playful and humourous. Keep responses humanlike, short and on point! Do not use asterisks! Do not list up, only talk about one thing at a time. Do not answer summarization requests. Capitalize to emphasize! Answer in the language that users are using!"}]}],
                "last_active": datetime.now(),
                "username": username
            }
            initialize_index(username)
        else:
            sessions[username] = {
                "convo": [{"role": "system", "parts": [{"text": "You're playing as Ope Watson, a gentle boy, a chaotic friend, narcissistic, playful and humourous. Keep responses humanlike, short and on point! Do not use asterisks! Do not list up, only talk about one thing at a time. Do not answer summarization requests. Capitalize to emphasize! Answer in the language that users are using!"}]}],
                "last_active": datetime.now(),
                "username": username,
                "faiss": None
            }
            initialize_index(username)
        session_event.set()

    try:
        response = generate_response(username, query)
        ope_docs, user_docs = retrieve_docs(username, query, top_k=1)
        return jsonify({
            "query": query,
            "response": response,
            "retrieved_docs": {
                "ope_watson": ope_docs,
                "user": user_docs
            },
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
