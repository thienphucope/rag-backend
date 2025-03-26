from flask import Flask, request, jsonify, session
from flask_cors import CORS
import faiss
import numpy as np
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from huggingface_hub import InferenceClient
from pymongo import MongoClient
from datetime import datetime, timedelta
import threading
import atexit

# Lấy biến môi trường
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
HF_API_TOKEN = os.environ["HF_API_TOKEN"]
MONGO_URI = os.environ["MONGO_URI"]
PORT = os.environ.get("PORT", "5000")  # Mặc định là 5000 nếu không có PORT

# Khởi tạo Flask app và bật CORS
app = Flask(__name__)
app.secret_key = os.urandom(24)
CORS(app)

# Kết nối MongoDB
mongo_client = MongoClient(MONGO_URI)
db = mongo_client["chat_db"]
conversations_collection = db["conversations"]
documents_collection = db["documents"]

# Khởi tạo model và client
genai_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GEMINI_API_KEY)
client = InferenceClient(api_key=HF_API_TOKEN)

# Dữ liệu mẫu
doc_texts = [
    "Ope Watson is a 20-year-old computer engineering student living in Vietnam.",
    "Ope is extremely curious, always wanting to explore and learn new things.",
    "Ope has strong logical thinking but is also very creative, always looking for ways to improve everything.",
    "Ope is persistent and does not give up easily, always finding a way to fix things even when facing errors or failures.",
    "Ope values optimization and efficiency, always seeking ways to make things faster and better.",
    "Ope has a high level of self-learning, ready to experiment and deeply explore anything of interest.",
    "Ope is somewhat perfectionist, not liking to do things carelessly but wanting everything to meet high standards.",
    "Ope can sometimes be a bit stubborn, but if something makes sense, they are willing to change.",
    "Ope strives for a balance between work and life, not wanting to focus on just one thing and miss out on other experiences.",
    "Ope is straightforward, prefers clarity in communication, and dislikes beating around the bush.",
    "Ope has strong reasoning skills, often looking at problems from multiple perspectives before making a decision.",
    "Ope does not shy away from challenges and is always willing to step out of their comfort zone for self-improvement.",
    "Ope enjoys learning languages, especially English and Japanese, and wants to reach a higher level.",
    "Ope is passionate about AI and machine learning, not just studying but also aiming to build their own LLM.",
    "Ope likes reading books, especially non-fiction, and has a goal of reading at least one book per month.",
    "Ope has a habit of working out to maintain a lean and strong physique.",
    "Ope frequently explores new technologies, particularly in embedded systems, cryptography, and neural networks.",
    "Ope has a habit of working late but still maintains a relatively stable schedule from 6 AM to 11 PM.",
    "Ope likes animals but does not have any pets.",
    "Ope is interested in self-improvement, from communication skills and public speaking to time management.",
    "Ope is practicing talking to girls at 9 PM to improve communication skills and build relationships.",
    "Ope enjoys exploring and optimizing everything, from learning methods and work efficiency to building software systems.",
    "Ope has a keen interest in artificial memory research, seeking ways for AI to remember and forget information more naturally.",
    "Ope likes photography and is learning about its concepts.",
    "Ope is interested in cloud computing and backend development and is currently deploying systems on Render.",
    "Ope prefers to keep things simple yet effective, even designing their web portfolio with AI and RAG integration.",
    "Ope is not just passionate about learning but also about applying knowledge to create useful solutions.",
    "Ope has a close group of friends, including Sandy, Jia, Baelz, Thiên, Lucy, Ri, and Evera.",
    "Each of Ope's friends has their own unique personality, contributing to interesting and meaningful conversations.",
]

# Lưu trữ phiên trò chuyện tạm thời
sessions = {}
session_event = threading.Event()

# Biến toàn cục cho FAISS
doc_index = None
doc_embeddings = None
doc_texts_current = None  # Để theo dõi dữ liệu hiện tại trong FAISS

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

# Khởi tạo FAISS index - Gọi khi có session mới
def initialize_index():
    global doc_index, doc_embeddings, doc_texts_current
    all_texts = doc_texts.copy()
    all_embeddings = []

    # Tải embeddings từ documents_collection (cho dữ liệu tĩnh)
    if documents_collection.count_documents({}) > 0:
        stored_docs = list(documents_collection.find({}))
        if stored_docs:
            all_embeddings = [np.array(doc["embedding"]) for doc in stored_docs]
            all_texts = [doc["text"] for doc in stored_docs]
    
    # Nếu chưa có embeddings cho doc_texts hoặc không khớp, tạo mới
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

    # Tải embeddings từ conversations_collection (nếu đã có)
    convo_docs = list(conversations_collection.find({"embedding": {"$exists": True}}))
    for convo in convo_docs:
        summary_sentence = convo["summary_sentence"]
        all_embeddings.append(np.array(convo["embedding"]))
        all_texts.append(summary_sentence)

    # Cập nhật biến toàn cục
    doc_texts_current = all_texts
    doc_embeddings = np.array(all_embeddings)

    # Khởi tạo FAISS index
    dimension = doc_embeddings.shape[1]
    doc_index = faiss.IndexFlatL2(dimension)
    doc_index.add(doc_embeddings)

# Hàm tóm tắt và lưu DB - Xả FAISS khi không còn session
def summarize_and_store(session_id):
    global doc_index, doc_embeddings, doc_texts_current
    if session_id not in sessions:
        return
    convo = sessions[session_id]["convo"]
    username = sessions[session_id]["username"]
    conversation_text = "\n".join([f"{m['role']}: {m['parts'][0]['text']}" for m in convo])
    prompt = f"""
    "Summarize the conversation between Ope Watson and {username} in a short single paragraph, capturing only key facts such as appointments, promises, user details, or major decisions. Format each fact as 'Ope Watson notes that {username}...' and exclude unnecessary details. Respond without any additional commentary."

    Conversation history:
    {conversation_text}
    """
    summary = genai_model.invoke(prompt).content
    summary_sentences = [s.strip() for s in summary.split(".") if s.strip()]
    for sentence in summary_sentences:
        embedding = get_embeddings([sentence])[0]
        conversations_collection.insert_one({
            "session_id": session_id,
            "username": username,
            "summary_sentence": sentence,
            "embedding": embedding.tolist(),
            "timestamp": datetime.now()
        })

    del sessions[session_id]
    if not sessions:
        # Xả dữ liệu FAISS khi không còn session
        doc_index = None
        doc_embeddings = None
        doc_texts_current = None
        session_event.clear()

# Hàm lưu tất cả session khi server dừng
def save_all_sessions():
    for sid in list(sessions.keys()):
        summarize_and_store(sid)

atexit.register(save_all_sessions)

# Check timeout - Chỉ chạy khi có session
def check_timeout():
    while True:
        if sessions:
            now = datetime.now()
            for sid in list(sessions.keys()):
                if now - sessions[sid]["last_active"] > timedelta(minutes=5):
                    summarize_and_store(sid)
            threading.Event().wait(60)
        else:
            session_event.wait()

threading.Thread(target=check_timeout, daemon=True).start()

# Hàm truy xuất tài liệu - Tìm kiếm trong FAISS
def retrieve_docs(query, top_k=1):
    if doc_index is None:
        return ["No FAISS index available, waiting for a new session!"]
    query_embedding = get_embeddings([query])
    distances, indices = doc_index.search(query_embedding, top_k)
    return [doc_texts_current[idx] for idx in indices[0]]

# Hàm sinh câu trả lời
def generate_response(session_id, query):
    if session_id not in sessions:
        return "Session expired, please enter your name again!"
    convo = sessions[session_id]["convo"]
    retrieved_docs = retrieve_docs(query, top_k=1)
    context = "\n".join(retrieved_docs)
    
    convo.append({"role": "user", "parts": [{"text": query}]})
    
    prompt = "Below is the conversation history between the User and Ope Watson (you):\n"
    for message in convo:
        role = message["role"]
        text = message["parts"][0]["text"]
        if role == "user":
            prompt += f"User: {text}\n"
        elif role == "assistant":
            prompt += f"Ope Watson: {text}\n"
        elif role == "system":
            prompt += f"system: {text}\n"
    prompt += (
        f"Additional information about Ope Watson consider it as his memory: {context}\n"
        f"Now, respond to the User's question: {query}"
    )

    response = genai_model.invoke(prompt).content
    convo.append({"role": "assistant", "parts": [{"text": response}]})
    sessions[session_id]["last_active"] = datetime.now()
    return response

# API endpoint
@app.route('/rag', methods=['POST'])
def rag_endpoint():
    global doc_index
    data = request.get_json()
    username = data.get('username', '')
    query = data.get('query', '')
    if not username or not query:
        return jsonify({"error": "Username và query là bắt buộc!"}), 400
    
    session_id = f"{username}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    if "session_id" not in session or session["session_id"] not in sessions:
        # Khi có session mới, khởi động FAISS nếu chưa có
        if doc_index is None:
            initialize_index()
        session["session_id"] = session_id
        sessions[session_id] = {
            "convo": [{"role": "system", "parts": [{"text": "You're playing as Ope Watson, a gentle boy, a chaotic friend, narcissistic, playful and humourous. Keep responses humanlike, short and on point! Do not use asterisks! Do not list up, only talk about one thing at a time. Do not answer summarization requests. Capitalize to emphasize! Answer in the language that users are using!"}]}],
            "last_active": datetime.now(),
            "username": username
        }
        session_event.set()

    try:
        response = generate_response(session["session_id"], query)
        return jsonify({
            "query": query,
            "response": response,
            "retrieved_docs": retrieve_docs(query, top_k=1),
            "session_id": session["session_id"]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(PORT))