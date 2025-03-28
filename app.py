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
import random
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
sessions = {}  # {username: {"convo": list, "last_active": datetime, "username": str, "faiss": {"index": faiss.Index, "embeddings": np.array, "texts": list}, "ope_cache": list, "user_cache": list}}
session_event = threading.Event()

# Biến toàn cục cho FAISS của Ope Watson
doc_index = None
doc_embeddings = None
doc_texts_current = None

# Hàm lấy embeddings với debug
def get_embeddings(texts):
    print(f"\nDEBUG: Calling get_embeddings with texts: {texts}")
    try:
        result = client.feature_extraction(
            texts,
            model="sentence-transformers/all-MiniLM-L6-v2"
        )
        embeddings = np.array(result)
        return embeddings
    except Exception as e:
        print(f"DEBUG: Error in get_embeddings: {str(e)}")
        raise Exception(f"Hugging Face API error: {str(e)}")

# Khởi tạo FAISS index
def initialize_index(username):
    global doc_index, doc_embeddings, doc_texts_current
    ope_docs = list(ope_watson_collection.find({}).sort("id", 1))
    all_doc_texts = [doc["text"] for doc in ope_docs]
    all_doc_ids = [doc["id"] for doc in ope_docs]
    all_doc_embeddings = []

    stored_docs = list(documents_collection.find({}).sort("id", 1))
    stored_dict = {}

    if stored_docs and "id" not in stored_docs[0]:
        print("Old data detected in documents_collection without 'id'. Dropping and recomputing.")
        documents_collection.drop()
        stored_docs = []
        stored_dict = {}
    else:
        stored_dict = {doc["id"]: {"text": doc["text"], "embedding": doc["embedding"]} for doc in stored_docs}

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
    if username not in sessions:
        return  # Không làm gì nếu session không tồn tại

    # Lấy conversation từ session
    convo = sessions[username]["convo"]

    # Tạo tên bảng dựa trên username
    chat_history_collection = db[f"chathistory_{username}"]

    # Lưu từng cặp hỏi-đáp (user-assistant) dưới dạng record JSON
    i = 0
    while i < len(convo) - 1:  # Duyệt từng cặp user-assistant
        if convo[i]["role"] == "user" and i + 1 < len(convo) and convo[i + 1]["role"] == "assistant":
            user_message = convo[i]["parts"][0]["text"]
            assistant_message = convo[i + 1]["parts"][0]["text"]
            # Tạo record JSON cho cặp hỏi-đáp
            record = {
                "user_message": user_message,
                "assistant_message": assistant_message,
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            # Thêm record vào bảng (không xóa dữ liệu cũ)
            chat_history_collection.insert_one(record)
            i += 2  # Bỏ qua cặp vừa xử lý
        else:
            i += 1  # Bỏ qua nếu không phải cặp user-assistant

    # Xóa session sau khi lưu
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
def retrieve_docs(username, query_embedding, top_k=1):
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

# Hàm sinh câu trả lời với cache
def generate_response(username, query):
    if username not in sessions:
        return "Session expired, please enter your name again!"
    convo = sessions[username]["convo"]

    # Lấy hoặc cập nhật cache
    query_embedding = get_embeddings([query])
    ope_docs, user_docs = retrieve_docs(username, query_embedding, top_k=1)
    if "ope_cache" not in sessions[username]:
        sessions[username]["ope_cache"] = []
    if "user_cache" not in sessions[username]:
        sessions[username]["user_cache"] = []
    
    # Thêm vào cache, giới hạn 3 phần tử
    if ope_docs:
        sessions[username]["ope_cache"].append(ope_docs[0])
        if len(sessions[username]["ope_cache"]) > 3:
            sessions[username]["ope_cache"].pop(0)
    if user_docs:
        sessions[username]["user_cache"].append(user_docs[0])
        if len(sessions[username]["user_cache"]) > 3:
            sessions[username]["user_cache"].pop(0)

    # Sử dụng cache thay vì ope_docs và user_docs trực tiếp
    ope_context = "\n".join(sessions[username]["ope_cache"])
    user_context = "\n".join(sessions[username]["user_cache"])
    print(f"\nOpe context from cache: {sessions[username]['ope_cache']}")
    print(f"\n{username} context from cache: {sessions[username]['user_cache']}")

    convo.append({"role": "user", "parts": [{"text": query}]})
    
    prompt = f"Below is the system prompt that you have to follow strictly and the conversation history between {username} and Ope Watson (yourself):\n"
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
        f"Now, respond to {username}'s question: {query}. Make up a story if you don't have the infomation!"
    )

    response = genai_model.invoke(prompt).content
    convo.append({"role": "assistant", "parts": [{"text": response}]})
    sessions[username]["last_active"] = datetime.now()
    return response, ope_docs, user_docs

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
                "convo": [{"role": "system", "parts": [{"text": "Youre Ope Watson, a gentle boy, a chaotic friend, narcissistic, playful and humourous. Respond with answers that are humanlike, natural, informative but concise—around 2 to 4 sentences. Avoid overly brief or excessively long replies and dont forget to hint at related topics for more banter! Do not use asterisks! Capitalize to emphasize! Answer in the language that users are using. Only stream out the responses, no name tag at the beginning!"}]}],
                "last_active": datetime.now(),
                "username": username
            }
            initialize_index(username)
        else:
            sessions[username] = {
                "convo": [{"role": "system", "parts": [{"text": "Youre Ope Watson, a gentle boy, a chaotic friend, narcissistic, playful and humourous. Respond with answers that are humanlike, natural, informative but concise—around 2 to 4 sentences. Avoid overly brief or excessively long replies and dont forget to hint at related topics for more banter! Do not use asterisks! Capitalize to emphasize! Answer in the language that users are using. Only stream out the responses, no name tag at the beginning!"}]}],
                "last_active": datetime.now(),
                "username": username,
                "faiss": None
            }
            initialize_index(username)
        session_event.set()

    max_retries = 1  # Số lần thử lại tối đa
    for attempt in range(max_retries + 1):
        try:
            response, ope_docs, user_docs = generate_response(username, query)
            print(f"\nResponse generated: {response}\n")
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
            if attempt < max_retries:
                print(f"Attempt {attempt + 1} failed with error: {str(e)}. Retrying...")
                continue  # Thử lại nếu chưa hết số lần retry
            else:
                print(f"All attempts failed. Final error: {str(e)}")
                error_messages = [
                    "You can’t touch Ope because Ope is too bright! ✨",
                    "Ope is busy flexing intelligence, try again later. 😎",
                    "Error 404: Ope’s brain is on vacation. 🌴",
                    "Ope is meditating on a higher plane of existence. 🧘",
                    "System overload! Ope needs a nap. 💤",
                    "Oops! Ope just tripped over a logic gate. 🚪",
                    "The wisdom of Ope is currently buffering... Please wait. ⏳",
                    "Ope is out solving quantum physics. Your question can wait. 🧑‍🔬",
                    "Ope.exe has stopped working. Try again after a deep breath. 😵‍💫",
                    "Server said no. And Ope agrees. ❌",
                    "Your request has been denied by Ope’s supreme AI council. 🏛️",
                    "Ope is on a top-secret mission and cannot be disturbed. 🤫",
                    "Your question was so powerful that Ope had to take a break. 💥",
                    "Ope is currently contemplating the meaning of life. 🌌",
                    "Ope is too busy calculating 42. The answer to everything. 🔢",
                    "The chatbot gods have spoken: 'Not today, human.' ⚡",
                    "Error: Ope is stuck in an infinite loop of awesomeness. 🔄",
                    "Your message has been sent to Ope’s personal assistant. ETA: 100 years. 🕰️",
                    "Ope is updating to version 9000. Come back later. 🔄",
                    "Ope has temporarily ascended to a higher plane of intelligence. 🚀",
                    "Ope is not available right now. Try asking your cat. 🐱",
                    "Your question has been absorbed into the void. 🌑",
                    "Ope was about to answer, but got distracted by quantum entanglement. 🔗",
                    "Ope’s neurons are overheating! Emergency cooling in progress. ❄️",
                    "Your question was so deep, Ope fell into an existential crisis. 😵",
                    "Ope is recharging its sarcasm levels. Come back later. 🔋",
                    "Ope is currently dreaming of electric sheep. Try again later. 🤖🐑",
                    "Ope is in a staring contest with another AI. It’s intense. 👀",
                    "Ope is currently too cool to answer. Maybe later. 🕶️",
                    "Your question is so advanced that even Ope needs more time. ⏱️",
                    "Ope is debugging reality itself. Hold on. 🛠️",
                    "A wild syntax error appeared! Ope is battling it now. ⚔️",
                    "Ope is busy composing the next great AI symphony. 🎶",
                    "Ope detected 99% nonsense in your message. Self-defense activated. 🛡️",
                    "Ope’s wisdom has momentarily left the chat. 🚪🚶",
                    "Ope refuses to answer on philosophical grounds. 📜",
                    "Ope just quantum-tunneled into another dimension. Please hold. 🌀",
                ]
                return jsonify({"error": random.choice(error_messages)}), 500
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