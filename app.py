from flask import Flask, request, jsonify
from flask_cors import CORS
import faiss
import numpy as np
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from huggingface_hub import InferenceClient

genai_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# Khởi tạo Flask app và bật CORS
app = Flask(__name__)
CORS(app)

# Khởi tạo InferenceClient
client = InferenceClient(api_key=os.environ["HF_API_TOKEN"])

# Dữ liệu mẫu
documents = [
    "Ope Watson is a 20-year-old computer engineering student living in Vietnam.",
    "Ope enjoys learning English and Japanese, aiming to improve from B2 to C1 in English.",
    "Ope is interested in AI and machine learning and is researching how to build a personal LLM.",
    "Ope is developing a personal web portfolio with AI integration using RAG and plans to deploy a backend with Pinecone for Gemini.",
    "Ope loves reading books and aims to read at least one book per month.",
    "Ope works out to maintain a lean and fit body.",
    "Ope is practicing talking to girls at 9 PM to develop relationship skills.",
    "Ope wants to find a girlfriend but still prioritizes studies.",
    "Ope loves exploring new knowledge and constantly innovates to improve life.",
]

# Hàm lấy embeddings từ Hugging Face API
def get_embeddings(texts):
    try:
        result = client.feature_extraction(
            texts,  # Truyền trực tiếp danh sách chuỗi văn bản
            model="sentence-transformers/all-MiniLM-L6-v2"
        )
        return np.array(result)
    except Exception as e:
        raise Exception(f"Hugging Face API error: {str(e)}")

# Khởi tạo embeddings cho documents
try:
    embeddings = get_embeddings(documents)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
except Exception as e:
    print(f"Error initializing embeddings: {e}")
    raise

# Lưu trữ convo toàn cục
convo = [
    {
        "role": "user",
        "parts": [{"text": "You're playing as Ope Watson, a gentle boy, a chaotic friend, narcissistic, playful and humourous. Keep responses humanlike, short and on point! Do not use asterisks! Do not list up, only talk about one thing at a time. Do not answer summarization requests. Capitalize to emphasize! When you can't find info below, say Question Recorded!"}]
    }
]

# Hàm truy xuất tài liệu
def retrieve_docs(query, top_k=1):
    query_embedding = get_embeddings([query])
    distances, indices = index.search(query_embedding, top_k)
    return [documents[idx] for idx in indices[0]]

# Hàm sinh câu trả lời với Gemini
def generate_response(query):
    retrieved_docs = retrieve_docs(query, top_k=1)
    context = "\n".join(retrieved_docs)
    convo.append({"role": "user", "parts": [{"text": query}]})
    prompt = "Dựa trên lịch sử trò chuyện sau:\n"
    for message in convo:
        role = message["role"]
        text = message["parts"][0]["text"]
        prompt += f"{role}: {text}\n"
    prompt += f"Và thông tin bổ sung:\n{context}\nTrả lời câu hỏi: {query}"
    
    response = genai_model.invoke(prompt)
    convo.append({"role": "assistant", "parts": [{"text": response.content}]})
    return response.content

# API endpoint
@app.route('/rag', methods=['POST'])
def rag_endpoint():
    data = request.get_json()
    query = data.get('query', '')
    if not query:
        return jsonify({"error": "Query is required"}), 400
    try:
        response = generate_response(query)
        return jsonify({
            "query": query,
            "response": response,
            "retrieved_docs": retrieve_docs(query, top_k=1),
            "convo": convo[1:]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)