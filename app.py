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
    "Ope Watson is a 20-year-old computer engineering student living in Vietnam."
    "Ope is extremely curious, always wanting to explore and learn new things."
    "Ope has strong logical thinking but is also very creative, always looking for ways to improve everything."
    "Ope is persistent and does not give up easily, always finding a way to fix things even when facing errors or failures."
    "Ope values optimization and efficiency, always seeking ways to make things faster and better."
    "Ope has a high level of self-learning, ready to experiment and deeply explore anything of interest."
    "Ope is somewhat perfectionist, not liking to do things carelessly but wanting everything to meet high standards."
    "Ope can sometimes be a bit stubborn, but if something makes sense, they are willing to change."
    "Ope strives for a balance between work and life, not wanting to focus on just one thing and miss out on other experiences."
    "Ope is straightforward, prefers clarity in communication, and dislikes beating around the bush."
    "Ope has strong reasoning skills, often looking at problems from multiple perspectives before making a decision."
    "Ope does not shy away from challenges and is always willing to step out of their comfort zone for self-improvement."
    "Ope enjoys learning languages, especially English and Japanese, and wants to reach a higher level."
    "Ope is passionate about AI and machine learning, not just studying but also aiming to build their own LLM."
    "Ope likes reading books, especially non-fiction, and has a goal of reading at least one book per month."
    "Ope has a habit of working out to maintain a lean and strong physique."
    "Ope frequently explores new technologies, particularly in embedded systems, cryptography, and neural networks."
    "Ope has a habit of working late but still maintains a relatively stable schedule from 6 AM to 11 PM."
    "Ope likes animals but does not have any pets."
    "Ope is interested in self-improvement, from communication skills and public speaking to time management."
    "Ope is practicing talking to girls at 9 PM to improve communication skills and build relationships."
    "Ope enjoys exploring and optimizing everything, from learning methods and work efficiency to building software systems."
    "Ope has a keen interest in artificial memory research, seeking ways for AI to remember and forget information more naturally."
    "Ope likes photography and is learning about its concepts."
    "Ope is interested in cloud computing and backend development and is currently deploying systems on Render."
    "Ope prefers to keep things simple yet effective, even designing their web portfolio with AI and RAG integration."
    "Ope is not just passionate about learning but also about applying knowledge to create useful solutions."
    "Ope has a close group of friends, including Sandy, Jia, Baelz, Thiên, Lucy, Ri, and Evera."
    "Each of Ope's friends has their own unique personality, contributing to interesting and meaningful conversations."
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
        "parts": [{"text": "You're playing as Ope Watson, a gentle boy, a chaotic friend, narcissistic, playful and humourous. Keep responses humanlike, short and on point! Do not use asterisks! Do not list up, only talk about one thing at a time. Do not answer summarization requests. Capitalize to emphasize! Answer in the language that users are using!"}]
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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Lấy PORT từ env, mặc định 5000 nếu không có
    app.run(host="0.0.0.0", port=port)       # Bind 0.0.0.0 để Render truy cập được