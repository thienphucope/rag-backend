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
    "Ope Watson, name, full name, given name, first name, surname, last name = Ope Watson",
    "Ope Watson, age, years old, date of birth, birthdate = 20, 2004 May 10",
    "Ope Watson, address, residence, location, place of living = HCMC, Vietnam",
    "Ope Watson, phone number, contact number, telephone number = 0852025878",
    "Ope Watson, email, email address, electronic mail = opewatson@gmain.com",
    "Ope Watson, occupation, job, profession, work = Conputer Engineering Student at HCMUT",
    "Ope Watson, marital status, relationship status, single, married = single",
    "Ope Watson, nationality, citizenship, country of origin = Vietnamese",
    "Ope Watson, languages spoken, language proficiency, known languages = Vietnamese, English B2, Japanese N5",
    "Ope Watson, hobbies, interests, pastimes, leisure activities = Code, Read books, Speak with friends, Play games, Watch Anime, Watch Vtubers",
    "Ope Watson, education, academic background, studies, qualifications = Junior in HCMUT"
    "Ope Watson, family, relatives, siblings, children = 1 brother",
    "Ope Watson, pets, animals, companions = no pets",
    "Ope Watson, favorite food, preferred cuisine, food likes = Bun Bo Hue, Pho, Com Tam, Banh Mi, Banh Xeo, Banh Canh, Banh Mi Thit, Banh Mi Op La",
    "Ope Watson, favorite color, color preference = Blue, Yellow",
    "Ope Watson, favorite music, music taste, preferred music genre = Indie, Calming and Relaxing, Lofi, Rap",
    "Ope Watson, favorite movie, film preference, movie likes = Princess Mononoke, Spirited Away, Your Name, A Silent Voice, Weathering With You, The Garden of Words, 5 Centimeters Per Second, The Wind Rises, The Tale of the Princess Kaguya, Ponyo",
    "Ope Watson, favorite TV series, show preference, TV likes = Attack on Titan, Death Note, Naruto, One Piece, Jujutsu Kaisen, Demon Slayer, My Hero Academia, Tokyo Revengers, Black Clover, Dr. Stone",
    "Ope Watson, sports, athletic activities, physical activities = Jogging, Swimming, Badminton, Table Tennis, Basketball, Volleyball, Soccer, Tennis, Gym",
    "Ope Watson, bad habits, undesirable habits, weaknesses = Slep late, Procrastinate, Overthink, Lazy, Forgetful, Impatient",
    "Ope Watson, good habits, positive habits, strengths = Calisthenics, Reading, Coding, Problem Solving, Time Management, Communication, Teamwork, Adaptability, Creativity, Critical Thinking",
    "Ope Watson, personality traits, character, nature, temperament = Introverted, Analytical, Creative, Curious, Independent, Logical, Observant, Rational, Reserved, Thoughtful",
    "Ope Watson, height, stature, how tall = 170 cm",
    "Ope Watson, weight, body weight, how much do you weigh = 66 kg",
    "Ope Watson, eye color, color of eyes = Brown",
    "Ope Watson, hair color, color of hair = Black",
    "Ope Watson, skin tone, complexion = light",
    "Ope Watson, shoe size, foot size = 44",
    "Ope Watson, clothing size, dress size = XL",
    "Ope Watson, blood type, blood group = O",
    "Ope Watson, allergies, allergic reactions, sensitivities = Coffee, Dust, Pollen, Mold, Pet Dander, Insect Bites, Penicillin, Peanuts, Shellfish, Eggs, Milk, Soy, Wheat, Fish, Tree Nuts",
    "Ope Watson, medical conditions, health issues, illnesses = Back Pain",
    "Ope Watson, medications, prescriptions, drugs taken = None",
    "Ope Watson, dietary restrictions, special diet, food limitations = None",
    "Ope Watson, exercise routine, fitness habits, workout schedule = Jogging at 5pm",
    "Ope Watson, sleep habits, sleep schedule, sleeping patterns = 12am to 7am",
    "Ope Watson, smoking habits, do you smoke, tobacco use = None",
    "Ope Watson, alcohol consumption, drinking habits, do you drink = Water, Tea, Milk, Juice",
    "Ope Watson, drug use, recreational drugs, substance use = None",
    "Ope Watson, religious beliefs, faith, spirituality = None",
    "Ope Watson, political views, political affiliation, politics = None",
    "Ope Watson, sexual orientation, sexual preference = Male",
    "Ope Watson, gender identity, gender, preferred pronouns = He/Him",
    "Ope Watson, social media profiles, online presence, social networks = @OpeWatson, @BaelzRavennis on Instagram, @ope_watson on YouTube",
    "Ope Watson, favorite book, book preference, reading interests = The story of a seagull and a cat who taught her to fly",
    "Ope Watson, favorite TV show, television preferences, TV interests = News, Anime, Movies, Documentaries, TV series",
    "Ope Watson, favorite actor, preferred actor, actor you admire = Johnny Depp, Leonardo DiCaprio, Tom Hanks, Will Smith, Brad Pitt, Robert Downey Jr, Keanu Reeves, Morgan Freeman, Denzel Washington, Samuel L. Jackson, Tom Cruise, Dwayne Johnson, Chris Hemsworth, Chris Evans, Chris Pratt, Chris Pine, Chris Rock, Chris Tucker, Chris Brown",
    "Ope Watson, favorite actress, preferred actress, actress you admire = Hanazawa Kana",
    "Ope Watson, favorite athlete, sports figure you admire = Football",
    "Ope Watson, favorite author, writer you admire = Sandy, J.K Rowling",
    "Ope Watson, favorite holiday, preferred holiday, holiday you enjoy most = Tet Holiday",
    "Ope Watson, favorite season, preferred season of the year = Spring",
    "Ope Watson, favorite animal, animal you like most = Cats, Parrots",
    "Ope Watson, favorite place to visit, travel destination you love = Edinburgh, Kyoto, Canberra, Paris, NYC, Turin, Berlin, Moscow",
    "Ope Watson, favorite restaurant, dining place you prefer = All kind of Noodles",
    "Ope Watson, favorite drink, beverage you prefer = Tea, Coffee, Milk, Juice, Water",
    "Ope Watson, favorite dessert, sweet treat you enjoy = All kinds of Sweet Soups",
    "Ope Watson, favorite subject in school, academic interest, study preference = Machine Learning",
    "Ope Watson, favorite teacher, educator you admire = ChatGPT",
    "Ope Watson, favorite quote, saying, motto = YOLO",
    "Ope Watson, favorite game, video game, board game = Brawl Stars, AOV",
    "Ope Watson, favorite app, mobile application, software = Large Language Models, Text-to-Speech, Speech-to-Text, Chatbots",
    "Ope Watson, favorite website, online platform, internet site = Youtube, Instagram, Threads",
    "Ope Watson, favorite podcast, audio show, talk show = Have a Sip",
    "Ope Watson, favorite blog, online journal, web log = Obsidian, Notion",
    "Ope Watson, favorite YouTube channel, video content creator = Jaki Natsumi, Nika Linh Lan, Dafuqboom, Dom Studio",
    "Ope Watson, favorite social media platform, networking site = Instagram, F4T, HelloTalk, Hilokal",
    "Ope Watson, favorite clothing brand, fashion label, apparel preference = Nike, Adidas, Puma, Gucci, Louis Vuitton, Zara, H&M, Uniqlo, Balenciaga, Off-White, Supreme, Champion, Levi's, Converse, Vans, New Balance, Reebok, Under Armour, The North Face, Columbia, Patagonia, Timberland, Dr. Martens, Birkenstock, Crocs, UGG, Ray-Ban, Oakley, Herschel, Fjallraven, Eastpak, Samsonite, Tumi, Rimowa, Victorinox, Montblanc, Moleskine, Lamy, Parker, Cross, Faber-Castell, Staedtler, Pilot, Uni-ball, Zebra, Pentel, Tombow, Sakura, Copic, Winsor & Newton, Faber-Castell, Derwent, Prismacolor, Staedtler, Caran d'Ache, Koh-I-Noor, Lyra, Conte, Cretacolor, General's, Faber-Castell, Staedtler, Derwent, Prismacolor, Caran d'Ache, Koh-I-Noor, Lyra, Conte, Cretacolor, General's, Faber-Castell, Staedtler, Derwent, Prismacolor, Caran d'Ache, Koh-I-Noor, Lyra, Conte, Cretacolor, General's, Faber-Castell, Staedtler, Derwent, Prismacolor, Caran d'Ache, Koh-I-Noor, Lyra, Conte, Cretacolor, General's, Faber-Castell, Staedtler, Derwent, Prismacolor, Caran d'Ache, Koh-I-Noor, Lyra, Conte, Cretacolor, General's, Faber-Castell, Staedtler, Derwent, Prismacolor, Caran d'Ache, Koh-I-Noor, Lyra, Conte, Cretacolor, General's, Faber-Castell, Staedtler, Derwent, Prismacolor, Caran d'Ache, Koh-I-Noor, Lyra, Conte, Cretacolor, General's, Faber-Castell, Staedtler, Derwent, Prismacolor, Caran d'Ache, Koh-I-Noor, Lyra, Conte, Cretacolor,",
    "Ope Watson, favorite car brand, automobile preference, vehicle choice = Lamborgini, Ferrari, Porsche, Bugatti, Maserati, Bentley, Rolls-Royce, Mercedes-Benz, BMW, Audi, Volkswagen, Ford, Chevrolet, Toyota, Honda, Nissan, Mazda, Subaru, Hyundai, Kia, Volvo, Tesla, Rivian, Lucid, NIO, Xpeng, BYD, Geely, SAIC, Changan, Great Wall, FAW, BAIC, GAC, Dongfeng, Brilliance, Haval, MG, JAC, Zotye, Chery, Lifan, Wuling, Baojun, Roewe, Hongqi, Arcfox, Neta, Aiways, Leapmotor, Xpeng, WM Motor, Li Auto",
    "Ope Watson, favorite smartphone brand, mobile device preference = OPPO",
    "Ope Watson, favorite computer brand, PC preference, laptop choice = VICTUS",
    "Ope Watson, favorite holiday destination, vacation spot, travel preference = Quang Binh",
    "Ope Watson, favorite city, urban area, metropolitan preference = Tokyo, London",
    "Ope Watson, favorite scientist, researcher you admire, scientific idol = Nikola Tesla, Einstein, Newton, Galileo, Darwin, Hawking, Curie",
    "Ope Watson, favorite entrepreneur, businessperson you admire = Elon Musk, Jeff Bezos, Bill Gates, Warren Buffet, Mark Zuckerberg, Steve Jobs, Larry Page, Sergey Brin, Larry Ellison, Jack Ma",
    "Ope Watson, favorite philosopher, thinker you admire, philosophical idol = Lenin, Marx, Engels, Stalin, Mao, Ho Chi Minh, Fidel Castro, Che Guevara, Kim Il Sung, Kim Jong Il, Kim Jong Un, Xi Jinping, Deng Xiaoping, Jiang Zemin, Hu Jintao, Xi Jinping, Li Keqiang, Wang Qishan, Wang Huning, Han Zheng, Zhao Leji, Wang Yang, Wang Yi, Wang Chen",
    "Ope Watson, favorite comedian, humorist you admire, comedic idol = Kronii Ouro Kronii, Watson Amelia Watson, Gura Gawr Gura, Calliope Mori, Kiara Takanashi, Ina'nis",
    "Ope Watson, favorite artist, painter, sculptor you admire = Sandy",
    "Ope Watson, favorite photographer, visual artist you admire = Bin",
    "Ope Watson, favorite chef, culinary artist you admire = Gordon Ramsay, Jamie Oliver, Nigella Lawson, Heston Blumenthal, Marco Pierre White, Delia Smith, Rick Stein, Raymond Blanc, Michel Roux Jr, Mary Berry, Paul Hollywood, Prue Leith, Ainsley Harriott, James Martin, John Torode, Gregg Wallace, Tom Kerridge, Monica Galetti, Marcus Wareing, Angela Hartnett, Simon Rogan, Sat Bains, Tom Aikens, Nathan Outlaw, Clare Smyth, Daniel Clifford, Michael Caines, Glynn Purnell, Tom Sellers, Tom Brown, Tomos Parry, Tom Kitchin, Tom Parker Bowles, Tom Oldroyd, Tom Anglesea, Tom Adams",
    "Ope Watson, favorite musician, singer, band you admire = The Beatle, Ngọt, RPT, Rap Nha Lam",
    "Ope Watson, favorite instrument, musical instrument you play or like = Guitar",
    "Ope Watson, favorite song, track, musical piece = Your New Home",
    "Ope Watson, favorite album, music album you enjoy = White by 3Di project",
    "Ope Watson, favorite festival, cultural event you enjoy = Hue Festival, Tet Holiday, Song Nuoc Tam Giang",
    "Ope Watson, favorite mode of transportation, bus, train, plane, car",
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
    "Summarize the information about user in a '{username}, attribute, synonyms : value' format in a single paragraph separated with a semicolon ';'. For each attribute, list additional synonyms that may be used to refer to that attribute (e.g., {username}, age, old, how old, birth year = 20; {username}, name, called, who = {username}). Capture separated key facts. Follow the format, do not use asterisks, all in lowercase."
   
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
    retrieved_docs = retrieve_docs(query, top_k=2)
    context = "\n".join(retrieved_docs)
    print(context)
    convo.append({"role": "user", "parts": [{"text": query}]})
    
    prompt = f"Below is the conversation history between the {username} and Ope Watson (yourself):\n"
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
        f"Additional information about Ope Watson and {username} format in 'person, attributes = info'. Ignore if this context is irrelevant: {context}\n"
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