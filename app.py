import streamlit as st
import pickle
import re
import nltk
import requests
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from streamlit_lottie import st_lottie
from streamlit_extras.let_it_rain import rain
from urllib.parse import urlparse, parse_qs
from googleapiclient.discovery import build

# ---- Page Config ----
st.set_page_config(page_title="Sentiment Analysis App", page_icon="💬", layout="centered")
st.title("💬 Sentiment Analysis on Product Reviews")
st.markdown("<h5 style='text-align:center;color:#00FFFF;'>Elevvo Internship | NLP Domain | Level 1 - Task 1</h5>", unsafe_allow_html=True)

# ---- Lottie Animation ----
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

st_lottie(load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_jcikwtux.json"), height=180)

# ---- Styling ----
st.markdown("""
<style>
.stApp { background-image: linear-gradient(rgba(0,0,0,0.6), rgba(0,0,0,0.6)),
          url("https://i0.wp.com/www.cognillo.com/blog/wp-content/uploads/2019/04/sentiment-analysis.jpg?fit=848%2C477&ssl=1");
          background-size: cover; background-position: center; color: white; }
.result-card { background-color: rgba(255,255,255,0.1); border-radius: 15px; padding: 20px; text-align:center; box-shadow:0px 4px 20px rgba(255,215,0,0.2);}
.positive { border-left:6px solid #00ff88; color:#00ff88; font-size:20px;}
.negative { border-left:6px solid #ff4b4b; color:#ff4b4b; font-size:20px;}
</style>
""", unsafe_allow_html=True)

# ---- NLP Setup ----
nltk.download('stopwords')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# ---- Sidebar Styling ----
st.markdown("""
<style>
[data-testid="stSidebar"] {
    background-color: rgba(0, 70, 30, 0.45);
    color: white;
}
[data-testid="stSidebar"] h1, h2, h3 { color: #00171F; }
::-webkit-scrollbar-thumb { background: #00cfff; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ---- Project Introduction ----
with st.sidebar.expander("ℹ️ Project Introduction"):
    st.markdown("""
    **Project Title:** Sentiment Analysis on Movie/Product Reviews  
    **Internship:** Elevvo Internship (NLP Domain)  
    **Level:** 1  
    **Task:** 1  
    **Website:** [Elevvo](https://elevvo.tech/home)

    This project analyzes reviews to determine whether the sentiment is **Positive 😄** or **Negative 😡**  
    using **Natural Language Processing (NLP)** and **Machine Learning**.
    """)

# ---- Developer's Intro ----
with st.sidebar.expander("👨‍💻 Developer's Intro"):
    st.markdown("- **Hi, I'm Rayyan Ahmed**")
    st.markdown("- **Google Certified AI Prompt Specialist**")
    st.markdown("- **IBM Certified Advanced LLM FineTuner**")
    st.markdown("- **Google Certified Soft Skill Professional**")
    st.markdown("- **Hugging Face Certified: Fundamentals of LLMs**")
    st.markdown("- **Expert in EDA, ML, RL, ANN, CNN, CV, RNN, NLP, LLMs**")
    st.markdown("[💼 Visit LinkedIn](https://www.linkedin.com/in/rayyan-ahmed-504725321/)")

# ---- Tech Stack ----
with st.sidebar.expander("🛠️ Tech Stack Used"):
    st.markdown("""
    - **Python Libraries:** Numpy, Pandas, Matplotlib, Seaborn, Requests, Re, Pickle  
    - **Machine Learning & AI:** Scikit-learn, TensorFlow, Keras, NLTK  
    - **Web App & UI:** Streamlit, Streamlit Extras, Lottie Animations  
    - **Image & Text Processing:** OpenCV, PIL (Pillow), Regex, NLP Techniques  
    - **Version Control / Deployment:** Git, Streamlit Cloud
    """)

# ---- Load Models ----
model_choice = st.selectbox("Select Model", ["Logistic Regression", "Naive Bayes"])
model = pickle.load(open("sentiment_model.pkl", "rb")) if model_choice=="Logistic Regression" else pickle.load(open("naivebayes_model.pkl","rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ---- Theme ----
theme = st.radio("🎨 Choose Theme:", ["Dark Mode","Light Mode"], horizontal=True)
if theme=="Light Mode":
    st.markdown("<style>.stApp {background: #f0f0f0; color: black;}</style>", unsafe_allow_html=True)

# ---- Ensure session_state ----
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# ---- Example Reviews ----
examples = [
    "The movie exceeded all my expectations. Brilliant direction, emotional depth, and outstanding performances make it a masterpiece worth watching again.",
    "A visually stunning film with a weak storyline. The acting was average, and the pacing made it hard to stay engaged throughout.",
    "The film was a total disappointment. Poor editing, predictable plot, and lackluster performances ruined the experience."
]

st.markdown("💡 **Try an Example Review:**")

# ---- Example Buttons Styling ----
st.markdown("""
<style>
div.stButton > button[kind] {
    background-color: #FFD700 !important;  /* Yellow */
    color: black !important;
    font-weight: bold !important;
    border-radius: 10px !important;
    padding: 0.5em 1em !important;
}
div.stButton > button[kind]:hover {
    background-color: #ffea00 !important;
    transform: scale(1.05);
}
</style>
""", unsafe_allow_html=True)

# ---- Example Buttons ----
cols = st.columns(len(examples))
for i, ex in enumerate(examples):
    if cols[i].button(f"Example {i+1}", key=f"example_{i}"):
        st.session_state.user_input = ex
        st.experimental_rerun()  # <- force rerun to update text area

# ---- Text Area Styling ----
st.markdown("""
<style>
div[data-testid="stTextArea"] textarea {
    border: 2px solid #FFD700 !important;
    border-radius: 10px !important;
    padding: 10px !important;
}
div[data-testid="stTextArea"] textarea:focus {
    outline: none !important;
    border: 2px solid #FFA500 !important;
    box-shadow: 0 0 5px #FFD700;
}
</style>
""", unsafe_allow_html=True)

# ---- Text Area ----
user_input = st.text_area(
    "✍️ Type your review here:",
    value=st.session_state.user_input,
    height=150,
    max_chars=300,
    key="user_input_box"
)

# ---- YouTube URL Input Styling ----
st.markdown("""
<style>
div[data-testid="stTextInput"] input {
    border: 2px solid #FFD700 !important;
    border-radius: 10px !important;
    padding: 10px !important;
}
div[data-testid="stTextInput"] input:focus {
    outline: none !important;
    border: 2px solid #FFA500 !important;
    box-shadow: 0 0 5px #FFD700;
}
</style>
""", unsafe_allow_html=True)

youtube_url = st.text_input("Enter YouTube Video URL:", placeholder="https://www.youtube.com/watch?v=VIDEO_ID")
max_comments = st.slider("Max comments to fetch:", 5, 50, 20)

# ---- Helper Functions ----
def extract_video_id(url):
    if "youtu.be" in url:
        return url.split("/")[-1]
    parsed = urlparse(url)
    return parse_qs(parsed.query).get("v", [None])[0]

def get_youtube_comments(video_url, api_key, max_results=20):
    video_id = extract_video_id(video_url)
    if not video_id:
        return []
    youtube = build('youtube','v3',developerKey=api_key)
    comments = []
    request = youtube.commentThreads().list(part="snippet", videoId=video_id, maxResults=max_results, textFormat="plainText")
    response = request.execute()
    for item in response.get("items",[]):
        comments.append(item["snippet"]["topLevelComment"]["snippet"]["textDisplay"])
    return comments

def analyze_sentiment(text):
    clean_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([clean_text])
    prob = model.predict_proba(vectorized_text)[0]
    prediction = model.predict(vectorized_text)[0]
    return prediction, prob

def display_highlighted_review(text):
    positive_words = {"amazing","excellent","good","great","fantastic","wonderful","brilliant","outstanding","superb","love","enjoyed","perfect","incredible","beautiful","worth"}
    negative_words = {"bad","terrible","awful","poor","boring","worst","disappointing","waste","dull","annoying","horrible","slow","predictable","mediocre"}
    highlighted_text = []
    for word in text.split():
        clean_word = re.sub(r'[^a-zA-Z]','',word.lower())
        if clean_word in positive_words:
            highlighted_text.append(f"<span style='color:#00ff88;font-weight:bold;'>{word}</span>")
        elif clean_word in negative_words:
            highlighted_text.append(f"<span style='color:#ff4b4b;font-weight:bold;'>{word}</span>")
        else:
            highlighted_text.append(word)
    return " ".join(highlighted_text)

# ---- Analyze Button ----
if st.button("🚀 Analyze Sentiment"):
    # --- Manual Review ---
    if user_input.strip() != "":
        prediction, prob = analyze_sentiment(user_input)
        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
        st.markdown(f"📝 <b>Highlighted Review:</b><br>{display_highlighted_review(user_input)}", unsafe_allow_html=True)
        if prediction==1:
            st.markdown(f"<p class='positive'>✅ Positive 😄 | Confidence: {prob[1]*100:.2f}%</p>", unsafe_allow_html=True)
            rain(emoji="😄", font_size=40, falling_speed=5, animation_length="infinite")
        else:
            st.markdown(f"<p class='negative'>❌ Negative 😡 | Confidence: {prob[0]*100:.2f}%</p>", unsafe_allow_html=True)
            rain(emoji="😡", font_size=40, falling_speed=5, animation_length="infinite")
        st.markdown("</div>", unsafe_allow_html=True)
        # Bar chart
        fig, ax = plt.subplots()
        ax.bar(["Negative","Positive"], prob, color=["#ff4b4b","#00ff88"])
        ax.set_ylabel("Confidence")
        st.pyplot(fig)

    # --- YouTube Comments ---
    elif youtube_url.strip() != "":
        try:
            api_key = st.secrets["YOUTUBE_API_KEY"]
            with st.spinner("Fetching comments..."):
                comments = get_youtube_comments(youtube_url, api_key, max_comments)
            if not comments:
                st.info("No comments found or invalid video URL.")
            else:
                st.success(f"Fetched {len(comments)} comments. Analyzing sentiment...")
                pos_count, neg_count = 0, 0
                for comment in comments:
                    pred, prob = analyze_sentiment(comment)
                    if pred==1:
                        pos_count +=1
                        sentiment = f"Positive 😄 | Confidence: {prob[1]*100:.2f}%"
                    else:
                        neg_count +=1
                        sentiment = f"Negative 😡 | Confidence: {prob[0]*100:.2f}%"
                    st.markdown(f"**Comment:** {comment}\n**Sentiment:** {sentiment}")
                # Summary chart
                fig, ax = plt.subplots()
                ax.bar(["Negative","Positive"], [neg_count,pos_count], color=["#ff4b4b","#00ff88"])
                ax.set_ylabel("Number of Comments")
                st.pyplot(fig)
        except Exception as e:
            st.error(f"Error fetching YouTube comments: {e}")

    else:
        st.warning("⚠️ Please enter a manual review or a YouTube URL first.")
