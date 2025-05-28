import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Custom CSS for background color, styling, and button hover
st.markdown("""
    <style>
        body {
            background-color: black;
        }
        .main {
            
            padding: 20px;
            border-radius: 10px;
            color: purple;
        }
        .title {
            color: purple;
            font-size: 40px;
            font-weight: bold;
            text-align: center;
        }
        .subheader {
            font-size: 24px;
            color: purple;
        }
        /* Custom button style */
        div.stButton > button {
            background-color: purple;
            color: white;
            font-weight: bold;
            border: none;
            border-radius: 5px;
            padding: 0.5em 1em;
            transition: background-color 0.3s ease;
        }
        div.stButton > button:hover {
            background-color: #a020f0; /* lighter purple */
            color: yellow;
        }
    </style>
""", unsafe_allow_html=True)

# App Title
st.markdown('<div class="title">Saif\'s Text Summarizer</div>', unsafe_allow_html=True)

@st.cache_resource
def load_model():
    tokenizer = T5Tokenizer.from_pretrained("t5_summarizer_model")
    model = T5ForConditionalGeneration.from_pretrained("t5_summarizer_model")
    return tokenizer, model

tokenizer, model = load_model()

def summarize(text, max_length=50):
    input_text = "summarize: " + text.strip()
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(input_ids, max_length=max_length, min_length=10, do_sample=False)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Input and Output
st.markdown('<div class="main">', unsafe_allow_html=True)
user_input = st.text_area("Enter text to summarize:", height=250)

if st.button("Summarize"):
    if user_input.strip():
        summary = summarize(user_input)
        st.markdown('<div class="subheader">Summary:</div>', unsafe_allow_html=True)
        st.write(f"<span style='color:purple'>{summary}</span>", unsafe_allow_html=True)
    else:
        st.warning("Please enter some text first.")
st.markdown('</div>', unsafe_allow_html=True)
