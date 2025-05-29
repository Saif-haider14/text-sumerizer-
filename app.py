# app.py
import os
os.environ["STREAMLIT_PATHS_NO_WATCH"] = "1"

import streamlit as st
import torch
# rest of your imports and app logic


import streamlit as st
from transformers import pipeline

# Load pretrained summarizer
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

st.title("Text Summarizer using Pretrained AI Model")
st.markdown("This app summarizes long text into short, meaningful summaries.")

# User input
text_input = st.text_area("Enter text to summarize", height=300)

if st.button("Summarize"):
    if text_input.strip() != "":
        with st.spinner("Summarizing..."):
            summary = summarizer(text_input, max_length=120, min_length=30, do_sample=False)
            st.subheader("Summary")
            st.write(summary[0]['summary_text'])
    else:
        st.warning("Please enter some text to summarize.")
