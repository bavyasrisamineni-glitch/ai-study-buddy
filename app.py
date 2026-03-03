import streamlit as st
from transformers import pipeline
import nltk
import speech_recognition as sr

# Download tokenizer
nltk.download('punkt')

st.set_page_config(page_title="AI-Powered Study Buddy")

st.title("📚 AI-Powered Study Buddy")

st.sidebar.title("Choose Feature")
option = st.sidebar.selectbox(
    "Select Tool",
    ("Explain Topic", "Summarize Notes", "Generate Quiz", "Speech to Notes")
)

# Load lighter models (better for Python 3.7 systems)
@st.cache_resource
def load_models():
    summarizer = pipeline("text-generation", model="t5-small")
    generator = pipeline("text-generation", model="gpt2")
    return summarizer, generator

summarizer, generator = load_models()

# -----------------------------------
# Explain Topic
# -----------------------------------
if option == "Explain Topic":
    st.header("Explain Any Topic Simply")

    topic = st.text_area("Enter Topic")

    if st.button("Explain"):
        if topic.strip() == "":
            st.warning("Please enter a topic.")
        else:
            prompt = "Explain in simple terms: " + topic
            result = generator(prompt, max_length=120, num_return_sequences=1)
            st.success(result[0]['generated_text'])

# -----------------------------------
# Summarize Notes
# -----------------------------------
elif option == "Summarize Notes":
    st.header("Summarize Study Notes")

    text = st.text_area("Paste Your Notes Here")

    if st.button("Summarize"):
        if len(text) < 50:
            st.warning("Please enter longer text.")
        else:
            summary = summarizer(
                "summarize: " + text,
                max_length=150,
                do_sample=False
            )
            st.success(summary[0]['generated_text'])
# -----------------------------------
# Generate Quiz
# -----------------------------------
elif option == "Generate Quiz":
    st.header("Generate Quiz Questions")

    content = st.text_area("Enter Topic")

    if st.button("Generate Quiz"):
        if content.strip() == "":
            st.warning("Please enter a topic.")
        else:
            prompt = "Generate 5 short quiz questions about " + content + ":"
            result = generator(prompt, max_length=150, num_return_sequences=1)
            st.success(result[0]['generated_text'])

# -----------------------------------
# Speech to Notes
# -----------------------------------
elif option == "Speech to Notes":
    st.header("Convert Speech to Text")

    audio_file = st.file_uploader("Upload WAV file", type=["wav"])

    if audio_file is not None:
        recognizer = sr.Recognizer()

        with sr.AudioFile(audio_file) as source:
            audio = recognizer.record(source)

        try:
            text = recognizer.recognize_google(audio)
            st.success("Transcribed Text:")
            st.write(text)

            st.subheader("Summary:")
            summary = summarizer("summarize: " + text,
                                 max_length=80,
                                 min_length=20,
                                 do_sample=False)
            st.write(summary[0]['summary_text'])

        except:
            st.error("Could not process audio.")