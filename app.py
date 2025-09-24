import streamlit as st
import os
import json
from transformers import pipeline
import spacy

# Load models (only once at startup)
@st.cache_resource
def load_models():
    nlp = spacy.load("en_core_web_sm")
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0)
    return nlp, classifier

nlp, classifier = load_models()

# Categories for stereotype detection
categories = ["profession present", "aspiration", "dependent relation", "objectification", "power attribute"]

# --- TEXT ANALYSIS FUNCTION ---
def analyze_text(text):
    doc = nlp(text)
    results = []
    for sent in doc.sents:
        output = classifier(sent.text, candidate_labels=categories)
        top_label = output["labels"][0]
        explanation = ""
        if top_label == "dependent relation":
            explanation = "Introduces character only in relation to another (e.g., daughter of, wife of)."
        elif top_label == "profession present":
            explanation = "Character is introduced with a profession."
        elif top_label == "aspiration":
            explanation = "Character is introduced with an aspiration/dream."
        elif top_label == "power attribute":
            explanation = "Character described with authority or control."
        elif top_label == "objectification":
            explanation = "Character described mainly in physical/relational terms."
        
        results.append({
            "sentence": sent.text,
            "label": top_label,
            "explanation": explanation
        })
    return results

# --- STREAMLIT APP UI ---
st.title("🎬 Bollywood Movie Gender Stereotype Analyzer")

option = st.sidebar.radio("Choose input type", ["Script / Wiki Text", "Poster (Image)", "Trailer (Video)"])

if option == "Script / Wiki Text":
    st.subheader("Paste your script or wiki text:")
    user_text = st.text_area("Enter text here", height=200)
    if st.button("Analyze Text"):
        if user_text.strip():
            analysis = analyze_text(user_text)

            # --- Human-friendly display ---
            st.subheader("📊 Analysis Results")
            for idx, item in enumerate(analysis, 1):
                st.markdown(f"### {idx}. Sentence")
                st.markdown(f"**➡️ Text:** {item['sentence']}")
                st.markdown(f"🔎 **Issue Identified:** {item['label'].capitalize()}")
                st.markdown(f"💡 **Explanation:** {item['explanation']}")
                st.markdown("---")
            
            # Optional: download results as JSON
            st.download_button(
                label="⬇️ Download Results as JSON",
                data=json.dumps(analysis, indent=4),
                file_name="analysis_results.json",
                mime="application/json"
            )
        else:
            st.warning("⚠️ Please enter some text before analyzing.")

elif option == "Poster (Image)":
    st.subheader("Upload a movie poster (image):")
    poster_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if poster_file:
        st.image(poster_file, caption="Uploaded Poster", use_column_width=True)
        st.info("⚡ Poster analysis coming soon (BLIP + OCR pipeline)")

elif option == "Trailer (Video)":
    st.subheader("Upload a short trailer (video):")
    trailer_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
    if trailer_file:
        st.video(trailer_file)
        st.info("⚡ Trailer analysis coming soon (Whisper + BLIP pipeline)")
