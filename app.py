import streamlit as st
import numpy as np
from pandas import DataFrame
from keybert import KeyBERT
from flair.embeddings import TransformerDocumentEmbeddings
import seaborn as sns
from functionforDownloadButtons import download_button
import os
import json

# Set Streamlit page config
st.set_page_config(
    page_title="BERT Keyword Extractor",
    page_icon="🎈",
)

# Function to set max width for content
def _max_width_():
    max_width_str = f"max-width: 1400px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )

_max_width_()

# Streamlit layout setup
c30, c31, c32 = st.columns([2.5, 1, 3])

with c30:
    st.title("🔑 Institut des Algorithmes du Sénégal")
    st.header("")

with st.expander("ℹ️ - About this app", expanded=True):
    st.write(
        """     
-   The *BERT Keyword Extractor* app is an easy-to-use interface built in Streamlit for the amazing [KeyBERT](https://github.com/MaartenGr/KeyBERT) library from Maarten Grootendorst!
-   It uses a minimal keyword extraction technique that leverages multiple NLP embeddings and relies on [Transformers] (https://huggingface.co/transformers/) 🤗 to create keywords/keyphrases that are most similar to a document.
	    """
    )

st.markdown("")
st.markdown("## **📌 Paste document **")
with st.form(key="my_form"):
    # ... (Code for document form)
    # (The section for selecting model, setting parameters, and pasting text)

    # Define domain-specific words (replace this with your actual domain-specific words)
    domain_specific_words = ["domain_word_1", "domain_word_2", "domain_word_3", ...]

    if StopWordsCheckbox:
        stop_words = set(nltk.corpus.stopwords.words('english'))
        stop_words -= set(domain_specific_words)
    else:
        stop_words = None

    # Code for keyword extraction
    keywords = kw_model.extract_keywords(
        doc,
        keyphrase_ngram_range=(min_Ngrams, max_Ngrams),
        use_mmr=mmr,
        stop_words=stop_words,
        top_n=top_N,
        diversity=Diversity,
    )

    submit_button = st.form_submit_button(label="✨ Get me the data!")

# Code for generating download buttons and displaying keyword results
if not submit_button:
    st.stop()

# Display download buttons for different formats
st.markdown("## **🎈 Check & download results **")
st.header("")

cs, c1, c2, c3, cLast = st.columns([2, 1.5, 1.5, 1.5, 2])

with c1:
    CSVButton2 = download_button(keywords, "Data.csv", "📥 Download (.csv)")
with c2:
    CSVButton2 = download_button(keywords, "Data.txt", "📥 Download (.txt)")
with c3:
    CSVButton2 = download_button(keywords, "Data.json", "📥 Download (.json)")

st.header("")

# Display keyword results in a table
df = (
    DataFrame(keywords, columns=["Keyword/Keyphrase", "Relevancy"])
    .sort_values(by="Relevancy", ascending=False)
    .reset_index(drop=True)
)

df.index += 1

cmGreen = sns.light_palette("green", as_cmap=True)
cmRed = sns.light_palette("red", as_cmap=True)
df = df.style.background_gradient(
    cmap=cmGreen,
    subset=[
        "Relevancy",
    ],
)

c1, c2, c3 = st.columns([1, 3, 1])

format_dictionary = {
    "Relevancy": "{:.1%}",
}

df = df.format(format_dictionary)

with c2:
    st.table(df)
