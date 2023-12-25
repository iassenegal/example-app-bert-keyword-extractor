import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
import numpy as np
from pandas import DataFrame
from keybert import KeyBERT
from flair.embeddings import TransformerDocumentEmbeddings
import seaborn as sns
from functionforDownloadButtons import download_button
import os
import matplotlib.pyplot as plt
import json
from wordcloud import WordCloud

st.set_page_config(
    page_title="BERT Keyword Extractor",
    page_icon="🎈",
)

# Function to scrape content from a URL
def scrape_content(url):
    try:
        page = requests.get(url)
        soup = BeautifulSoup(page.content, "html.parser")
        paragraphs = soup.find_all("p")
        content = " ".join([p.text.strip() for p in paragraphs])
        return content
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

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

c30, c31, c32 = st.columns([2.5, 1, 3])

with c30:
    st.title("🔑 BERT Keyword Extractor")
    st.header("")

with st.expander("ℹ️ - About this app", expanded=True):
    st.write(
        """     
-   The *BERT Keyword Extractor* app is an easy-to-use interface built in Streamlit for the amazing [KeyBERT](https://github.com/MaartenGr/KeyBERT) library from Maarten Grootendorst!
-   It uses a minimal keyword extraction technique that leverages multiple NLP embeddings and relies on [Transformers](https://huggingface.co/transformers/) 🤗 to create keywords/keyphrases that are most similar to a document.
	    """
    )
    st.markdown("")

st.markdown("## **📌 Paste document or Enter URL**")
with st.form(key="my_form"):
    ce, c1, ce, c2, c3 = st.columns([0.07, 1, 0.07, 5, 0.07])
    with c1:
        option = st.radio(
            "Choose input type",
            ["Paste Text", "Enter URL"],
            help="Choose how you want to input the document: directly paste text or enter a URL to scrape content.",
        )
        if option == "Paste Text":
            doc = st.text_area(
                "Paste your text below (max 500 words)",
                height=510,
            )
        else:
            url = st.text_input("Enter URL to scrape content")
            if st.button("Get content"):
                doc = scrape_content(url)

                MAX_WORDS = 500
                if doc and len(re.findall(r"\w+", doc)) > MAX_WORDS:
                    st.warning(
                        f"⚠️ The scraped text contains more than {MAX_WORDS} words. Only the first {MAX_WORDS} words will be reviewed."
                    )
                    doc = doc[:MAX_WORDS]                

            else:
                doc = None

        submit_button = st.form_submit_button(label="✨ Get me the data!")

    if use_MMR:
        mmr = True
    else:
        mmr = False

    if StopWordsCheckbox:
        StopWords = "english"
    else:
        StopWords = None

if not submit_button:
    st.stop()

if min_Ngrams > max_Ngrams:
    st.warning("min_Ngrams can't be greater than max_Ngrams")
    st.stop()

keywords = kw_model.extract_keywords(
    doc,
    keyphrase_ngram_range=(min_Ngrams, max_Ngrams),
    use_mmr=mmr,
    stop_words=StopWords,
    top_n=top_N,
    diversity=Diversity,
)

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

df = (
    DataFrame(keywords, columns=["Keyword/Keyphrase", "Relevancy"])
    .sort_values(by="Relevancy", ascending=False)
    .reset_index(drop=True)
)
##############

# Assuming 'df' is your DataFrame containing keywords/keyphrases
top_keywords = df.head(10)["Keyword/Keyphrase"].tolist()
wordcloud_text = ' '.join(top_keywords)

wordcloud = WordCloud(width=600, height=200, background_color='white').generate(wordcloud_text)

# Display the word cloud
st.markdown("## **🌟 Word Cloud of Top 10 Relevant Keywords**")
fig, ax = plt.subplots(figsize=(6, 5))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis('off')
st.pyplot(fig)
#################
df.index += 1

# Add styling
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

