import streamlit as st
from pandas import DataFrame
from keybert import KeyBERT
import seaborn as sns

st.set_page_config(
    page_title="Domain-Specific Keyword Extractor",
    page_icon="🔍",
)

def _max_width_():
    max_width_str = f"max-width: 1400px;"
    st.markdown(
        f"""
        <style>
            .reportview-container .main .block-container{{{max_width_str}}}
        </style>
        """,
        unsafe_allow_html=True,
    )

_max_width_()

with st.expander("ℹ️ - About this app", expanded=True):
    st.write(
        """
        - The *Domain-Specific Keyword Extractor* app is an interface built in Streamlit using the KeyBERT library.
        - It extracts domain-specific keywords/keyphrases from provided text leveraging NLP techniques.
        """
    )

st.markdown("")
st.markdown("## **📌 Paste document **")

with st.form(key="my_form"):
    doc = st.text_area(
        "Paste your text below (max 500 words)",
        height=300,
    )

    submit_button = st.form_submit_button(label="✨ Get Keywords!")

if submit_button:
    kw_model = KeyBERT()
    
    # You can replace this list with domain-specific stop words
    domain_specific_stopwords = ["specific_word_1", "specific_word_2"]  # Add your domain-specific stopwords
    
    keywords = kw_model.extract_keywords(doc, stop_words=domain_specific_stopwords)

    st.markdown("## **🎈 Extracted Keywords **")

    df = (
        DataFrame(keywords, columns=["Keyword/Keyphrase", "Relevance"])
        .sort_values(by="Relevance", ascending=False)
        .reset_index(drop=True)
    )

    df.index += 1

    # Add styling
    cmGreen = sns.light_palette("green", as_cmap=True)
    df_styled = df.style.background_gradient(cmap=cmGreen, subset=["Relevance"])
    format_dictionary = {"Relevance": "{:.2f}"}

    df_styled = df_styled.format(format_dictionary)

    st.table(df_styled)
