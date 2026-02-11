# =========================================
# 🟣 News Topic Discovery Dashboard
# =========================================

import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

from scipy.cluster.hierarchy import dendrogram, linkage


# =========================================
# Page Settings
# =========================================

st.set_page_config(
    page_title="News Topic Discovery Dashboard",
    layout="wide"
)

st.title("🟣 News Topic Discovery Dashboard")

st.markdown("""
This system uses **Hierarchical Clustering** to automatically group
similar news articles based on textual similarity.
""")


# =========================================
# Sidebar: Dataset
# =========================================

st.sidebar.header("📂 Dataset Handling")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV File",
    type=["csv"]
)


# =========================================
# Load Dataset
# =========================================

if uploaded_file is not None:

    try:
        df = pd.read_csv(uploaded_file, encoding="utf-8")
    except:
        df = pd.read_csv(uploaded_file, encoding="latin1")

    st.success("Dataset Loaded Successfully")

    # Detect text column
    text_column = df.select_dtypes(include="object").columns[0]

    st.sidebar.write("Detected Text Column:")
    st.sidebar.code(text_column)

    texts = df[text_column].astype(str)


    # =====================================
    # Sidebar: TF-IDF Controls
    # =====================================

    st.sidebar.header("📝 Text Vectorization")

    max_features = st.sidebar.slider(
        "Maximum TF-IDF Features",
        100, 2000, 1000
    )

    use_stopwords = st.sidebar.checkbox(
        "Use English Stopwords",
        value=True
    )

    ngram_option = st.sidebar.selectbox(
        "N-gram Range",
        ["Unigrams", "Bigrams", "Unigrams + Bigrams"]
    )

    if ngram_option == "Unigrams":
        ngram_range = (1,1)
    elif ngram_option == "Bigrams":
        ngram_range = (2,2)
    else:
        ngram_range = (1,2)


    # =====================================
    # Sidebar: Clustering Controls
    # =====================================

    st.sidebar.header("🌳 Hierarchical Controls")

    linkage_method = st.sidebar.selectbox(
        "Linkage Method",
        ["ward", "complete", "average", "single"]
    )

    distance_metric = st.sidebar.selectbox(
        "Distance Metric",
        ["euclidean", "cosine"]
    )

    dendro_size = st.sidebar.slider(
        "Articles for Dendrogram",
        20, 200, 100
    )


    # =====================================
    # Sidebar: Clustering Control
    # =====================================

    st.sidebar.header("🟩 Apply Clustering")

    num_clusters = st.sidebar.slider(
        "Number of Clusters",
        2, 10, 5
    )


    # =====================================
    # Text Cleaning
    # =====================================

    def clean_text(text):

        text = text.lower()
        text = re.sub(r'[^a-zA-Z ]', '', text)
        text = re.sub(r'\s+', ' ', text)

        return text.strip()


    clean_texts = texts.apply(clean_text)


    # =====================================
    # TF-IDF Vectorization
    # =====================================

    stop_words = "english" if use_stopwords else None

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words=stop_words,
        ngram_range=ngram_range
    )

    X = vectorizer.fit_transform(clean_texts)

    # Convert to dense ONCE
    X_dense = X.toarray()


    # =====================================
    # Generate Dendrogram
    # =====================================

    if st.sidebar.button("🟦 Generate Dendrogram"):

        st.subheader("🌳 Dendrogram")

        X_sample = X_dense[:dendro_size]

        Z = linkage(
            X_sample,
            method=linkage_method,
            metric=distance_metric
        )

        fig, ax = plt.subplots(figsize=(12,6))

        dendrogram(Z, ax=ax)

        ax.set_title("Hierarchical Dendrogram")
        ax.set_xlabel("Article Index")
        ax.set_ylabel("Distance")

        st.pyplot(fig)

        st.info("""
        Inspect large vertical gaps to decide the number of clusters.
        """)


    # =====================================
    # Apply Clustering
    # =====================================

    if st.sidebar.button("Apply Clustering"):

        st.subheader("📊 Clustering Results")


        # ---------------------------------
        # Model
        # ---------------------------------

        hc = AgglomerativeClustering(
            n_clusters=num_clusters,
            linkage=linkage_method,
            metric=distance_metric
        )

        labels = hc.fit_predict(X_dense)   # ✅ FIXED

        df["Cluster"] = labels


        # ---------------------------------
        # PCA Visualization
        # ---------------------------------

        st.subheader("📈 Cluster Visualization (PCA)")

        pca = PCA(n_components=2)

        X_pca = pca.fit_transform(X_dense)

        fig, ax = plt.subplots()

        ax.scatter(
            X_pca[:,0],
            X_pca[:,1],
            c=labels
        )

        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.set_title("2D Projection of Clusters")

        st.pyplot(fig)


        # ---------------------------------
        # Cluster Summary
        # ---------------------------------

        st.subheader("📋 Cluster Summary")

        feature_names = vectorizer.get_feature_names_out()

        summary_data = []


        for i in range(num_clusters):

            cluster_docs = X[labels == i]

            if cluster_docs.shape[0] > 0:

                mean_tfidf = cluster_docs.mean(axis=0)

                top_idx = np.argsort(mean_tfidf.A1)[-10:]

                keywords = [feature_names[j] for j in top_idx]

                count = cluster_docs.shape[0]

                sample = df[df["Cluster"]==i][text_column].iloc[0][:200]

                summary_data.append([
                    i,
                    count,
                    ", ".join(keywords),
                    sample
                ])


        summary_df = pd.DataFrame(
            summary_data,
            columns=[
                "Cluster ID",
                "Number of Articles",
                "Top Keywords",
                "Sample Article"
            ]
        )

        st.dataframe(summary_df)


        # ---------------------------------
        # Validation Section
        # ---------------------------------

        st.subheader("📊 Validation")

        sil = silhouette_score(
            X_dense, labels,
            metric=distance_metric
        )   # ✅ FIXED

        st.metric("Silhouette Score", round(sil,4))

        st.info("""
        • Close to 1 → Well-separated clusters  
        • Close to 0 → Overlapping clusters  
        • Negative → Poor clustering
        """)


        # ---------------------------------
        # Business Interpretation
        # ---------------------------------

        st.subheader("🧠 Business Interpretation")

        for i in range(len(summary_df)):

            first_word = summary_df.iloc[i]["Top Keywords"].split(",")[0]

            st.write(
                f"🟣 Cluster {i}: Articles mainly related to {first_word} and similar topics."
            )


        # ---------------------------------
        # Insight Box
        # ---------------------------------

        st.subheader("💡 User Guidance")

        st.success("""
        Articles grouped in the same cluster share similar vocabulary and themes.
        These clusters can be used for automatic tagging, recommendations,
        and content organization.
        """)


else:

    st.warning("Please upload a CSV file to start.")
