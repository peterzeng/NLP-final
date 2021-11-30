import random
import re
import string

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt

# plotting: https://scikit-learn.org/stable/auto_examples/cluster/plot_mini_batch_kmeans.html
# WEIGHTED AVERAGE USING TF-IDF
def generate_vectors(corpus, model):
    features = []
    size = model.vector_size

    print(len(corpus))
    for tokens in corpus:
        zeros = np.zeros(size)
        vectors = []

        counter = 0
        for token in tokens:
            if token in model.wv:
                counter += 1
                try: 
                    vectors.append(model.wv[token])
                except KeyError:
                    continue
        if vectors:
            vectors = np.asarray(vectors)
            avg = vectors.mean(axis=0)
            features.append(avg)
        else:
            features.append(zeros)
            
    return features

def mini_kmeans(X: list, k: int, num_batches: int):
    model = MiniBatchKMeans(n_clusters = k, batch_size = num_batches).fit(X)
    # data_to_be_plotted = model.astype("int")

    print(f"For n_clusters = {k}")
    print(f"Silhouette coefficient: {silhouette_score(X, model.labels_):0.2f}")
    print(f"Inertia:{model.inertia_}")


    sample_silhouette_values = silhouette_samples(X, model.labels_)
    print(f"Silhouette values:")
    silhouette_values = []
    for i in range(k):
        cluster_silhouette_values = sample_silhouette_values[model.labels_ == i]
        silhouette_values.append(
            (
                i,
                cluster_silhouette_values.shape[0],
                cluster_silhouette_values.mean(),
                cluster_silhouette_values.min(),
                cluster_silhouette_values.max(),
            )
        )
    silhouette_values = sorted(
        silhouette_values, key=lambda tup: tup[2], reverse=True
    )
    for s in silhouette_values:
        print(f"    Cluster {s[0]}: Size:{s[1]} | Avg:{s[2]:.2f} | Min:{s[3]:.2f} | Max: {s[4]:.2f}")

    return model, model.labels_

if __name__ == "__main__":
    corpus = pd.read_csv("corpus.csv")
    processed = []
    for sequence in corpus["text"]:
        processed.append([sequence, word_tokenize(sequence)])

    processed = pd.DataFrame((processed), columns=["text", "tokens"])
    # processed.to_csv("tokenized_clustering_data.csv", header=True, index=False)

    # tokens = pd.read_csv("tokenized_clustering_data.csv")
    tokenized_data = processed["tokens"].values
    # print(tokenized_data[0])
    # print(tokenized_data)
    model = Word2Vec(sentences=tokenized_data, vector_size=100, workers=4)
    vectors = generate_vectors(tokenized_data, model)
    # print(len(vectors), len(vectors[0]))
    num_clusters = 10

    clustering, cluster_labels = mini_kmeans(
        X=vectors,
        k=num_clusters,
        num_batches=500,
    )
    
    df_clusters = pd.DataFrame({
        "text": corpus["text"],
        "tokens": [" ".join(text) for text in tokenized_data],
        "cluster": cluster_labels
    })

    print("Most representative terms per cluster (based on centroids):")
    for i in range(10):
        tokens_per_cluster = ""
        most_representative = model.wv.most_similar(positive=[clustering.cluster_centers_[i]], topn=5)
        for t in most_representative:
            tokens_per_cluster += f"{t[0]} "
        print(f"Cluster {i}: {tokens_per_cluster}")

    test_cluster = 1
    most_representative_docs = np.argsort(
        np.linalg.norm(vectors - clustering.cluster_centers_[test_cluster], axis=1)
    )
    for d in most_representative_docs[:3]:
        print(corpus["text"][d])
        print("-------------")