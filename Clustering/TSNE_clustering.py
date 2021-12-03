import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from nltk import word_tokenize
import nltk
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
nltk.download('punkt')
from sklearn.feature_extraction.text import TfidfVectorizer
import plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

def generate_vectors(corpus, model):
    features = []
    size = model.vector_size

    for tokens in corpus:
        ones = np.ones(size)
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
            features.append(ones)
            
    return features

if __name__ == '__main__':
    corpus = pd.read_csv("corpus.csv")
    processed = []
    for sequence in corpus["text"]:
        processed.append([sequence, word_tokenize(sequence)])
    
    processed = pd.DataFrame((processed), columns=['text', 'tokens'])

    # tokenized_data = processed["tokens"].values
    tokens = processed["tokens"].values
    model = Word2Vec(sentences=tokens, vector_size=100, workers=4)
    vectors = generate_vectors(tokens, model)
    # vectors = np.asarray(vectors)
    # print(vectors[0])
    # X = pd.DataFrame((vectors, corpus["author_num"].values), columns=['vector', 'author_num'])
    # 10 clusters TSNE visualization
    X = pd.DataFrame(vectors)
    num_clusters = 3
    kmeans_model = MiniBatchKMeans(n_clusters = num_clusters, batch_size=256)
    kmeans_model.fit(X)
    clusters = kmeans_model.predict(X)
    X["Cluster"] = clusters

    plotX = pd.DataFrame(X)
    plotX.columns = X.columns
    perplexity = 40

    tsne_1d = TSNE(n_components=1, perplexity=perplexity)
    tsne_2d = TSNE(n_components=2, perplexity=perplexity)
    TCs_1d = pd.DataFrame(tsne_1d.fit_transform(plotX.drop(["Cluster"], axis=1)))
    TCs_2d = pd.DataFrame(tsne_2d.fit_transform(plotX.drop(["Cluster"], axis=1)))
    TCs_1d.columns = ["TC1_1d"]
    TCs_2d.columns = ["TC1_2d","TC2_2d"]
 
    plotX = pd.concat([plotX,TCs_1d,TCs_2d], axis=1, join='inner')
    plotX["dummy"] = 0
    
    cluster0 = plotX[plotX["Cluster"] == 0]
    cluster1 = plotX[plotX["Cluster"] == 1]
    cluster2 = plotX[plotX["Cluster"] == 2]
    
    #Instructions for building the 1-D plot

    #trace1 is for 'Cluster 0'
    trace1 = go.Scatter(
                        x = cluster0["TC1_1d"],
                        y = cluster0["dummy"],
                        mode = "markers",
                        name = "Cluster 0",
                        marker = dict(color = 'rgba(255, 128, 255, 0.8)'),
                        text = None)

    #trace2 is for 'Cluster 1'
    trace2 = go.Scatter(
                        x = cluster1["TC1_1d"],
                        y = cluster1["dummy"],
                        mode = "markers",
                        name = "Cluster 1",
                        marker = dict(color = 'rgba(255, 128, 2, 0.8)'),
                        text = None)

    #trace3 is for 'Cluster 2'
    trace3 = go.Scatter(
                        x = cluster2["TC1_1d"],
                        y = cluster2["dummy"],
                        mode = "markers",
                        name = "Cluster 2",
                        marker = dict(color = 'rgba(0, 255, 200, 0.8)'),
                        text = None)

    data = [trace1, trace2, trace3]

    title = "Visualizing Clusters in One Dimension Using T-SNE (perplexity=" + str(perplexity) + ")"

    layout = dict(title = title,
                xaxis= dict(title= 'TC1',ticklen= 5,zeroline= False),
                yaxis= dict(title= '',ticklen= 5,zeroline= False)
                )

    fig = dict(data = data, layout = layout)

    iplot(fig)