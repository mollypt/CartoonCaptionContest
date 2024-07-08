import csv
import os
import pandas as pd
from openai import OpenAI
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, silhouette_samples

API_KEY = os.getenv('API_KEY')
client = OpenAI(api_key=API_KEY)
file = open("anomaliesFromDescription.txt", "r")

images = []
anomaly_list = []

contents = file.read()
for line in contents.splitlines():
    line = line.split(", ", 1)
    print(line)
    image = line[0]
    anomaly = line[1]

    images.append(image)
    anomaly_list.append(anomaly.lower())

df = pd.DataFrame({'Cartoon': images, 'Anomaly': anomaly_list})


# Given a string, return the OpenAI embedding
def get_embedding(text, model="text-embedding-3-large"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model=model).data[0].embedding


embedding_model = "text-embedding-3-large"
embedding_encoding = "cl100k_base"
max_tokens = 8000

embeddings = []
filename = "anomaliesEmbeddings.txt"

# Get word embeddings and write to "anomaliesEmbeddings.txt"
file_object2 = open(filename, "w")
for index, row in df.iterrows():
    anomaly = row['Anomaly']
    print(index, anomaly)
    embedding = get_embedding(anomaly)
    file_object2.write(f"{str(embedding)}\n")
    embeddings.append(embedding)

# Use when "anomaliesEmbeddings.txt" already exists
#with open(filename, 'r') as file:
#    Iterate over each line in the file
#    for line in file:
#        Process the line as needed
#        elements = line.strip()[1:-1].split(',')
#        Convert each element to float and create a numpy array
#        array_data = np.array([float(element) for element in elements])
#        embeddings.append(array_data)
#    print(embeddings)

matrix = np.vstack(embeddings)

X, y = make_blobs(
    n_samples=500,
    n_features=2,
    centers=4,
    cluster_std=1,
    center_box=(-10.0, 10.0),
    shuffle=True,
    random_state=1,
)  # For reproducibility

all_losses = []
all_silhouette_scores = []  # List to store silhouette scores
max_n = 50
for i in range(1, max_n):
    loss = []
    silhouette_scores = []  # List to store silhouette scores for each iteration
    for j in range(20):
        # Perform K-Means
        n_clusters = i
        kmeans = KMeans(n_clusters=n_clusters, init="k-means++")
        kmeans.fit(matrix)

        labels = kmeans.labels_
        df["Cluster"] = labels

        # Calculate Silhouette Score
        if n_clusters > 1:  # Silhouette score is only meaningful if there are at least two clusters
            silhouette_avg = silhouette_score(matrix, labels)
            silhouette_scores.append(silhouette_avg)

        filename = "AnomalyResults" + str(i) + "_" + str(j) + ".txt"
        with open(filename, "w") as file_object:
            height=[]
            for k in range(n_clusters):
                file_object.write(f"Cluster {k}:")
                file_object.write(df.loc[df['Cluster'] == k].to_string(index=False))
                height.append(len(df.loc[df['Cluster'] == k]))
                file_object.write("\n\n")

        loss.append(kmeans.inertia_)

    all_losses.append(loss)
    all_silhouette_scores.append(silhouette_scores)  # Store silhouette scores

# Print Silhouette Scores
for m in range(len(all_silhouette_scores)):
    print(f"{m+1} Clusters: Silhouette Scores: {all_silhouette_scores[m]}")
    print()

# Collect losses and silhouette scores for evaluation
indices= []
values = []
for i in range(1, max_n-1):
    indices.append(i)
    values.append(sum(all_losses[i]) / len(all_losses[i]))

indices2= []
values2 = []
for i in range(1, max_n-1):
    indices2.append(i)
    values2.append(sum(all_silhouette_scores[i]) / len(all_silhouette_scores[i]))

# Plot loss curves
plt.figure(figsize=(10, 5))  # Optional: Specifies the size of the figure
plt.plot(indices, values, color='b')  # Line chart with points marked
plt.title('Setting Cluster Loss')  # Title of the plot
plt.xlabel('k')  # Label for the x-axis
plt.ylabel('Loss')  # Label for the y-axis
plt.grid(True)  # Optional: Adds a grid for better visibility of the axes scales
plt.show()  # Display the plot

# Plot silhouette scores
plt.figure(figsize=(10, 5))  # Optional: Specifies the size of the figure
plt.plot(indices2, values2, color='g')  # Line chart with points marked
plt.title('Setting Silhoutte Scores')  # Title of the plot
plt.xlabel('k')  # Label for the x-axis
plt.ylabel('Silhoutte Score')  # Label for the y-axis
plt.grid(True)  # Optional: Adds a grid for better visibility of the axes scales
plt.show()  # Display the plot

