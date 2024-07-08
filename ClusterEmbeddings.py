import csv
import os

import pandas as pd
from openai import OpenAI
from sklearn.datasets import make_blobs
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.metrics.pairwise import cosine_similarity

API_KEY = os.getenv('API_Key')
client = OpenAI(api_key=API_KEY)

def remove_prefix(location):
    location = location.lower()
    prefixes = ['a ', 'an ', 'the ']
    for prefix in prefixes:
        if location.startswith(prefix):
            return location[len(prefix):]
    return location.rstrip('.')

cartoons = []
char_list = []

# Read from list of characters
file = open("charactersFromImage.txt", "r")
contents = file.read()
for line in contents.splitlines():
    line = line.split(",")
    image = line[0]
    characters = line[1:]
    added = []

    for character in characters:
        character = character.strip()
        # Only get character once
        if character.lower() not in added:
            cartoons.append(image)
            character = remove_prefix(character.lower())
            char_list.append(character)
            added.append(character.lower())

df = pd.DataFrame({'Cartoon': cartoons, 'Character': char_list})
df.to_csv('CartoonCharacters.csv', index=False)
print(df.head(30))

def get_embedding(text, model="text-embedding-3-large"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model=model).data[0].embedding

embeddings = []
filename = "charembeddingsfromimage.txt"

# Write embeddings to "charembeddingsfromimage.txt"
file_object2 = open(filename, "w")
for index, row in df.iterrows():
    character = row['Character']
    print(index, character)
    if character != "":
        embedding = get_embedding(character)
        file_object2.write(f"{str(embedding)}\n")
        embeddings.append(embedding)

#with open(filename, 'r') as file:
#    Iterate over each line in the file
#    for line in file:
#       Process the line as needed
#       elements = line.strip()[1:-1].split(',')
#       Convert each element to float and create a numpy array
#       array_data = np.array([float(element) for element in elements])
#       embeddings.append(array_data)
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

# Optional: Reduce dimensions
#matrix = PCA(n_components=2).fit_transform(matrix)
#matrix = tsne.fit_transform(matrix)

# Normalize embeddings
for i, embedding in enumerate(matrix):
    magnitude = np.linalg.norm(embedding)
    matrix[i] = embedding / magnitude

# Assuming 'matrix' is your data
max_n = 50  # Maximum number of clusters to test
iterations_per_k = 10  # Number of iterations per number of clusters

# Initialize lists to store results
all_avg_silhouette_scores = []
all_avg_losses = []
all_losses = []

for n_clusters in range(2, max_n):  # Start from 2 clusters to calculate silhouette scores
    silhouette_scores_per_k = []
    losses_per_k = []

    for iteration in range(iterations_per_k):
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42 + iteration)
        labels = kmeans.fit_predict(matrix)

        # labels = kmeans.labels_
        df["Cluster"] = labels

        filename = "CharResultsImage" + str(n_clusters) + "_" + str(iteration) + ".txt"
        with open(filename, "w") as file_object:
            height = []
            for k in range(n_clusters):
                file_object.write(f"Cluster {k}:")
                file_object.write(df.loc[df['Cluster'] == k].to_string(index=False))
                height.append(len(df.loc[df['Cluster'] == k]))
                file_object.write("\n\n")
        print()

        # Store inertia (loss) for this iteration
        inertia = kmeans.inertia_
        losses_per_k.append(inertia)

        # Calculate overall silhouette score for this iteration
        silhouette_avg = silhouette_score(matrix, labels)
        silhouette_scores_per_k.append(silhouette_avg)

        filename = "charSilhouettesImage.txt"
        with open(filename, "a") as file_object:
        # Print each cluster's silhouette score for this iteration
            silhouette_vals = silhouette_samples(matrix, labels)
            file_object.write(f'Iteration {iteration + 1} with {n_clusters} clusters:')
            for cluster_label in range(n_clusters):
                cluster_silhouette_vals = silhouette_vals[labels == cluster_label]
                average_silhouette_score = np.mean(cluster_silhouette_vals)
                file_object.write(f'  Cluster {cluster_label + 1}: Silhouette Score = {average_silhouette_score:.4f}\n')

    # Calculate and store average silhouette and loss for this k
    avg_silhouette_score = np.mean(silhouette_scores_per_k)
    avg_loss = np.mean(losses_per_k)
    all_avg_silhouette_scores.append(avg_silhouette_score)
    all_avg_losses.append(avg_loss)
    all_losses.append(losses_per_k)

    print(f'\nAverage Silhouette Score for {n_clusters} clusters: {avg_silhouette_score:.4f}')
    print(f'Average Loss for {n_clusters} clusters: {avg_loss:.4f}')
    print('-' * 50)

# Specify the file to write to
file_path = 'charLossesImage.csv'

# Open the file and write to it
with open(file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Writing multiple rows
    writer.writerows(all_losses)

# Specify the file to write to
file_path = 'charSilhouttesImage.csv'

# Open the file and write to it
with open(file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Writing multiple rows
    writer.writerow(all_avg_silhouette_scores)

df.to_csv('charcluster_dataframeImage.csv', index=False)

# Plotting Average Silhouette Scores for each k
plt.figure(figsize=(10, 7))
plt.plot(range(2, max_n), all_avg_silhouette_scores, marker='o', linestyle='-', color='r')
plt.title('Average Silhouette Score for Each Number of Clusters k')
plt.xlabel('Number of Clusters')
plt.ylabel('Average Silhouette Score')
plt.grid(True)
plt.show()

# Plotting Average Loss for each k
plt.figure(figsize=(10, 7))
plt.plot(range(2, max_n), all_avg_losses, marker='o', linestyle='-', color='b')
plt.title('Average Loss for Each Number of Clusters')
print("here")
plt.xlabel('Number of Clusters')
plt.ylabel('Loss (Inertia)')
plt.grid(True)
plt.show()
