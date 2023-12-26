import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the anime dataset
anime_data = pd.read_csv('anime_data.csv')

# For simplicity, let's consider only anime with known ratings
anime_data = anime_data[anime_data['rating'] != -1]

# Drop 'genres' that has missing values
anime_data = anime_data.dropna(subset=['genres'])

anime_data.to_csv("anime_genre.csv");










# Convert genres values to a list of strings
anime_data['genres'] = anime_data['genres'].apply(lambda x: x.split(', ') if isinstance(x, str) else [])

# Use MultiLabelBinarizer to create binary indicators for each genres
mlb = MultiLabelBinarizer()
genre_matrix = mlb.fit_transform(anime_data['genres'])

# Create a DataFrame for the binary genres matrix
genre_df = pd.DataFrame(genre_matrix, columns=mlb.classes_)

# Concatenate the binary genres matrix with the original DataFrame
anime_data = pd.concat([anime_data, genre_df], axis=1)
# anime_data.to_csv("anime_genre.csv");

# Use TF-IDF vectorizer for title similarity
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(anime_data['title'])

# Calculate cosine similarity between anime based on titles
title_similarity = linear_kernel(tfidf_matrix, tfidf_matrix)

# Function to get content-based recommendations for a given anime
def get_content_based_recommendations(anime_title, num_recommendations=10):
    # Find the index of the anime in the dataset
    anime_index = anime_data.index[anime_data['title'] == anime_title].tolist()[0]

    # Get the cosine similarity scores for the anime
    title_sim_scores = list(enumerate(title_similarity[anime_index]))

    # Sort the anime based on title similarity
    title_sim_scores = sorted(title_sim_scores, key=lambda x: x[1], reverse=True)

    # Get the top N recommendations based on title similarity
    top_title_recommendations = title_sim_scores[1:num_recommendations + 1]

    # Display the recommended anime based on title similarity
    title_recommendations = [(anime_data.iloc[idx]['title'], sim_score) for idx, sim_score in top_title_recommendations]

    return title_recommendations

# Example: Get content-based recommendations for an anime
anime_title = 'One Piece'
title_recommendations = get_content_based_recommendations(anime_title)
print(f"Top 5 content-based recommendations after you watched {anime_title} based on title similarity:")
for title, similarity in title_recommendations:
    print(f"{title} (Similarity Score: {similarity:.2f})")

# Example: Get content-based recommendations for an anime based on genres
def get_genre_based_recommendations(anime_title, num_recommendations=10):
    # Find the index of the anime in the dataset
    anime_index = anime_data.index[anime_data['title'] == anime_title].tolist()[0]

    # Get the genre matrix for the anime
    genre_matrix_anime = genre_matrix[anime_index]

    # Calculate cosine similarity between anime based on genres
    genre_similarity = cosine_similarity([genre_matrix_anime], genre_matrix).flatten()

    # Exclude the input anime and its sequels from recommendations
    anime_exclude_indices = [anime_index] + sequels_indices(anime_title)
    genre_similarity[anime_exclude_indices] = -1  # Set similarity to -1 for exclusion

    # Get the indices of top N recommendations based on genre similarity
    top_genre_recommendations = genre_similarity.argsort()[-num_recommendations:][::-1]

    # Display the recommended anime based on genre similarity
    genre_recommendations = [(anime_data.iloc[idx]['title'], genre_similarity[idx]) for idx in top_genre_recommendations]

    return genre_recommendations

def sequels_indices(anime_title):
    # Helper function to get indices of sequels for a given anime title
    anime_titles = anime_data['title']
    sequels_indices = [idx for idx, title in enumerate(anime_titles) if title.startswith(anime_title) and title != anime_title]
    return sequels_indices

# Example: Get content-based recommendations for an anime based on genres
genre_recommendations = get_genre_based_recommendations(anime_title)
print(f"\nTop 5 content-based recommendations after you watched {anime_title} based on genre similarity:")
for title, similarity in genre_recommendations:
    print(f"{title} (Similarity Score: {similarity:.2f})")
