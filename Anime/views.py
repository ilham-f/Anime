from django.shortcuts import get_object_or_404, redirect, render
from django.http import HttpResponse
from django.contrib.auth.decorators import login_required
from django.contrib.auth import logout as auth_logout
from django.contrib.auth import login as auth_login
from django.contrib.auth.forms import AuthenticationForm
from .forms import UserCreationForm
from .models import Anime
from django.db.models import Count, Avg
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def register(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            auth_login(request, user)
            return redirect('home')
        else:
            print(form.errors)
    else:
        form = UserCreationForm()
    
    return render(request, 'registration/register.html', {'form': form})

def login(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            auth_login(request, user)
            return redirect('home')
        else:
            print(form.errors)
            # Handle authentication errors
            if 'username' in form.errors and 'password' in form.errors:
                # Both username and password are invalid
                form.add_error(None, 'Invalid username and password')
            elif 'username' in form.errors:
                # Only the username is invalid
                form.add_error(None, 'Account not found')
    else:
        form = AuthenticationForm()

    return render(request, 'registration/login.html', {'form': form})

def logout(request):
    auth_logout(request)
    return redirect('home')

# def profile(request):
#     return render(request,'home.html')
    
def home(request):
    user = request.user

    # Load the anime dataset from Django model
    anime_data = pd.DataFrame(list(Anime.objects.all().values()))

    # For simplicity, let's consider only anime with known ratings
    anime_data = anime_data[anime_data['rating'] != -1]

    # Drop rows with missing genres
    anime_data = anime_data.dropna(subset=['genres'])

    # Convert genres values to a list of strings
    anime_data['genres'] = anime_data['genres'].apply(lambda x: x.split(', ') if isinstance(x, str) else [])

    # Use MultiLabelBinarizer to create binary indicators for each genre
    mlb = MultiLabelBinarizer()
    genre_matrix = mlb.fit_transform(anime_data['genres'])

    # Create a DataFrame for the binary genres matrix
    genre_df = pd.DataFrame(genre_matrix, columns=mlb.classes_)

    # Concatenate the binary genres matrix with the original DataFrame
    anime_data = pd.concat([anime_data, genre_df], axis=1)

    # Example: Get content-based recommendations for an anime
    if user.is_authenticated and user.last_watched_anime is not None:
        anime_id = user.last_watched_anime
        anime = get_object_or_404(Anime, pk=anime_id)
        anime_title = anime.title

        # Function to get content-based recommendations for a given anime based on genres
        def get_genre_based_recommendations(anime_title, num_recommendations=5):
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
            genre_recommendations = [
                {
                    'id': anime_data.iloc[idx]['id'],
                    'title': anime_data.iloc[idx]['title'],
                    'similarity': genre_similarity[idx]
                }
                for idx in top_genre_recommendations
            ]

            return genre_recommendations

        def sequels_indices(anime_title):
            # Helper function to get indices of sequels for a given anime title
            anime_titles = anime_data['title']
            sequels_indices = [idx for idx, title in enumerate(anime_titles) if title.startswith(anime_title) and title != anime_title]
            return sequels_indices

        # Example: Get content-based recommendations for an anime based on genres
        genre_recommendations = get_genre_based_recommendations(anime_title)
    else:
        genre_recommendations = []
        anime_title = 0

    trendingAnimes = Anime.objects.all().order_by('-rating')[:10]

    if user.is_authenticated:
        auth_user = 1
    else:
        auth_user = 0

    if user.is_authenticated and user.last_watched_anime is not None:
        watched = 1
    else:
        watched = 0

    context = {
        'trending': trendingAnimes,
        'genreRecs': genre_recommendations,
        'anime_title': anime_title,
        'auth_user': auth_user,
        'watched': watched
    }
    print(genre_recommendations)
    return render(request,'home.html',context)

@login_required
def watch(request, anime_id):
    anime = get_object_or_404(Anime, pk=anime_id)
    user = request.user

    user.last_watched_anime = anime_id
    user.save()

    context = {
        'anime': anime,
        'auth_user': 1
    }

    return render(request,'watch.html',context)