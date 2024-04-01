import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

# Load the DataFrame
df_tracks = pd.read_parquet("D:\\GUVI\Project\\0000 (1).parquet")
df_tracks.head()

def setting_bg(background_color):
    st.markdown(f"""<style>
                    .stApp {{
                        background-color: {background_color};
                        color: #a5ba93; /* Text color */
                    }}
                    </style>""", unsafe_allow_html=True)

# Call setting_bg function to set background color similar to Apple's website
setting_bg("#a5ba93")

st.markdown('<img width="200" height="200" src="https://img.icons8.com/clouds/100/spotify--v1.png" alt="spotify--v1"/>', unsafe_allow_html=True)
st.header(''':green[Spotify]''')

# Data Exploration and Cleaning
st.subheader(''':green[Data Exploration and Data Cleaning]''')
tab1, tab2= st.tabs([''':green[Head and shape]''', ''':green[Null Value]'''])

with tab1:
    st.header(":green[head and Shape of the Data]")
    tab1.write(df_tracks.head())
    tab1.write(df_tracks.shape)

with tab2:
    st.header("Shape and Header")
    tab2.write(pd.isnull(df_tracks).sum())
    st.header("Description of data")
    tab2.write(df_tracks.describe())
    st.header("Filling missing atrists name with unknown")
    tab2.write(df_tracks['artists'].fillna('Unknown', inplace=True))
    tab2.write(df_tracks.isnull().sum())
    st.header("Same with the album name and track name")
    tab2.write(df_tracks['album_name'].fillna('Unknown', inplace=True))
    tab2.write(df_tracks['track_name'].fillna('Unknown', inplace=True))
    tab2.write(df_tracks.isnull().sum())
    st.subheader("Data cleaning Done")

# Null value analysis
with st.expander("Data sorting"):
    sorted_df = df_tracks.sort_values('popularity', ascending = True). head(10)
    st.write(sorted_df)

    st.write("<h1 style='color:blue'>Switches the rows and columns</h1>",df_tracks.describe().transpose())

    most_popular=df_tracks.query('popularity>90', inplace = False).sort_values('popularity', ascending= False)
    st.write(most_popular[:10])

    st.write(df_tracks[['artists']].iloc[18])

# Duration Conversion
    df_tracks["duration"]= df_tracks["duration_ms"].apply(lambda x: round(x/1000))
    st.write(df_tracks.drop("duration_ms", inplace=True, axis=1))

    st.write(df_tracks.duration.head())

    st.write(df_tracks.duration)

# Select numeric columns only
    numeric_columns = df_tracks.select_dtypes(include=['float64', 'int64']).columns

# Drop non-numeric columns before computing the correlation matrix
corr_df = df_tracks[numeric_columns].corr(method="pearson")

sample_df=df_tracks.sample(int(0.020*len(df_tracks)))

print(len(sample_df))

options=st.multiselect("Select options", ['Correlation Heatmap', "Regression"])

if 'Correlation Heatmap' in options:
    # Correlation Heatmap
    st.subheader(":red[Heat Map between Variable]")
    plt.figure(figsize=(14, 6))
    heatmap = sns.heatmap(corr_df, annot=True, fmt=".1g", vmin=-1, vmax=1, center=0, cmap="inferno", linewidths=1, linecolor='black')
    heatmap.set_title("Correlation HeatMap Between Variables")
    heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=90)
    st.pyplot(plt)
if 'Regression' in options:
    st.subheader(":blue[Regression plot]")
    plt.figure(figsize=(12,6))
    sns.regplot(data = sample_df, y = "loudness", x = 'energy', color='c').set(title= "Loudness vs Energy Correction")
    st.pyplot(plt)


# Regression Plot
st.subheader(":blue[Popularity vs Acousticness Correction]")
plt.figure(figsize=(10,6))
sns.regplot(data = sample_df, y = "popularity", x = 'acousticness', color='b').set(title= "Popularity vs Acousticness Correction")
st.pyplot(plt)

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Select relevant audio features for clustering
audio_features = df_tracks[['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']]

# Standardize the features
scaler = StandardScaler()
scaled_audio_features = scaler.fit_transform(audio_features)

# Perform K-means clustering
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(scaled_audio_features)

# Add cluster labels to the dataframe
df_tracks['cluster'] = clusters

# Explore cluster distribution
print(df_tracks['cluster'].value_counts())

st.subheader(":green[Clustering]")
fig, ax = plt.subplots()
sns.scatterplot(x='energy', y='danceability', hue='cluster', data=df_tracks, ax=ax)
plt.title('Clustering of Tracks based on Energy and Danceability')
plt.xlabel('Energy')
plt.ylabel('Danceability')
st.pyplot(fig)

# Define the recommendation function
def recommend_tracks(num_tracks=5):
    # Sort the tracks by popularity in descending order
    popular_tracks = df_tracks.sort_values(by='popularity', ascending=False)
    # Select the top 'num_tracks' popular tracks
    recommended_tracks = popular_tracks.head(num_tracks)
    return recommended_tracks

# Display recommendations using Streamlit
st.header("Simple Track Recommendation System")

# Set the number of tracks to recommend
num_tracks_to_recommend = st.slider("Select number of tracks to recommend", 1, 10, 5)

# Display the recommended tracks
st.subheader(f"Top {num_tracks_to_recommend} Recommended Tracks:")
recommended_tracks = recommend_tracks(num_tracks_to_recommend)
st.write(recommended_tracks[['track_name', 'artists', 'popularity']])


# Define the recommendation function
def recommend_tracks(num_tracks=5):
    # Sort the tracks by popularity in descending order
    popular_tracks = df_tracks.sort_values(by='popularity', ascending=False)
    # Select the top 'num_tracks' popular tracks
    recommended_tracks = popular_tracks.head(num_tracks)
    return recommended_tracks

# Display recommendations using Streamlit
st.header("Track Recommendation System")

# Define the number of tracks to recommend
num_tracks_to_recommend = st.slider("Select number of tracks to recommend", 1, 100, 50)

# Display the recommended tracks
st.subheader(f"Top {num_tracks_to_recommend} Recommended Tracks:")
recommended_tracks = recommend_tracks(num_tracks_to_recommend)
st.write(recommended_tracks[['track_name', 'artists', 'popularity']])

# Define the recommendation function
def recommend_tracks_by_artist(num_artists=5):
    # Group tracks by artists and count the number of tracks for each artist
    artist_counts = df_tracks['artists'].value_counts().reset_index()
    artist_counts.columns = ['artists', 'track_count']
    # Sort the artists by the number of tracks in descending order
    popular_artists = artist_counts.sort_values(by='track_count', ascending=False)
    # Select the top 'num_artists' popular artists
    recommended_artists = popular_artists.head(num_artists)
    # Get the tracks for the recommended artists
    recommended_tracks = df_tracks[df_tracks['artists'].isin(recommended_artists['artists'])]
    return recommended_tracks

# Display recommendations using Streamlit
st.header("Track Recommendation System by Artist")

# Define the number of artists to recommend
num_artists_to_recommend = st.slider("Select number of artists to recommend", 1, 10, 5)

# Display the recommended tracks by artist
st.subheader(f"Top {num_artists_to_recommend} Recommended Artists:")
recommended_tracks = recommend_tracks_by_artist(num_artists_to_recommend)
st.write(recommended_tracks[['track_name', 'artists', 'popularity']])

# Define the recommendation function
def recommend_tracks_by_album(num_albums=5):
    # Group tracks by album and count the number of tracks for each album
    album_counts = df_tracks['album_name'].value_counts().reset_index()
    album_counts.columns = ['album_name', 'track_count']
    # Sort the albums by the number of tracks in descending order
    popular_albums = album_counts.sort_values(by='track_count', ascending=False)
    # Select the top 'num_albums' popular albums
    recommended_albums = popular_albums.head(num_albums)
    # Get the tracks for the recommended albums
    recommended_tracks = df_tracks[df_tracks['album_name'].isin(recommended_albums['album_name'])]
    return recommended_tracks

# Display recommendations using Streamlit
st.header("Track Recommendation System by Album")

# Define the number of albums to recommend
num_albums_to_recommend = st.slider("Select number of albums to recommend", 1, 10, 5)

# Display the recommended tracks by album
st.subheader(f"Top {num_albums_to_recommend} Recommended Albums:")
recommended_tracks = recommend_tracks_by_album(num_albums_to_recommend)
st.write(recommended_tracks[['track_name', 'album_name', 'artists', 'popularity']])
