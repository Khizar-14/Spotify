import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


# Load the DataFrame
df_tracks = pd.read_parquet("D:\\GUVI\Project\\0000 (1).parquet")
df_tracks.head()

def setting_bg(background_color):
    st.markdown(f""" <style>.stApp {{
                        background-color:{background_color};
                       }}
                       </style>""", unsafe_allow_html=True)
setting_bg("#90ee90")

# Data Exploration and Cleaning
st.markdown("Data Exploration and Data Cleaning")
tab1, tab2= st.tabs(["Head and shape", "Null value"])

with tab1:
    st.header("head and Shape of the Data")
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

# Correlation Heatmap
st.subheader("Heat Map between Variable")
plt.figure(figsize=(14, 6))
heatmap = sns.heatmap(corr_df, annot=True, fmt=".1g", vmin=-1, vmax=1, center=0, cmap="inferno", linewidths=1, linecolor='black')
heatmap.set_title("Correlation HeatMap Between Variables")
heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=90)

st.pyplot(plt)

sample_df=df_tracks.sample(int(0.020*len(df_tracks)))

print(len(sample_df))

# Regression Plot
plt.figure(figsize=(10,6))
sns.regplot(data = sample_df, y = "loudness", x = 'energy', color='c').set(title= "Loudness vs Energy Correction")
st.pyplot(plt)

# Regression Plot
plt.figure(figsize=(10,6))
sns.regplot(data = sample_df, y = "popularity", x = 'acousticness', color='b').set(title= "Popularity vs Acousticness Correction")
st.pyplot(plt)


plt.title("Duration of the songs in Different Genress")
sns.color_palette("rocket", as_cmap=True)
plot=sns.barplot(y="track_genre", x="duration", data=sample_df)
plt.xlabel("Duration in milli seconds")
plt.ylabel("Genres")
# Get the Figure object from the plot
fig = plot.get_figure()

    # Display the plot in Streamlit
st.pyplot(fig)



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


plot1=sns.scatterplot(x='energy', y='danceability', hue='cluster', data=df_tracks)
plt.title('Clustering of Tracks based on Energy and Danceability')
plt.xlabel('Energy')
plt.ylabel('Danceability')

# Save the plot
plt.savefig('clustering_plot.png')

    # Display the plot in Streamlit
st.pyplot(plot1.figure)

