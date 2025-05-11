from flask import Flask, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

app = Flask(__name__, static_folder='static')

netflix = pd.read_csv(r'C:\Users\tanke\Downloads\netflix_titles.csv')

filledna = netflix.fillna('')
def clean_data(x):
    return str.lower(x.replace(" ", ""))

features = ['title', 'director', 'cast', 'listed_in', 'description']
for feature in features:
    filledna[feature] = filledna[feature].apply(clean_data)

def create_soup(x):
    return x['title'] + ' ' + x['director'] + ' ' + x['cast'] + ' ' + x['listed_in'] + ' ' + x['description']

filledna['soup'] = filledna.apply(create_soup, axis=1)

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(filledna['soup'])

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

filledna = filledna.reset_index()
indices = pd.Series(filledna.index, index=filledna['title'])

def get_recommendation_new(title, cosine_sim=cosine_sim2):
    title = title.replace(' ', '').lower()
    if title not in indices:
        return ["No recommendations found. Please check the movie/series title."]
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return netflix['title'].iloc[movie_indices].tolist()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    movie_title = request.form['movie_title']
    recommendations = get_recommendation_new(movie_title)
    return render_template('index.html', recommendations=recommendations, input_movie=movie_title)

if __name__ == '__main__':
    app.run(debug=True)