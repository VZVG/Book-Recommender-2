from flask import Flask, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load and preprocess the dataset
file_path = 'C:/Users/91976/Desktop/files/All Codes/My Projects/Book Recommender 2/br2/books.csv'
books_df = pd.read_csv(file_path)

books_df['author'].fillna('', inplace=True)
books_df['desc'].fillna('', inplace=True)
books_df['publisher'].fillna('', inplace=True)

books_df['combined_features'] = books_df['title'] + ' ' + \
    books_df['author'] + ' ' + books_df['desc'] + ' ' + books_df['publisher']

# Create the TF-IDF matrix
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(books_df['combined_features'])
cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)


def get_recommendations(title, cosine_sim=cosine_sim_matrix):
    # Check if the title exists in the DataFrame
    if title not in books_df['title'].values:
        return []

    # Get the index of the book that matches the title
    idx = books_df[books_df['title'] == title].index[0]

    # Get the pairwise similarity scores of all books with that book
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the books based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar books
    sim_scores = sim_scores[1:11]

    # Get the book indices
    book_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar books
    return books_df['title'].iloc[book_indices].tolist()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/recommend', methods=['POST'])
def recommend():
    book_title = request.form['book_title']
    recommendations = get_recommendations(book_title, cosine_sim_matrix)
    if not recommendations:
        error_message = f"No recommendations found for '{book_title}'. Please try another title."
        return render_template('index.html', error_message=error_message, book_title=book_title)
    return render_template('index.html', recommendations=recommendations, book_title=book_title)


if __name__ == '__main__':
    app.run(debug=True)
