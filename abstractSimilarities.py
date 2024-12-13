import requests
from bs4 import BeautifulSoup
import numpy as np
from scipy.spatial.distance import jaccard # untuk menghitung jaccard similarity
from sklearn.feature_extraction.text import TfidfVectorizer # untuk tf idf vectorizer
from sklearn.feature_extraction.text import CountVectorizer # untuk bow vectorizer
from sklearn.metrics.pairwise import cosine_similarity # untuk menghitung cosine similarity
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def fetch_abstracts_from_urls(urls):
    """
    Fetch abstracts from a list of URLs. Extracts abstracts from
    <div class="articleAbstract"> and looks for <em> tags within the inner div.
    Sesuai format di web IJCCS
    """
    all_abstracts = []

    for url in urls:
        print(f"Fetching abstracts from {url}...")
        try:
            response = requests.get(url)
            if response.status_code != 200:
                print(f"Failed to fetch {url}. Status code: {response.status_code}")
                continue

            soup = BeautifulSoup(response.content, 'html.parser')

            # Find the div with class 'articleAbstract'
            article_abstracts = soup.find_all('div', id='articleAbstract')

            for article_abstract in article_abstracts:
                # Find the inner div containing <em> tags
                inner_div = article_abstract.find('div')
                if inner_div:
                    # Extract the text from all <em> tags inside the inner div
                    abstract_texts = [em.text.strip() for em in inner_div.find_all('em')]
                    if abstract_texts:
                        all_abstracts.append(" ".join(abstract_texts))
                else:
                    print(f"No inner div found within articleAbstract on {url}.")

        except Exception as e:
            print(f"Error fetching {url}: {str(e)}")
    return all_abstracts


def preprocess_abstracts(abstracts):
    """
    Preprocess the list of abstracts by:
    1. Converting text to lowercase.
    2. Removing stopwords.
    """
    processed_abstracts = []
    
    for abstract in abstracts:
        # Convert to lowercase
        abstract = abstract.lower()

        # Remove stopwords
        filtered_abstract = " ".join(
            word for word in abstract.split() if word not in ENGLISH_STOP_WORDS
        )
        
        processed_abstracts.append(filtered_abstract)

    return processed_abstracts


def calculate_cosine_tfidf_similarity_matrix(abstracts):
    """
    Calculate the similarity matrix using TF-IDF and cosine similarity.
    """
    # Initialize TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()
    
    # Convert abstracts to TF-IDF matrix
    tfidf_matrix = vectorizer.fit_transform(abstracts)
    
    # Compute the cosine similarity matrix
    similarity_matrix = cosine_similarity(tfidf_matrix)

    return similarity_matrix

def calculate_jaccard_bow_similarity_matrix(abstracts):
    """
    Calculate the similarity matrix using Bag-of-Words (BoW) and Jaccard Similarity.
    """
    # Initialize CountVectorizer for Bag-of-Words representation
    vectorizer = CountVectorizer(binary=True)

    # Convert abstracts to BoW matrix
    bow_matrix = vectorizer.fit_transform(abstracts).toarray()

    # Compute the Jaccard Similarity matrix
    n_abstracts = len(bow_matrix)
    similarity_matrix = np.zeros((n_abstracts, n_abstracts))

    for i in range(n_abstracts):
        for j in range(n_abstracts):
            if i != j:
                similarity_matrix[i, j] = 1 - jaccard(bow_matrix[i], bow_matrix[j])
            else:
                similarity_matrix[i, j] = 1.0  # Jaccard similarity with itself is 1

    return similarity_matrix

def save_similarity_matrix(similarity_matrix, filename="similarity_matrix.csv"):
    """
    Save the similarity matrix as a CSV file.
    """
    # Round the similarity matrix to 2 decimal places
    rounded_matrix = np.round(similarity_matrix, 4)

    # Convert the matrix to a DataFrame
    similarity_df = pd.DataFrame(
        rounded_matrix,
        columns=[f"Abstract {i+1}" for i in range(len(similarity_matrix))],
        index=[f"Abstract {i+1}" for i in range(len(similarity_matrix))]
    )
    
    # Save the DataFrame to a CSV file
    similarity_df.to_csv(filename, index=True)
    print(f"Similarity matrix saved to '{filename}'.")


def main():
    # List of URLs to scrape (sumber dari ijccs)
    urls = [
        "https://jurnal.ugm.ac.id/ijccs/article/view/73334",
        "https://jurnal.ugm.ac.id/ijccs/article/view/78537",
        "https://jurnal.ugm.ac.id/ijccs/article/view/79623",
        "https://jurnal.ugm.ac.id/ijccs/article/view/80776",
        "https://jurnal.ugm.ac.id/ijccs/article/view/80077",
        "https://jurnal.ugm.ac.id/ijccs/article/view/80214",
        "https://jurnal.ugm.ac.id/ijccs/article/view/80049",
        "https://jurnal.ugm.ac.id/ijccs/article/view/90437",
        "https://jurnal.ugm.ac.id/ijccs/article/view/81178",
        "https://jurnal.ugm.ac.id/ijccs/article/view/80956"
    ]
    
    # Step 1: Fetch abstracts from the URLs
    abstracts = fetch_abstracts_from_urls(urls)
    print(f"Fetched {len(abstracts)} abstracts.")
    
    if abstracts:
        # Step 2: Preprocess abstract untuk menghilangkan stop word dan mengubah tulisan ke dalam huruf kecil
        preprocessed_abstracts = preprocess_abstracts(abstracts)

        # Step 3: Kalkulasikan similarity matrix
        cosine_similarity_matrix = calculate_cosine_tfidf_similarity_matrix(preprocessed_abstracts)
        jaccard_similarity_matrix = calculate_jaccard_bow_similarity_matrix(preprocessed_abstracts)
        # Step 4: Save similarity matrix ke dalam file csv
        save_similarity_matrix(cosine_similarity_matrix, filename="cosine_similarity_matrix.csv")
        save_similarity_matrix(jaccard_similarity_matrix, filename="jaccard_similarity_matrix.csv")
    else:
        print("No abstracts to process.")


if __name__ == "__main__":
    main()
