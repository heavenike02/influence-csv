import os
import shutil

# Ensure the embedding storage directory exists at the start
embedding_storage_path = "./embedding_storage"
os.makedirs(embedding_storage_path, exist_ok=True)  # Create directory if it doesn't exist

# Debugging: Check if the directory was created successfully
if os.path.exists(embedding_storage_path):
    print(f"Directory '{embedding_storage_path}' exists.")
else:
    print(f"Failed to create directory '{embedding_storage_path}'.")    

# Check if the environment variable is necessary
if 'KMP_DUPLICATE_LIB_OK' not in os.environ:
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import streamlit as st
import pandas as pd
import hashlib
import pickle
import time

from txtai.embeddings import Embeddings


embeddings = Embeddings({"path": "sentence-transformers/all-MiniLM-L6-v2"})



def create_1d_string_list(data, cols):
    data_rows = data[cols].astype(str).values
    return [" ".join(row) for row in data_rows]


def get_data_hash(data):
    data_str = data.to_string()
    return hashlib.md5(data_str.encode()).hexdigest()


def load_embeddings_from_db(data_hash):
    try:
        file_path = f"{embedding_storage_path}/{data_hash}.pkl"
        print(f"Loading embeddings from: {file_path}")  # Debugging statement
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"File not found: {file_path}")  # Debugging statement
        return None  # Return None if the file does not exist


def save_embeddings_to_db(data_hash, embeddings):
    file_path = f"{embedding_storage_path}/{data_hash}.pkl"
    print(f"Saving embeddings to: {file_path}")  # Debugging statement
    with open(file_path, "wb") as f:
        pickle.dump(embeddings, f)


@st.cache_data
def index_data(data, search_field=None):
    data_hash = get_data_hash(data)
    cached_embeddings = load_embeddings_from_db(data_hash)
    if cached_embeddings is not None:
        return cached_embeddings

    if search_field:
        data_1d = data[search_field].astype(str).tolist()
    else:
        data_1d = create_1d_string_list(data, data.columns)
    
    total_items = len(data_1d)
    
    # Create a progress bar
    progress_bar = st.progress(0)
    
    # Create a placeholder for the percentage text
    percentage_text = st.empty()
    
    for i, (uid, text) in enumerate(enumerate(data_1d)):
        embeddings.index([(uid, text, None)])
        
        # Update progress
        progress = (i + 1) / total_items
        progress_bar.progress(progress)
        percentage = int(progress * 100)
        percentage_text.text(f"Indexing Progress: {percentage}%")
        
        # Add a small delay to make the progress visible
        time.sleep(0.01)

    percentage_text.text("Indexing Complete: 100%")
    
    save_embeddings_to_db(data_hash, embeddings)

    return embeddings


def search_with_scores(embeddings, query, limit):
    results = embeddings.search(query, limit)
    return [(uid, score) for uid, score in results]


# Function to clear the embedding storage directory
def clear_embedding_storage():
    if os.path.exists(embedding_storage_path):
        shutil.rmtree(embedding_storage_path)  # Remove the directory and all its contents
        os.makedirs(embedding_storage_path)  # Recreate the directory
        return True
    return False

# Streamlit app title
st.title("CSV File Query App")

# Button to clear history
if st.button("Clear History"):
    if clear_embedding_storage():
        st.success("History cleared successfully!")
    else:
        st.error("Failed to clear history.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file, encoding_errors="ignore")
    st.write("Data Preview:")
    st.write(data.head())

    # Add radio button for search type
    search_type = st.radio("Search Type", ["All Fields", "Single Field"])

    search_field = None
    if search_type == "Single Field":
        search_field = st.selectbox("Select field to search", data.columns)

    with st.spinner('Preparing to index data...'):
        embeddings = index_data(data, search_field)
    st.success('Indexing complete!')

    query = st.text_input("Enter Query", "")

    # Add a number input for result limit
    result_limit = st.number_input("Number of results", min_value=1, max_value=20, value=5)

    if query:
        try:
            st.write(f"Top {result_limit} results:")
            results_with_scores = search_with_scores(embeddings, query, result_limit)
            
            result_ids = [uid for uid, _ in results_with_scores]
            scores = [score for _, score in results_with_scores]
            
            result_df = data.loc[result_ids].reset_index(drop=True)
            result_df['Similarity Score'] = scores
            
            # Round the similarity score to 4 decimal places
            result_df['Similarity Score'] = result_df['Similarity Score'].apply(lambda x: round(x, 4))
            
            st.write(result_df)
        except Exception as e:
            st.error(f"An error occurred: {e}")
