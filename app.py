import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os
import re
from difflib import SequenceMatcher
from concurrent.futures import ThreadPoolExecutor
import time

# Set the page title
st.title("AI-Powered Redirect Mapping Tool - Version 4.0")

st.markdown("""

Relevancy Script made by Daniel Emery

Everything else by: NDA

⚡ **What It Is:**  
This tool automates redirect mappings during site migrations by matching URLs from an old site to a new site based on content similarity and custom fallback rules for unmatched URLs.

⚡ **How to Use It:**  
1. Upload `origin.csv` and `destination.csv` files. Ensure that your files have the following headers: Address,Title 1,Meta Description 1,H1-1.
2. Ensure that you remove any duplicates, and the http status of all URLs is 200. For best results, use relative URLs.
3. Customize the settings below to fit your use case.
4. Click **"Let's Go!"** to initiate the matching process.
5. Download the resulting `output.csv` file containing matched URLs with similarity scores or fallback rules.
""")

# Step 1: Upload Files
st.header("Upload Your Files")
uploaded_origin = st.file_uploader("Upload origin.csv", type="csv")
uploaded_destination = st.file_uploader("Upload destination.csv", type="csv")

# Load rules.csv from the backend
rules_path = 'rules.csv'  # Path to the rules CSV on the backend
if os.path.exists(rules_path):
    rules_df = pd.read_csv(rules_path, encoding="ISO-8859-1")
else:
    st.error("Rules file not found on the backend.")
    st.stop()

if uploaded_origin and uploaded_destination:
    st.success("Files uploaded successfully!")
    
    # Step 2: Load Data with Encoding Handling
    try:
        origin_df = pd.read_csv(uploaded_origin, encoding="ISO-8859-1")
        destination_df = pd.read_csv(uploaded_destination, encoding="ISO-8859-1")
    except UnicodeDecodeError:
        st.error("Error reading CSV files. Please ensure they are saved in a supported encoding (UTF-8 or ISO-8859-1).")
        st.stop()

    # Remove BOM and strip whitespace from column names
    origin_df.columns = origin_df.columns.str.replace('ï»¿', '').str.strip()
    destination_df.columns = destination_df.columns.str.replace('ï»¿', '').str.strip()

    # Check for required columns (use the first column if "Address" is not found)
    if 'Address' not in origin_df.columns:
        origin_df.rename(columns={origin_df.columns[0]: 'Address'}, inplace=True)
    if 'Address' not in destination_df.columns:
        destination_df.rename(columns={destination_df.columns[0]: 'Address'}, inplace=True)

    # Combine all columns for similarity matching
    origin_df['combined_text'] = origin_df.fillna('').apply(lambda x: ' '.join(x.astype(str)), axis=1)
    destination_df['combined_text'] = destination_df.fillna('').apply(lambda x: ' '.join(x.astype(str)), axis=1)

    # Step 3: User Customization Settings
    st.header("Settings")
    prioritize_partial_match = st.selectbox("Prioritize Partial Match over Similarity Scores?", ["Yes", "No"], index=0)
    partial_match_threshold = st.slider("Partial Match Threshold (in %)", min_value=50, max_value=100, value=65, step=5)
    similarity_score_threshold = st.slider("Similarity Score Threshold (in %)", min_value=50, max_value=100, value=60, step=5)

    # Step 4: Button to Process Matching
    if st.button("Let's Go!"):
        start_time = time.time()
        st.info("Processing data... This may take a while.")
        progress_bar = st.progress(0)

        # Step 5: Apply Partial Match First if Prioritized
        def get_partial_match_url(origin_url):
            highest_score = 0
            best_match = '/'
            for destination_url in destination_df['Address']:
                score = SequenceMatcher(None, origin_url.lower(), destination_url.lower()).ratio() * 100
                if score > highest_score:
                    highest_score = score
                    best_match = destination_url
            return best_match if highest_score > partial_match_threshold else '/'

        if prioritize_partial_match == "Yes":
            # Use ThreadPoolExecutor for parallel processing of partial matches
            with ThreadPoolExecutor() as executor:
                partial_matches = list(executor.map(get_partial_match_url, origin_df['Address']))

            # Apply partial matches before calculating similarity scores
            matches_df = pd.DataFrame({
                'origin_url': origin_df['Address'],
                'matched_url': partial_matches,
                'similarity_score': ['Partial Match'] * len(origin_df),
                'fallback_applied': ['Partial Match'] * len(origin_df)
            })
        else:
            # Initialize matches_df without partial matches
            matches_df = pd.DataFrame({
                'origin_url': origin_df['Address'],
                'matched_url': ['/'] * len(origin_df),
                'similarity_score': [''] * len(origin_df),
                'fallback_applied': ['No'] * len(origin_df)
            })

        # Step 6: Calculate Similarity Scores for URLs that still need it
        unmatched_indices = matches_df['matched_url'] == '/'
        if unmatched_indices.any():
            # Use a pre-trained model for embedding
            model = SentenceTransformer('all-MiniLM-L6-v2')

            # Vectorize the combined text
            origin_embeddings = model.encode(origin_df['combined_text'].tolist(), show_progress_bar=True)
            destination_embeddings = model.encode(destination_df['combined_text'].tolist(), show_progress_bar=True)

            # Create a FAISS index
            dimension = origin_embeddings.shape[1]
            faiss_index = faiss.IndexFlatL2(dimension)
            faiss_index.add(destination_embeddings.astype('float32'))

            # Perform the search for the nearest neighbors
            D, I = faiss_index.search(origin_embeddings.astype('float32'), k=1)

            # Calculate similarity scores
            similarity_scores = 1 - (D / (np.max(D) + 1e-10))  # Add small value to avoid division by zero

            # Update the DataFrame with similarity scores for unmatched URLs
            matches_df.loc[unmatched_indices, 'matched_url'] = destination_df.iloc[I[unmatched_indices].flatten()]['Address'].values
            matches_df.loc[unmatched_indices, 'similarity_score'] = np.round(similarity_scores[unmatched_indices].flatten() * 100, 2)
            matches_df.loc[unmatched_indices, 'fallback_applied'] = 'No'

        # Step 7: Apply Fallbacks for Remaining Low Scores
        low_score_indices = matches_df['similarity_score'].apply(lambda x: isinstance(x, float) and x < similarity_score_threshold)

        def get_fallback_url(origin_url):
            fallback_url = "/"  # Default fallback to homepage
            origin_url_normalized = origin_url.lower().strip().rstrip('/')
            
            # Apply CSV rules
            applicable_rules = rules_df.sort_values(by='Priority')  # Sort rules by priority
            for _, rule in applicable_rules.iterrows():
                keyword_normalized = rule['Keyword'].lower().strip().rstrip('/')
                if keyword_normalized in origin_url_normalized:
                    if rule['Destination URL Pattern'] in destination_df['Address'].values:
                        return rule['Destination URL Pattern']
            
            return fallback_url
        
        matches_df.loc[low_score_indices, 'matched_url'] = matches_df['origin_url'].apply(get_fallback_url)
        matches_df.loc[low_score_indices, 'similarity_score'] = 'Fallback'
        matches_df.loc[low_score_indices, 'fallback_applied'] = 'Yes'

        # Step 8: Final Check for Homepage Redirection
        homepage_indices = matches_df['origin_url'].str.lower().str.strip().isin(['/', 'index.html', 'index.php', 'index.asp'])
        matches_df.loc[homepage_indices, 'matched_url'] = '/'
        matches_df.loc[homepage_indices, 'similarity_score'] = 'Homepage'
        matches_df.loc[homepage_indices, 'fallback_applied'] = 'Last Resort'

        # Step 9: Display and Download Results
        end_time = time.time()
        total_time = end_time - start_time
        avg_time_per_url = total_time / len(origin_df)

        st.success(f"Matching complete in {total_time:.2f} seconds! Average processing time per URL: {avg_time_per_url:.2f} seconds. Total URLs processed: {len(origin_df)}.")
        st.write(matches_df)

        st.download_button(
            label="Download Results as CSV",
            data=matches_df.to_csv(index=False),
            file_name="redirect_mapping_output_v4.csv",
            mime="text/csv",
        )
