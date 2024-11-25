import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os

# Set the page title
st.title("AI-Powered Redirect Mapping Tool - Version 2.0")

st.markdown("""

Relevancy Script made by Daniel Emery

Everything else by: NDA

⚡ **What It Is:**  
This tool automates redirect mappings during site migrations by matching URLs from an old site to a new site based on content similarity and custom fallback rules for unmatched URLs.

⚡ **How to Use It:**  
1. Upload `origin.csv` and `destination.csv` files. Ensure that your files have the following headers: Address,Title 1,Meta Description 1,H1-1.
2. Ensure that you remove any duplicates and the http status of all URLs is 200.
3. Click **"Let's Go!"** to initiate the matching process.
4. Download the resulting `output.csv` file containing matched URLs with similarity scores or fallback rules.
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

    # Combine all columns for similarity matching
    origin_df['combined_text'] = origin_df.fillna('').apply(lambda x: ' '.join(x.astype(str)), axis=1)
    destination_df['combined_text'] = destination_df.fillna('').apply(lambda x: ' '.join(x.astype(str)), axis=1)

    # Step 3: Button to Process Matching
    if st.button("Let's Go!"):
        st.info("Processing data... This may take a while.")
        progress_bar = st.progress(0)

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

        # Create the output DataFrame with similarity scores
        matches_df = pd.DataFrame({
            'origin_url': origin_df.iloc[:, 0],
            'matched_url': destination_df.iloc[:, 0].iloc[I.flatten()].apply(lambda x: x.split()[0]).values,  # Ensure only the URL is added
            'similarity_score': np.round(similarity_scores.flatten(), 4),
            'fallback_applied': ['No'] * len(origin_df)  # Default to 'No' for fallback
        })

        # Step 4: Apply Fallbacks for Low Scores
        fallback_threshold = 0.6
        low_score_indices = matches_df['similarity_score'] < fallback_threshold
        low_score_matches = matches_df[low_score_indices]

        # Apply fallbacks using vectorized operations
        def get_fallback_url(origin_url):
            fallback_url = "/"  # Default fallback to homepage
            origin_url_normalized = origin_url.lower().strip().rstrip('/')
            
            # Apply CSV rules
            applicable_rules = rules_df.sort_values(by='Priority')  # Sort rules by priority
            for _, rule in applicable_rules.iterrows():
                keyword_normalized = rule['Keyword'].lower().strip().rstrip('/')
                if keyword_normalized in origin_url_normalized:
                    return rule['Destination URL Pattern']
            
            return fallback_url
        
        matches_df.loc[low_score_indices, 'matched_url'] = low_score_matches['origin_url'].apply(get_fallback_url)
        matches_df.loc[low_score_indices, 'similarity_score'] = 'Fallback'
        matches_df.loc[low_score_indices, 'fallback_applied'] = 'Yes'

        # Step 5: Display and Download Results
        st.success("Matching complete! Download your results below.")
        st.write(matches_df)

        st.download_button(
            label="Download Results as CSV",
            data=matches_df.to_csv(index=False),
            file_name="redirect_mapping_output_v2.csv",
            mime="text/csv",
        )
