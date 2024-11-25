import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# Set the page title
st.title("AI-Powered Redirect Mapping Tool")

# Step 1: Upload Files
st.header("Upload Files")
uploaded_origin = st.file_uploader("Upload origin.csv", type="csv")
uploaded_destination = st.file_uploader("Upload destination.csv", type="csv")

if uploaded_origin and uploaded_destination:
    origin_df = pd.read_csv(uploaded_origin)
    destination_df = pd.read_csv(uploaded_destination)

    st.success("Files uploaded successfully!")
    
    # Step 2: Column Selection
    st.header("Select Columns for Similarity Matching")
    common_columns = list(set(origin_df.columns) & set(destination_df.columns))
    selected_columns = st.multiselect("Choose columns for matching", common_columns)

    if selected_columns:
        # Step 3: Perform Matching
        st.header("Running Matching Process")
        origin_df['combined_text'] = origin_df[selected_columns].fillna('').apply(lambda x: ' '.join(x), axis=1)
        destination_df['combined_text'] = destination_df[selected_columns].fillna('').apply(lambda x: ' '.join(x), axis=1)

        # Load pre-trained model
        st.info("Loading pre-trained model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')

        # Vectorize text
        st.info("Generating embeddings...")
        origin_embeddings = model.encode(origin_df['combined_text'].tolist(), show_progress_bar=True)
        destination_embeddings = model.encode(destination_df['combined_text'].tolist(), show_progress_bar=True)

        # Create FAISS index
        dimension = origin_embeddings.shape[1]
        faiss_index = faiss.IndexFlatL2(dimension)
        faiss_index.add(destination_embeddings.astype('float32'))

        # Perform search
        D, I = faiss_index.search(origin_embeddings.astype('float32'), k=1)
        similarity_scores = 1 - (D / np.max(D))

        # Prepare output
        matches_df = pd.DataFrame({
            'origin_url': origin_df['Address'],
            'matched_url': destination_df['Address'].iloc[I.flatten()].values,
            'similarity_score': np.round(similarity_scores.flatten(), 4)
        })

        # Display and download output
        st.header("Results")
        st.write(matches_df)
        st.download_button(
            label="Download Results as CSV",
            data=matches_df.to_csv(index=False),
            file_name="redirect_mapping_output.csv",
            mime="text/csv",
        )
