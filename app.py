import os
import re
import time
from urllib.parse import urlsplit, parse_qsl

import faiss
import numpy as np
import pandas as pd
import streamlit as st
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer

# -----------------------------
# App Header
# -----------------------------
st.title("AI-Powered Redirect Mapping Tool – v5.2")

st.markdown("""
Relevancy Script made by Daniel Emery  
Everything else by: NDA

**What it does:**  
Maps URLs from an old site to a new site by (1) canonical fast matching, (2) partial URL similarity, (3) semantic similarity, and (4) CSV fallback rules.

**How to use:**  
1) Upload `origin.csv` and `destination.csv` with columns: **Address, Title 1, Meta Description 1, H1-1** (only Address is required).  
2) Tune settings (optional).  
3) Click **Let's Go!** and download the results.
""")

# -----------------------------
# Password Gate
# -----------------------------
password = st.text_input("Enter Password to Access the Tool:", type="password")
if password != "@SEOvaga!!!":
    st.warning("Please enter the correct password to proceed.")
    st.stop()

# -----------------------------
# File Uploaders
# -----------------------------
st.header("Upload Your Files")
uploaded_origin = st.file_uploader("Upload origin.csv", type="csv")
uploaded_destination = st.file_uploader("Upload destination.csv", type="csv")

# -----------------------------
# Load rules.csv from repo root
# -----------------------------
def load_rules_csv(path="rules.csv"):
    if not os.path.exists(path):
        st.error("rules.csv not found beside app.py")
        st.stop()
    # be tolerant with encodings
    for enc in ["utf-8", "ISO-8859-1", "utf-16", "cp1252"]:
        try:
            df = pd.read_csv(path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        st.error("Could not read rules.csv (encoding problem).")
        st.stop()

    # make sure required columns exist
    needed = {"Keyword", "Destination URL Pattern", "Priority"}
    missing = [c for c in needed if c not in df.columns]
    if missing:
        st.error(f"rules.csv is missing columns: {missing}")
        st.stop()
    # coerce types
    df["Keyword"] = df["Keyword"].astype(str).str.strip()
    df["Destination URL Pattern"] = df["Destination URL Pattern"].astype(str).str.strip()
    if "Priority" in df.columns:
        df["Priority"] = pd.to_numeric(df["Priority"], errors="coerce").fillna(999999).astype(int)
    return df

rules_df = load_rules_csv()

# -----------------------------
# Settings
# -----------------------------
st.header("Settings")
prioritize_partial_match_toggle = st.radio(
    "If fast canonical match fails, what should run first?",
    ("Partial Match", "Similarity Score"),
    index=0,
)
partial_match_threshold = st.slider("Partial Match Threshold (in %)", 50, 100, 65, step=5)
similarity_score_threshold = st.slider("Similarity Score Threshold (in %)", 50, 100, 60, step=5)

# -----------------------------
# Canonicalization helpers
# -----------------------------
PAGINATION_PARAMS = {"page", "pg", "p", "paged", "offset", "start", "limit"}
DROP_PARAM_PREFIXES = ("utm_", "idx-", "idx_d", "idxd", "idx-d", "gclid", "fbclid", "msclkid")

def _clean_path(p: str) -> str:
    if not p:
        return "/"
    p = re.sub(r"//+", "/", p)
    if not p.startswith("/"):
        p = "/" + p
    # ensure trailing slash for non-file paths
    if not re.search(r"\.[a-z0-9]{1,6}$", p) and not p.endswith("/"):
        p += "/"
    return p

def canonicalize_url(u: str) -> str:
    """Strip scheme/host, pagination segments/params, tracking junk. Return normalized path."""
    if not isinstance(u, str) or not u.strip():
        return "/"
    u = u.strip()

    # strip scheme+host
    u = re.sub(r"^https?://[^/]+", "", u, flags=re.I)

    parts = urlsplit(u)
    path = parts.path or "/"

    # remove pagination path segments like /page/2, /p/3, /paged/4
    path = re.sub(r"/(?:page|pg|p|paged)/\d+/?$", "/", path, flags=re.I)

    # special cases
    path = re.sub(r"/site-map/?$", "/", path, flags=re.I)

    path = _clean_path(path)

    # drop pagination/tracking/idx params entirely
    # (We ignore query for matching to maximize hit rate)
    return path.lower()

# -----------------------------
# Core
# -----------------------------
if uploaded_origin and uploaded_destination:
    st.success("Files uploaded successfully!")

    # Robust CSV reads
    def read_csv_flexible(uploaded):
        for enc in ["utf-8", "ISO-8859-1", "utf-16", "cp1252"]:
            try:
                return pd.read_csv(uploaded, encoding=enc)
            except UnicodeDecodeError:
                continue
        st.error("Could not read uploaded CSV (encoding problem).")
        st.stop()

    origin_df = read_csv_flexible(uploaded_origin)
    destination_df = read_csv_flexible(uploaded_destination)

    # Column cleanup
    origin_df.columns = origin_df.columns.astype(str).str.replace("ï»¿", "", regex=False).str.strip()
    destination_df.columns = destination_df.columns.astype(str).str.replace("ï»¿", "", regex=False).str.strip()

    if "Address" not in origin_df.columns:
        origin_df.rename(columns={origin_df.columns[0]: "Address"}, inplace=True)
    if "Address" not in destination_df.columns:
        destination_df.rename(columns={destination_df.columns[0]: "Address"}, inplace=True)

    # Combine all columns for similarity
    if "Title 1" not in origin_df.columns: origin_df["Title 1"] = ""
    if "Meta Description 1" not in origin_df.columns: origin_df["Meta Description 1"] = ""
    if "H1-1" not in origin_df.columns: origin_df["H1-1"] = ""

    if "Title 1" not in destination_df.columns: destination_df["Title 1"] = ""
    if "Meta Description 1" not in destination_df.columns: destination_df["Meta Description 1"] = ""
    if "H1-1" not in destination_df.columns: destination_df["H1-1"] = ""

    origin_df["combined_text"] = origin_df.fillna("").astype(str).apply(lambda x: " ".join(x.values), axis=1)
    destination_df["combined_text"] = destination_df.fillna("").astype(str).apply(lambda x: " ".join(x.values), axis=1)

    # Canonical forms
    origin_df["__canon"] = origin_df["Address"].apply(canonicalize_url)
    destination_df["__canon"] = destination_df["Address"].apply(canonicalize_url)

    # Lowercased sets for O(1) lookup
    DEST_SET = set(destination_df["__canon"].tolist())

    # -----------------------------
    # Fast canonical pass
    # -----------------------------
    def fast_canonical_match(canon_path: str) -> str:
        if canon_path in DEST_SET:
            return canon_path
        # try parent directory
        if canon_path != "/":
            parent = canon_path.rstrip("/")
            if "/" in parent:
                parent = parent.rsplit("/", 1)[0] + "/"
                if parent in DEST_SET:
                    return parent
        return "/"

    st.info("Phase 1/4: Fast canonical matching…")
    fast_matches = origin_df["__canon"].apply(fast_canonical_match)

    matches_df = pd.DataFrame({
        "origin_url": origin_df["Address"],
        "matched_url": fast_matches,
        "similarity_score": np.where(fast_matches != "/", "Canonical", ""),
        "fallback_applied": np.where(fast_matches != "/", "No", "No")
    })

    # -----------------------------
    # Partial match (optional first) and semantic similarity for unmatched
    # -----------------------------
    unmatched_mask = matches_df["matched_url"] == "/"

    # helper: partial best match
    def best_partial(origin_url: str) -> tuple[str, float]:
        if not isinstance(origin_url, str) or not origin_url:
            return "/", 0.0
        best_score = 0.0
        best_match = "/"
        ou = origin_url.lower()
        for du in destination_df["Address"].astype(str):
            s = SequenceMatcher(None, ou, du.lower()).ratio() * 100.0
            if s > best_score:
                best_score, best_match = s, du
        return (best_match if best_score >= partial_match_threshold else "/"), best_score

    if unmatched_mask.any():
        st.info("Phase 2/4: Partial matching & semantic similarity…")
        # Decide which goes first
        if prioritize_partial_match_toggle == "Partial Match":
            # Partial for all unmatched
            partial_results = [best_partial(u) for u in origin_df.loc[unmatched_mask, "Address"]]
            partial_urls = [u for (u, _) in partial_results]

            # Apply partial winners
            matches_df.loc[unmatched_mask, "matched_url"] = partial_urls
            matches_df.loc[(unmatched_mask) & (matches_df["matched_url"] != "/"), "similarity_score"] = "Partial"
            matches_df.loc[(unmatched_mask) & (matches_df["matched_url"] != "/"), "fallback_applied"] = "No"

            # Recompute who still needs similarity
            still_unmatched = matches_df["matched_url"] == "/"
            if still_unmatched.any():
                # Build FAISS once with destination embeddings
                model = SentenceTransformer("all-MiniLM-L6-v2")
                dest_emb = model.encode(destination_df["combined_text"].tolist(), show_progress_bar=False).astype("float32")
                index = faiss.IndexFlatL2(dest_emb.shape[1])
                index.add(dest_emb)

                # Only encode the still-unmatched origins
                sub = origin_df.loc[still_unmatched, "combined_text"].tolist()
                orig_emb = model.encode(sub, show_progress_bar=True).astype("float32")

                D, I = index.search(orig_emb, k=1)
                sim_scores = (1 - (D / (np.max(D) + 1e-10))).flatten() * 100.0
                # Write back
                dest_addrs = destination_df.iloc[I.flatten()]["Address"].values
                matches_df.loc[still_unmatched, "matched_url"] = dest_addrs
                matches_df.loc[still_unmatched, "similarity_score"] = np.round(sim_scores, 2)
                matches_df.loc[still_unmatched, "fallback_applied"] = "No"

        else:
            # Similarity first
            model = SentenceTransformer("all-MiniLM-L6-v2")
            dest_emb = model.encode(destination_df["combined_text"].tolist(), show_progress_bar=False).astype("float32")
            index = faiss.IndexFlatL2(dest_emb.shape[1])
            index.add(dest_emb)

            sub = origin_df.loc[unmatched_mask, "combined_text"].tolist()
            orig_emb = model.encode(sub, show_progress_bar=True).astype("float32")

            D, I = index.search(orig_emb, k=1)
            sim_scores = (1 - (D / (np.max(D) + 1e-10))).flatten() * 100.0
            dest_addrs = destination_df.iloc[I.flatten()]["Address"].values
            matches_df.loc[unmatched_mask, "matched_url"] = dest_addrs
            matches_df.loc[unmatched_mask, "similarity_score"] = np.round(sim_scores, 2)
            matches_df.loc[unmatched_mask, "fallback_applied"] = "No"

            # Partial next, but only for those still unmatched by threshold later

    # -----------------------------
    # Fallbacks for low scores
    # -----------------------------
    st.info("Phase 3/4: Applying fallback rules…")

    # Low-score rows (numeric & below threshold) OR rows still "/"
    is_low = matches_df["similarity_score"].apply(lambda x: isinstance(x, (int, float, np.floating)) and x < similarity_score_threshold)
    needs_fallback = (matches_df["matched_url"] == "/") | is_low

    # Make a quick set for existence checks (use canonical)
    DEST_CANON_SET = set(destination_df["__canon"].tolist())

    def pick_existing_destination(patterns: str) -> str:
        """Return first candidate from pipe-separated list that exists in destination (after canonicalization)."""
        for p in str(patterns).split("|"):
            c = canonicalize_url(p.strip())
            if c in DEST_CANON_SET:
                return c
        return "/"

    # Sort rules by ascending priority
    rules_sorted = rules_df.sort_values(by="Priority", ascending=True).reset_index(drop=True)

    def get_fallback_url(origin_url: str) -> str:
        origin_canon = canonicalize_url(origin_url)

        for _, rule in rules_sorted.iterrows():
            key = str(rule["Keyword"]).strip().lower()
            if not key:
                continue
            if re.search(re.escape(key), origin_canon):
                dest = pick_existing_destination(rule["Destination URL Pattern"])
                if dest != "/":
                    return dest

        # smart catch-alls (optional)
        if "/idx" in origin_canon or "property-search" in origin_canon:
            for candidate in ["/properties/sale", "/properties/"]:
                if canonicalize_url(candidate) in DEST_CANON_SET:
                    return canonicalize_url(candidate)

        return "/"

    matches_df.loc[needs_fallback, "matched_url"] = matches_df.loc[needs_fallback, "origin_url"].apply(get_fallback_url)
    matches_df.loc[needs_fallback, "similarity_score"] = np.where(
        matches_df.loc[needs_fallback, "matched_url"] != "/", "Fallback", "Homepage"
    )
    matches_df.loc[needs_fallback, "fallback_applied"] = np.where(
        matches_df.loc[needs_fallback, "matched_url"] != "/", "Yes", "Last Resort"
    )

    # -----------------------------
    # Homepages (explicit)
    # -----------------------------
    st.info("Phase 4/4: Finalizing homepages…")
    homepage_mask = origin_df["Address"].astype(str).str.lower().str.strip().isin(
        ["/", "index.html", "index.php", "index.asp"]
    )
    matches_df.loc[homepage_mask, ["matched_url", "similarity_score", "fallback_applied"]] = ["/", "Homepage", "Last Resort"]

    # -----------------------------
    # Output
    # -----------------------------
    total_time = time.time() - st.session_state.get("start_time", time.time())
    avg_time = total_time / max(len(origin_df), 1)

    st.success(
        f"Done! URLs: {len(origin_df)} | Avg/URL: {avg_time:.2f}s"
    )
    st.write(matches_df)

    st.download_button(
        label="Download Results as CSV",
        data=matches_df.to_csv(index=False),
        file_name="redirect_mapping_output_v5_2.csv",
        mime="text/csv",
    )
