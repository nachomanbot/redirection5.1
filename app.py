import os
import re
import time
import hashlib
import numpy as np
import pandas as pd
import streamlit as st
from difflib import SequenceMatcher

# Try RapidFuzz (much faster than difflib); fall back gracefully
try:
    from rapidfuzz import fuzz, process as rf_process
    HAVE_RAPIDFUZZ = True
except Exception:
    HAVE_RAPIDFUZZ = False

# Semantic + FAISS (same as V5)
import faiss
from sentence_transformers import SentenceTransformer


# ---------------------------
# Helpers: parsing & indexing
# ---------------------------
_TOKEN_RE = re.compile(r"[a-z0-9]+")

def normalize(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip()
    # keep relative-like paths; lowercase
    return s.lower()

def tokenize(path: str) -> list[str]:
    return _TOKEN_RE.findall(normalize(path))

def first_segment(path: str) -> str:
    path = normalize(path)
    if path.startswith("/"):
        path = path[1:]
    return path.split("/", 1)[0] if path else ""

@st.cache_data(show_spinner=False)
def build_destination_indexes(dest_addresses: pd.Series):
    """
    Build:
      - bucket by first segment
      - inverted token index
      - set of all addresses
    Returns dicts of ints (row indices) to keep memory light.
    """
    seg_bucket: dict[str, list[int]] = {}
    inv_index: dict[str, set[int]] = {}
    addr_set = set()

    for i, addr in enumerate(dest_addresses):
        a = normalize(addr)
        addr_set.add(a)

        seg = first_segment(a)
        seg_bucket.setdefault(seg, []).append(i)

        for tok in tokenize(a):
            if len(tok) <= 1:
                continue
            inv_index.setdefault(tok, set()).add(i)

    return seg_bucket, inv_index, addr_set


# ---------------------------
# Lexical matching (fast path)
# ---------------------------
def best_lexical_match(origin: str, candidates: list[str]) -> tuple[str, float]:
    """
    Return (best_candidate, score_0_100). Uses RapidFuzz if available.
    """
    origin_n = normalize(origin)
    if not candidates:
        return "", 0.0

    if HAVE_RAPIDFUZZ:
        # WRatio tends to be robust for URL-ish strings
        match = rf_process.extractOne(origin_n, candidates, scorer=fuzz.WRatio)
        if match is None:
            return "", 0.0
        cand, score, _ = match  # score already 0..100
        return cand, float(score)
    else:
        # difflib fallback
        best, best_s = "", 0.0
        for c in candidates:
            s = SequenceMatcher(None, origin_n, normalize(c)).ratio() * 100.0
            if s > best_s:
                best, best_s = c, s
        return best, best_s


def candidate_indices_for_origin(origin: str,
                                 seg_bucket: dict[str, list[int]],
                                 inv_index: dict[str, set[int]],
                                 max_candidates: int = 200) -> list[int]:
    """
    Merge first-segment bucket + token-overlap candidates.
    """
    seg = first_segment(origin)
    cand_idx = set(seg_bucket.get(seg, []))

    for tok in tokenize(origin):
        ids = inv_index.get(tok)
        if ids:
            cand_idx |= ids
        if len(cand_idx) >= max_candidates:
            break

    # if still empty and we have a seg, try empty-seg (root)
    if not cand_idx and "" in seg_bucket:
        cand_idx = set(seg_bucket[""])

    return list(cand_idx)[:max_candidates]


# ---------------------------
# Semantic search (slow path for unmatched only)
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data(show_spinner=False)
def embed_destinations(dest_texts: list[str]) -> np.ndarray:
    model = load_model()
    emb = model.encode(dest_texts, show_progress_bar=False)
    return emb.astype('float32')

def embed_texts(texts: list[str]) -> np.ndarray:
    model = load_model()
    emb = model.encode(texts, show_progress_bar=True)
    return emb.astype('float32')

def semantic_candidates(unmatched_origin_texts: list[str],
                        dest_emb: np.ndarray,
                        top_k: int = 10):
    """
    Build a FAISS L2 index over destination embeddings and search.
    Returns (D, I) arrays (distances and indices).
    """
    if dest_emb.size == 0:
        return None, None
    dim = dest_emb.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(dest_emb)
    orig_emb = embed_texts(unmatched_origin_texts)
    D, I = index.search(orig_emb, top_k)
    return D, I


# ---------------------------
# Rules (precompiled)
# ---------------------------
def looks_like_regex(s: str) -> bool:
    return bool(re.search(r"[\\\^\$\.\*\+\?\{\}\[\]\|\(\)]", s or ""))

def parents(path: str):
    """
    /a/b/c -> /a/b, /a, /
    """
    p = normalize(path)
    while p and p != "/":
        p = p.rsplit("/", 1)[0] or "/"
        yield p

def precompile_rules(rules_df: pd.DataFrame):
    """
    Expect columns: Keyword, Destination URL Pattern, Priority
    """
    # coerce Priority to numeric and sort
    rules_df = rules_df.copy()
    if 'Priority' in rules_df.columns:
        rules_df['Priority'] = pd.to_numeric(rules_df['Priority'], errors='coerce').fillna(999999).astype(int)
        rules_df = rules_df.sort_values('Priority')
    else:
        rules_df['Priority'] = 1

    compiled = []
    for _, r in rules_df.iterrows():
        kw = normalize(r.get('Keyword', ''))
        dest_pat = str(r.get('Destination URL Pattern', '')).strip()
        alts = [normalize(x) for x in dest_pat.split('|') if x.strip()]

        # literal by default; allow power users to put real regex
        if looks_like_regex(kw):
            pat = re.compile(kw)
            compiled.append(("regex", pat, alts, kw))
        else:
            # match as substring (escaped)
            pat = re.compile(re.escape(kw))
            compiled.append(("substr", pat, alts, kw))
    return compiled


def apply_rules(origin_url: str,
                compiled_rules,
                dest_set: set[str]) -> tuple[str, str] | tuple[None, None]:
    """
    Try rules in priority order. Return (matched_url, reason) or (None, None).
    If an alternative path doesn't exist, climb to parent segments.
    """
    o = normalize(origin_url)
    for kind, pat, alts, kw_raw in compiled_rules:
        if not o:
            continue
        if pat.search(o):
            for alt in alts:
                if alt in dest_set:
                    return alt, f"rule:{kw_raw}"
                # parent walking
                for p in parents(alt):
                    if p in dest_set:
                        return p, f"rule:{kw_raw}>parent"
    return None, None


# ---------------------------
# Streamlit UI / Pipeline
# ---------------------------
st.title("AI-Powered Redirect Mapping Tool – V5.1 (optimized)")

st.markdown("""
This version speeds things up by:
- filtering lexical candidates by first path segment + token overlap  
- using RapidFuzz if available (auto-fallback to difflib)  
- embedding **only unmatched** origins and re-ranking FAISS top-k
""")

# Auth
password = st.text_input("Enter Password to Access the Tool:", type="password")
if password != "@SEOvaga!!!":
    st.warning("Please enter the correct password to proceed.")
    st.stop()

# Uploads
st.header("Upload Your Files")
uploaded_origin = st.file_uploader("Upload origin.csv", type="csv")
uploaded_destination = st.file_uploader("Upload destination.csv", type="csv")

# Load rules.csv co-located with the app
rules_path = 'rules.csv'
if os.path.exists(rules_path):
    rules_df = pd.read_csv(rules_path, encoding="ISO-8859-1")
else:
    st.error("rules.csv not found.")
    st.stop()

if uploaded_origin and uploaded_destination:
    st.success("Files uploaded successfully!")

    # Read CSVs (keep your encoding and column-normalization logic)
    try:
        origin_df = pd.read_csv(uploaded_origin, encoding="ISO-8859-1")
        destination_df = pd.read_csv(uploaded_destination, encoding="ISO-8859-1")
    except UnicodeDecodeError:
        st.error("Error reading CSV files. Please ensure they are UTF-8 or ISO-8859-1.")
        st.stop()

    # Clean BOM/whitespace; ensure Address column exists
    origin_df.columns = origin_df.columns.str.replace('ï»¿', '', regex=False).str.strip()
    destination_df.columns = destination_df.columns.str.replace('ï»¿', '', regex=False).str.strip()
    if 'Address' not in origin_df.columns:
        origin_df.rename(columns={origin_df.columns[0]: 'Address'}, inplace=True)
    if 'Address' not in destination_df.columns:
        destination_df.rename(columns={destination_df.columns[0]: 'Address'}, inplace=True)

    # Combined text (same as V5)
    origin_df['combined_text'] = origin_df.fillna('').apply(lambda x: ' '.join(x.astype(str)), axis=1)
    destination_df['combined_text'] = destination_df.fillna('').apply(lambda x: ' '.join(x.astype(str)), axis=1)

    # Settings
    st.header("Settings")
    prioritize_partial_match_toggle = st.radio(
        "Prioritize Matching Method:",
        ("Partial Match", "Similarity Score"),
        index=0,
        help="Choose whether to prioritize partial matches (meaningful URLs) or similarity scores (strong titles/descriptions)."
    )
    prioritize_partial = (prioritize_partial_match_toggle == "Partial Match")
    partial_threshold = st.slider("Partial Match Threshold (%)", 50, 100, 65, 5)
    semantic_threshold = st.slider("Blended Similarity Threshold (%)", 50, 100, 60, 5)
    top_k = st.slider("FAISS top-k (rerank pool)", 3, 25, 10, 1)
    max_cands = st.slider("Max lexical candidates per URL", 50, 500, 200, 50)

    if st.button("Let's Go!"):
        t0 = time.time()
        st.info("Indexing destinations...")
        seg_bucket, inv_index, dest_set = build_destination_indexes(destination_df['Address'].map(normalize))

        # Precompile rules once
        compiled_rules = precompile_rules(rules_df)

        # Prepare result table
        matches = []
        reason_counts = {"partial": 0, "semantic": 0, "rule": 0, "homepage": 0}

        # Pass 1: Lexical (fast)
        st.info("Pass 1/3: Lexical candidate matching…")
        progress = st.progress(0)
        unmatched_rows = []
        for i, row in origin_df.iterrows():
            origin_url = row['Address']
            origin_n = normalize(origin_url)

            # If user chose to skip partial, force unmatched
            if not prioritize_partial:
                unmatched_rows.append(i)
                matches.append([origin_url, "/", "", "No", ""])
                progress.progress(int((i+1)/len(origin_df)*100))
                continue

            cand_idx = candidate_indices_for_origin(origin_n, seg_bucket, inv_index, max_candidates=max_cands)
            cand_strings = [destination_df['Address'].iat[j] for j in cand_idx]
            best, s = best_lexical_match(origin_n, cand_strings)

            if s >= partial_threshold and best:
                matches.append([origin_url, best, round(s, 2), "No", "partial"])
                reason_counts["partial"] += 1
            else:
                unmatched_rows.append(i)
                matches.append([origin_url, "/", "", "No", ""])
            progress.progress(int((i+1)/len(origin_df)*100))

        matches_df = pd.DataFrame(matches, columns=["origin_url", "matched_url", "similarity_score", "fallback_applied", "match_method"])

        # Collect unmatched for semantic pass
        unmatched_mask = matches_df['matched_url'].eq("/")
        unmatched_idx = matches_df.index[unmatched_mask].tolist()

        # Pass 2: Semantic (only unmatched)
        if len(unmatched_idx):
            st.info("Pass 2/3: Semantic FAISS top-k + re-rank…")
            dest_emb = embed_destinations(destination_df['combined_text'].tolist())
            if dest_emb.size:
                # Prepare unmatched texts
                orig_texts = origin_df.loc[unmatched_idx, 'combined_text'].tolist()
                D, I = semantic_candidates(orig_texts, dest_emb, top_k=top_k)

                # Convert FAISS L2 distances to a 0..100 similarity-ish score per candidate
                # We'll rescale per row using max distance in that row's top-k.
                for r, row_i in enumerate(unmatched_idx):
                    origin_url = matches_df.at[row_i, 'origin_url']
                    # Build candidate list from FAISS
                    faiss_idx = I[r]
                    faiss_cands = [destination_df['Address'].iat[j] for j in faiss_idx if j >= 0]

                    # Lexical rescoring on this small pool
                    best_cand, lex_s = best_lexical_match(origin_url, faiss_cands)

                    # crude semantic score scale (relative within row)
                    row_D = D[r]
                    dmax = float(row_D.max()) if np.isfinite(row_D).any() else 1.0
                    # pick the distance of the best_cand (if present)
                    try:
                        j_idx = faiss_cands.index(best_cand)
                        d = float(row_D[j_idx])
                    except Exception:
                        d = float(row_D.min()) if np.isfinite(row_D).any() else 1.0

                    sem_s = (1.0 - (d / (dmax + 1e-9))) * 100.0  # 0..100
                    blended = 0.6 * lex_s + 0.4 * sem_s

                    if blended >= semantic_threshold and best_cand:
                        matches_df.at[row_i, 'matched_url'] = best_cand
                        matches_df.at[row_i, 'similarity_score'] = round(blended, 2)
                        matches_df.at[row_i, 'fallback_applied'] = "No"
                        matches_df.at[row_i, 'match_method'] = "semantic"
                        reason_counts["semantic"] += 1

        # Pass 3: Rules / Homepage
        st.info("Pass 3/3: Fallback rules…")
        remaining_mask = matches_df['matched_url'].eq("/")
        remaining_idx = matches_df.index[remaining_mask].tolist()

        for row_i in remaining_idx:
            origin_url = matches_df.at[row_i, 'origin_url']
            # Homepage-like origins go straight to homepage
            if normalize(origin_url) in ("/", "index.html", "index.php", "index.asp"):
                matches_df.at[row_i, 'matched_url'] = "/"
                matches_df.at[row_i, 'similarity_score'] = "Homepage"
                matches_df.at[row_i, 'fallback_applied'] = "Last Resort"
                matches_df.at[row_i, 'match_method'] = "homepage"
                reason_counts["homepage"] += 1
                continue

            alt, why = apply_rules(origin_url, compiled_rules, dest_set)
            if alt:
                matches_df.at[row_i, 'matched_url'] = alt
                matches_df.at[row_i, 'similarity_score'] = "Fallback"
                matches_df.at[row_i, 'fallback_applied'] = "Yes"
                matches_df.at[row_i, 'match_method'] = why
                reason_counts["rule"] += 1
            else:
                # last resort homepage
                matches_df.at[row_i, 'matched_url'] = "/"
                matches_df.at[row_i, 'similarity_score'] = "Homepage"
                matches_df.at[row_i, 'fallback_applied'] = "Last Resort"
                matches_df.at[row_i, 'match_method'] = "homepage"
                reason_counts["homepage"] += 1

        # Timing + summary
        elapsed = time.time() - t0
        total = len(origin_df)
        st.success(
            f"Done in {elapsed:.2f}s — total URLs: {total}. "
            f"Partial: {reason_counts['partial']} · "
            f"Semantic: {reason_counts['semantic']} · "
            f"Rules: {reason_counts['rule']} · "
            f"Homepage: {reason_counts['homepage']}"
        )

        # Show table
        st.write(matches_df)

        # Download
        st.download_button(
            label="Download Results as CSV",
            data=matches_df.to_csv(index=False),
            file_name="redirect_mapping_output_v5_1.csv",
            mime="text/csv",
        )
