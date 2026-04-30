import time
import io
import requests
import pandas as pd
import streamlit as st
from typing import Optional, Dict, Any, List

# ----------------------------
# OpenAlex helpers
# ----------------------------
BASE_URL = "https://api.openalex.org/works"

def invert_index_to_text(inv: Optional[Dict[str, List[int]]]) -> str:
    """
    OpenAlex abstracts are often stored as an inverted index:
    { "word": [positions], ... }
    Reconstruct original token order.
    """
    if not inv or not isinstance(inv, dict):
        return ""
    positions = {}
    for word, idxs in inv.items():
        if not isinstance(idxs, list):
            continue
        for i in idxs:
            positions[i] = word
    if not positions:
        return ""
    return " ".join(positions[i] for i in sorted(positions.keys()))

def build_params(
    cursor: str,
    mode: str,
    query: str,
    year_from: Optional[int],
    year_to: Optional[int],
    api_key: Optional[str],
    mailto: Optional[str],
    per_page: int = 200,
) -> Dict[str, Any]:
    # OpenAlex Works supports filters like has_doi and has_abstract.  [1](https://developers.openalex.org/api-reference/works)
    filters = ["has_doi:true", "has_abstract:true"]

    if year_from is not None:
        filters.append(f"from_publication_date:{year_from}-01-01")
    if year_to is not None:
        filters.append(f"to_publication_date:{year_to}-12-31")

    params = {
        "filter": ",".join(filters),
        "per_page": per_page,
        "cursor": cursor,
        # Select only what we need to reduce payload
        "select": ",".join([
            "id",
            "doi",
            "display_name",
            "abstract_inverted_index",
            "publication_year",
            "primary_location",
        ]),
    }

    if mode == "Life science keyword" and query.strip():
        params["search"] = query.strip()

    if api_key and api_key.strip():
        params["api_key"] = api_key.strip()

    if mailto and mailto.strip():
        params["mailto"] = mailto.strip()

    return params

def fetch_openalex_sample(
    n_rows: int,
    mode: str,
    query: str,
    year_from: Optional[int],
    year_to: Optional[int],
    api_key: Optional[str],
    mailto: Optional[str],
    sleep_s: float,
    progress_cb=None,
    status_cb=None,
) -> pd.DataFrame:
    session = requests.Session()
    headers = {"User-Agent": "streamlit-openalex-sampler/1.0"}

    cursor = "*"
    rows = []
    collected = 0
    page_count = 0

    while collected < n_rows:
        params = build_params(
            cursor=cursor,
            mode=mode,
            query=query,
            year_from=year_from,
            year_to=year_to,
            api_key=api_key,
            mailto=mailto,
            per_page=200,
        )

        r = session.get(BASE_URL, params=params, headers=headers, timeout=60)
        r.raise_for_status()
        data = r.json()

        results = data.get("results", [])
        meta = data.get("meta", {})
        cursor = meta.get("next_cursor")
        page_count += 1

        if status_cb:
            status_cb(f"Fetched page {page_count} | collected {collected}/{n_rows}")

        if not results:
            break

        for w in results:
            doi = w.get("doi") or ""
            title = w.get("display_name") or ""
            year = w.get("publication_year") or None
            pl = w.get("primary_location") or {}
            src = pl.get("source") or {}
            journal = src.get("display_name") or ""
            openalex_id = w.get("id") or ""

            abstract = invert_index_to_text(w.get("abstract_inverted_index"))

            # Keep only robust records for clustering: DOI + title + abstract
            if doi and title and abstract:
                rows.append({
                    "DOI": doi,
                    "Title": title,
                    "Abstract": abstract,
                    "Journal": journal,
                    "PublicationYear": year,
                    "OpenAlexID": openalex_id,
                    "OpenAlexURL": openalex_id,  # id is already a URL
                })
                collected += 1
                if collected >= n_rows:
                    break

        if progress_cb:
            progress_cb(min(collected / n_rows, 1.0))

        if not cursor:
            break

        time.sleep(sleep_s)

    return pd.DataFrame(rows)

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="OpenAlex Sample Dataset Builder", layout="wide")
st.title("OpenAlex → sample CSV for clustering (Streamlit)")

st.markdown(
    "Build a **shareable demo dataset** (e.g., 5,000 rows) with DOI/Title/Abstract/Journal/Year from OpenAlex. "
    "OpenAlex provides a Works API with filters like `has_doi` and `has_abstract`, and supports cursor pagination. "
    "[1](https://developers.openalex.org/api-reference/works)[2](https://developers.openalex.org/)"
)

with st.sidebar:
    st.header("Settings")

    mode = st.selectbox(
        "Mode",
        ["Broad", "Life science keyword"],
        index=0
    )

    query = ""
    if mode == "Life science keyword":
        query = st.text_input("Keyword (search)", value="cancer")

    n_rows = st.number_input("Rows to collect", min_value=500, max_value=50000, value=5000, step=500)

    col1, col2 = st.columns(2)
    with col1:
        year_from = st.number_input("Year from (optional)", min_value=1900, max_value=2100, value=2015, step=1)
        use_year_from = st.checkbox("Enable year_from", value=False)
    with col2:
        year_to = st.number_input("Year to (optional)", min_value=1900, max_value=2100, value=2020, step=1)
        use_year_to = st.checkbox("Enable year_to", value=False)

    year_from_val = int(year_from) if use_year_from else None
    year_to_val = int(year_to) if use_year_to else None

    st.subheader("API (optional)")
    api_key = st.text_input("OpenAlex API key (optional)", value="", type="password")
    mailto = st.text_input("Contact email (optional)", value="")

    st.subheader("Politeness")
    sleep_s = st.slider("Sleep between requests (seconds)", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

    go = st.button("Fetch dataset", type="primary")

# Session storage
if "df" not in st.session_state:
    st.session_state.df = None

if go:
    progress = st.progress(0.0)
    status = st.empty()

    try:
        with st.spinner("Querying OpenAlex…"):
            df = fetch_openalex_sample(
                n_rows=int(n_rows),
                mode=mode,
                query=query,
                year_from=year_from_val,
                year_to=year_to_val,
                api_key=api_key,
                mailto=mailto,
                sleep_s=float(sleep_s),
                progress_cb=progress.progress,
                status_cb=status.write,
            )
        st.session_state.df = df
        status.success(f"Done. Collected {len(df):,} rows.")
    except Exception as e:
        st.session_state.df = None
        status.error(f"Error: {e}")

df = st.session_state.df

if df is not None and not df.empty:
    st.subheader("Preview")
    st.dataframe(df.head(50), use_container_width=True)

    st.subheader("Quick stats")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Unique DOIs", f"{df['DOI'].nunique():,}")
    c3.metric("Year min", f"{int(df['PublicationYear'].min()) if df['PublicationYear'].notna().any() else '—'}")
    c4.metric("Year max", f"{int(df['PublicationYear'].max()) if df['PublicationYear'].notna().any() else '—'}")

    st.subheader("Download")
    csv_bytes = df_to_csv_bytes(df)
    st.download_button(
        label="Download CSV",
        data=csv_bytes,
        file_name=f"openalex_sample_{len(df)}.csv",
        mime="text/csv"
    )

    st.caption("Columns: DOI, Title, Abstract, Journal, PublicationYear, OpenAlexID, OpenAlexURL")
else:
    st.info("Click **Fetch dataset** to build a dataset. Tip: keep `has_abstract` on (built-in) for clustering text. [1](https://developers.openalex.org/api-reference/works)")