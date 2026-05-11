### Lucie's super stupid experimental app for retrieving OA datasets for experimenting in 
###  https://embedder.streamlit.app/
###  https://stclusteringscout.streamlit.app/
###  https://clusterlab.streamlit.app/ 
###  https://clusterviz.streamlit.app/
############

import time
import json
import requests
import pandas as pd
import streamlit as st
from typing import Optional, Dict, Any, List, Tuple

WORKS_URL = "https://api.openalex.org/works"
SOURCES_URL = "https://api.openalex.org/sources"


# ----------------------------
# Helpers
# ----------------------------


def extract_primary_topic_fields(work: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten work['primary_topic'] into a few useful columns.
    """
    pt = work.get("primary_topic") or {}
    if not isinstance(pt, dict):
        pt = {}

    # These sub-objects are typically nested dictionaries
    domain = (pt.get("domain") or {}) if isinstance(pt.get("domain"), dict) else {}
    field = (pt.get("field") or {}) if isinstance(pt.get("field"), dict) else {}
    subfield = (pt.get("subfield") or {}) if isinstance(pt.get("subfield"), dict) else {}

    return {
        "PrimaryTopic": pt.get("display_name", ""),
        "PrimaryTopicID": (pt.get("id", "") or "").rsplit("/", 1)[-1],  # Txxxx
        "Domain": domain.get("display_name", ""),
        "Field": field.get("display_name", ""),
        "Subfield": subfield.get("display_name", ""),
    }


def topics_to_strings(work: Dict[str, Any], top_n: int = 5) -> Dict[str, Any]:
    """
    Flatten work['topics'] list into CSV-friendly strings.
    Keeps top N topics (by score if present).
    """
    topics = work.get("topics") or []
    if not isinstance(topics, list):
        topics = []

    # Some topics include "score". Sort if present, else keep order.
    def score(t):
        s = t.get("score")
        return s if isinstance(s, (int, float)) else -1

    topics_sorted = sorted(
        [t for t in topics if isinstance(t, dict)],
        key=score,
        reverse=True
    )

    top = topics_sorted[:top_n]
    names = [t.get("display_name", "") for t in top if t.get("display_name")]
    ids = [(t.get("id", "") or "").rsplit("/", 1)[-1] for t in top if t.get("id")]

    return {
        "TopicsTopN": "; ".join(names),
        "TopicIDsTopN": "; ".join(ids),
        "TopicsCount": len(topics),
    }

def add_citations_by_year_columns(df: pd.DataFrame, years: list[int],
                                 src_col: str = "CountsByYear",
                                 drop_src: bool = True,
                                 also_keep_json: bool = False) -> pd.DataFrame:
    """
    Expand df[src_col] (list of dicts like {"year": 2026, "cited_by_count": 5})
    into wide columns for each year in `years`. Missing years -> 0.

    drop_src=True will remove the raw object column to avoid [object Object] display.
    also_keep_json=True keeps a JSON-string version for export/debug.
    """
    if src_col not in df.columns:
        return df

    # Ensure we have a writable copy
    df = df.copy()

    # optional: keep readable version of the raw data
    if also_keep_json:
        df[f"{src_col}_json"] = df[src_col].apply(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, list) else "")

    # start with zeros
    for y in years:
        df[str(y)] = 0

    # fill from CountsByYear
    for i, items in enumerate(df[src_col]):
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            y = item.get("year")
            c = item.get("cited_by_count", 0)
            if y in years:
                df.at[i, str(y)] = c

    if drop_src:
        df = df.drop(columns=[src_col])

    return df



def invert_index_to_text(inv: Optional[Dict[str, List[int]]]) -> str:
    """Reconstruct abstract text from OpenAlex abstract_inverted_index."""
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


def safe_get_journal_from_primary_location(work: Dict[str, Any]) -> str:
    """
    host_venue is deprecated; use primary_location.source.display_name instead.
    """
    pl = work.get("primary_location") or {}
    src = pl.get("source") or {}
    return src.get("display_name") or ""


def build_works_params(
    cursor: str,
    mode: str,
    keyword_query: str,
    year_from: Optional[int],
    year_to: Optional[int],
    api_key: Optional[str],
    mailto: Optional[str],
    per_page: int = 200,
    source_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build query params for /works.

    Notes:
    - Use per_page (underscore) for paging/cursor paging.
    - Use primary_location (host_venue is deprecated).
    - Combine filters with commas = AND.
    """
    filters = ["has_doi:true", "has_abstract:true"]

    if year_from is not None:
        filters.append(f"from_publication_date:{year_from}-01-01")
    if year_to is not None:
        filters.append(f"to_publication_date:{year_to}-12-31")

    if source_id:
        # Filter works to a specific journal/venue source
        filters.append(f"primary_location.source.id:{source_id}")

    params: Dict[str, Any] = {
        "filter": ",".join(filters),
        "per_page": per_page,   # IMPORTANT: underscore
        "cursor": cursor,
        "select": ",".join([
            "id",
            "doi",
            "display_name",
            "abstract_inverted_index",
            
            "primary_topic",
            "topics",

            "publication_year",
            "publication_date",
            "primary_location",
            "type",
            "cited_by_count",
            "counts_by_year",
            "referenced_works_count",
        ]),
    }

    # Optional keyword search (works search across titles/abstracts/etc.)
    if mode == "Life science keyword" and keyword_query.strip() and not source_id:
        params["search"] = keyword_query.strip()

    if api_key and api_key.strip():
        params["api_key"] = api_key.strip()
    if mailto and mailto.strip():
        params["mailto"] = mailto.strip()

    return params


def find_sources_by_name(
    journal_query: str,
    api_key: Optional[str],
    mailto: Optional[str],
    max_results: int = 25,
) -> List[Dict[str, Any]]:
    """
    Search OpenAlex sources (journals/venues) by name string.
    """
    if not journal_query.strip():
        return []

    params: Dict[str, Any] = {
        "search": journal_query.strip(),
        "per_page": min(max_results, 200),
        "select": ",".join([
            "id",
            "display_name",
            "host_organization_name",
            "issn",
            "issn_l",
            "type",
            "works_count",
            "cited_by_count",
        ]),
    }

    if api_key and api_key.strip():
        params["api_key"] = api_key.strip()
    if mailto and mailto.strip():
        params["mailto"] = mailto.strip()

    r = requests.get(SOURCES_URL, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()
    return data.get("results", [])


def fetch_works(
    n_rows: int,
    mode: str,
    keyword_query: str,
    year_from: Optional[int],
    year_to: Optional[int],
    api_key: Optional[str],
    mailto: Optional[str],
    sleep_s: float,
    source_id: Optional[str] = None,
    progress_cb=None,
    status_cb=None,
) -> pd.DataFrame:
    """
    Cursor-page through /works until we collect n_rows.
    """
    session = requests.Session()
    headers = {"User-Agent": "streamlit-openalex-demo-builder/1.0"}

    cursor = "*"
    collected = 0
    page_count = 0
    rows = []

    while collected < n_rows:
        params = build_works_params(
            cursor=cursor,
            mode=mode,
            keyword_query=keyword_query,
            year_from=year_from,
            year_to=year_to,
            api_key=api_key,
            mailto=mailto,
            per_page=200,
            source_id=source_id,
        )

        r = session.get(WORKS_URL, params=params, headers=headers, timeout=60)
        r.raise_for_status()
        data = r.json()

        results = data.get("results", [])
        cursor = (data.get("meta") or {}).get("next_cursor")
        page_count += 1

        if status_cb:
            status_cb(f"Fetched page {page_count} | collected {collected}/{n_rows}")

        if not results:
            break

        for w in results:
            doi = w.get("doi") or ""
            title = w.get("display_name") or ""
            abstract = invert_index_to_text(w.get("abstract_inverted_index"))
            if not (doi and title and abstract):
                continue
            row = {
                "DOI": doi,
                "Title": title,
                "Abstract": abstract,
                "PublicationYear": w.get("publication_year"),
                "PublicationDate": w.get("publication_date"),
                "JournalOrVenue": safe_get_journal_from_primary_location(w),
                "WorkType": w.get("type"),
                "CitedByCount": w.get("cited_by_count"),
                "CountsByYear": w.get("counts_by_year") or [],
                "ReferencedWorksCount": w.get("referenced_works_count"),
                "OpenAlexID": w.get("id"),
                "OpenAlexURL": w.get("id"),
            }

            # ✅ add flattened topic metadata here
            row.update(extract_primary_topic_fields(w))
            row.update(topics_to_strings(w, top_n=5))

            rows.append(row)

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


def source_id_short(openalex_source_id_url: str) -> str:
    """
    Convert 'https://openalex.org/S123' -> 'S123' (what filter expects).
    """
    if not openalex_source_id_url:
        return ""
    return openalex_source_id_url.rsplit("/", 1)[-1]


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="OpenAlex Demo Dataset Builder", layout="wide")
st.title("OpenAlex demo dataset builder")

st.caption(
    "New use case: fetch papers by **Journal name** + **year range** (e.g., Advanced Science 2025–2026). "
    "Implementation: resolve journal name → OpenAlex Source ID → filter works by source.id + date range."
)

with st.sidebar:
    st.header("Inputs")

    # --- Journal mode ---
    st.subheader("Journal filter (new)")
    journal_name = st.text_input('Journal name (e.g., "Advanced Science")', value="")
    find_journals = st.button("Find journals")

    # store found journals
    if "source_candidates" not in st.session_state:
        st.session_state.source_candidates = []

    if find_journals:
        api_key_tmp = st.session_state.get("api_key_tmp", "")
        mailto_tmp = st.session_state.get("mailto_tmp", "")
        try:
            st.session_state.source_candidates = find_sources_by_name(
                journal_query=journal_name,
                api_key=api_key_tmp,
                mailto=mailto_tmp,
                max_results=25,
            )
            if not st.session_state.source_candidates:
                st.warning("No journal matches found.")
        except Exception as e:
            st.session_state.source_candidates = []
            st.error(f"Journal search error: {e}")

    source_candidates = st.session_state.source_candidates

    selected_source_id = None
    if source_candidates:
        options = []
        for s in source_candidates:
            sid = source_id_short(s.get("id", ""))
            name = s.get("display_name", "")
            org = s.get("host_organization_name", "")
            issn_l = s.get("issn_l", "")
            typ = s.get("type", "")
            works_count = s.get("works_count", "")
            options.append((sid, f"{name}  |  {org}  |  ISSN-L: {issn_l}  |  type: {typ}  |  works: {works_count}"))

        chosen = st.selectbox(
            "Select the journal/venue",
            options=options,
            format_func=lambda x: x[1],
        )
        selected_source_id = chosen[0]
        st.success(f"Selected source id: {selected_source_id}")

    include_citations_by_year = st.checkbox("Add citations by year columns", value=False)
    cite_year_from = st.number_input("Cite year from", 1900, 2100, 2017)
    cite_year_to   = st.number_input("Cite year to",   1900, 2100, 2026)

    # --- Other modes still available ---
    st.subheader("Other search modes (optional)")
    mode = st.selectbox("Mode", ["Broad", "Life science keyword"], index=0)
    keyword_query = ""
    if mode == "Life science keyword":
        keyword_query = st.text_input("Keyword (used only if no journal is selected)", value="cancer")

    st.subheader("Year range")
    col1, col2 = st.columns(2)
    with col1:
        year_from_enabled = st.checkbox("Enable year_from", value=True)
        year_from = st.number_input("Year from", min_value=1900, max_value=2100, value=2025, step=1)
    with col2:
        year_to_enabled = st.checkbox("Enable year_to", value=True)
        year_to = st.number_input("Year to", min_value=1900, max_value=2100, value=2026, step=1)

    year_from_val = int(year_from) if year_from_enabled else None
    year_to_val = int(year_to) if year_to_enabled else None

    st.subheader("Sampling")
    n_rows = st.number_input("Rows to collect", min_value=500, max_value=50000, value=5000, step=500)

    st.subheader("API (optional)")
    api_key = st.text_input("OpenAlex API key (optional)", value="", type="password")
    mailto = st.text_input("Contact email (optional)", value="")

    # stash for journal search button
    st.session_state.api_key_tmp = api_key
    st.session_state.mailto_tmp = mailto

    st.subheader("Politeness")
    sleep_s = st.slider("Sleep between requests (seconds)", 0.0, 1.0, 0.2, 0.05)

    go = st.button("Fetch dataset", type="primary")


if "df" not in st.session_state:
    st.session_state.df = None

if go:
    progress = st.progress(0.0)
    status = st.empty()
    try:
        with st.spinner("Querying OpenAlex…"):
            df = fetch_works(
                n_rows=int(n_rows),
                mode=mode,
                keyword_query=keyword_query,
                year_from=year_from_val,
                year_to=year_to_val,
                api_key=api_key,
                mailto=mailto,
                sleep_s=float(sleep_s),
                source_id=selected_source_id,
                progress_cb=progress.progress,
                status_cb=status.write,
            )

            # ✅ ADD HERE (before saving into session state)
            if include_citations_by_year:  # your checkbox/toggle
                years = list(range(cite_year_from, cite_year_to + 1))
                df = add_citations_by_year_columns(df, years)

            #years = list(range(2023, 2027))  # or dynamically from year_from/year_to   ########################### change this to dynamic
            #df = add_citations_by_year_columns(df, years)    
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
    st.download_button(
        "Download CSV",
        data=df_to_csv_bytes(df),
        file_name=f"openalex_sample_{len(df)}.csv",
        mime="text/csv",
    )
else:
    st.info("Use the sidebar to select a journal (optional) and year range, then click **Fetch dataset**.")
