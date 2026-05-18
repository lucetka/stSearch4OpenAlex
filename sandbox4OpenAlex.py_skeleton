### experimental app for retrieving OA datasets for experimenting in 
### this version is dornfelder_SPECTER2 but here with SPECTER turned off
###  fetch_specter2 = st.checkbox("Fetch SPECTER2 embeddings (Semantic Scholar)", value=False, disabled = True, ...
###_calvados with 2 falvors of impact : total/cumulative, and IF window

### Current behavior:
### 👉 Journal overrides keyword (keyword is ignored)
### Actual query:
### 👉 Journal AND year filters only
### Not:
### 👉 Journal AND keyword
###
### turning Search4OpenAlex to enable "dashboarding"

### this is an ultraquick prototype: I'm asking Copilot to help me implement
### some of the same functionality that I have in stClusterViz (which in turn comes from my old Jupyter notebooks),
### Here strating with just OpenAlex primary topic and growth rates

import time
import json
import requests
import pandas as pd
import streamlit as st
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import plotly.express as px
import io
import re


WORKS_URL = "https://api.openalex.org/works"
SOURCES_URL = "https://api.openalex.org/sources"

# Semantic Scholar (S2) Graph API (for SPECTER2 embeddings)
S2_PAPER_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"



# ----------------------------
# Helpers
# ----------------------------


def compute_topic_growth_table(
    df: pd.DataFrame,
    topic_col: str = "PrimaryTopic",
    year_col: str = "PublicationYear",
    start_year: int = 2025,
    end_year: int = 2026,
    smoothing: float = 1.0,
    min_total: int = 5,
) -> pd.DataFrame:
    """
    Returns one row per topic with:
      Topic, N_total, N_start, N_end, Delta, CAGR, SlopePerYear
    """
    if df is None or df.empty or topic_col not in df.columns or year_col not in df.columns:
        return pd.DataFrame(columns=["Topic", "N_total", "N_start", "N_end", "Delta", "CAGR", "SlopePerYear"])

    d = df[[topic_col, year_col]].copy()
    d[year_col] = pd.to_numeric(d[year_col], errors="coerce").astype("Int64")
    d = d.dropna(subset=[topic_col, year_col])
    d[topic_col] = d[topic_col].astype(str)

    # total counts per topic (in currently retrieved dataset)
    totals = d.groupby(topic_col).size().rename("N_total")

    # counts in start/end year
    c_start = d[d[year_col] == start_year].groupby(topic_col).size().rename("N_start")
    c_end = d[d[year_col] == end_year].groupby(topic_col).size().rename("N_end")

    out = pd.concat([totals, c_start, c_end], axis=1).fillna(0)
    out["N_start"] = out["N_start"].astype(int)
    out["N_end"] = out["N_end"].astype(int)
    out["N_total"] = out["N_total"].astype(int)

    # optionally filter tiny topics
    out = out[out["N_total"] >= int(min_total)].copy()

    years = max(int(end_year) - int(start_year), 1)
    out["Delta"] = out["N_end"] - out["N_start"]

    # CAGR with smoothing (prevents div-by-zero, avoids infinite growth)
    out["CAGR"] = ((out["N_end"] + smoothing) / (out["N_start"] + smoothing)) ** (1 / years) - 1

    # simple slope per year
    out["SlopePerYear"] = out["Delta"] / years

    out = out.reset_index().rename(columns={topic_col: "Topic"})
    return out

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



# ----------------------------
# Semantic Scholar: fetch SPECTER2 embeddings (proof-of-principle)
# ----------------------------
def _doi_to_s2_id(doi_url_or_raw: str) -> str:
    """Convert DOI URL (https://doi.org/...) or raw DOI into Semantic Scholar ID format DOI:..."""
    if not doi_url_or_raw:
        return ''
    s = str(doi_url_or_raw).strip()
    s = re.sub(r'^https?://doi\.org/', '', s, flags=re.IGNORECASE)
    s = re.sub(r'^doi:', '', s, flags=re.IGNORECASE)
    return f'DOI:{s}' if s else ''

def s2_fetch_specter2_embeddings(s2_ids: list[str], s2_api_key: str = '', sleep_s: float = 0.35) -> dict:
    """Fetch SPECTER2 embeddings from Semantic Scholar Graph API /paper/batch.

    Uses fields=paperId,corpusId,embedding.specter_v2 (SPECTER2).
    Returns dict mapping input s2_id -> dict with keys: paperId, corpusId, specter2 (vector list).
"""
    out = {}
    if not s2_ids:
        return out
    headers = {}
    if s2_api_key:
        headers['x-api-key'] = s2_api_key
    fields = 'paperId,corpusId,embedding' #embedding.specter_v2'
    BATCH = 500
    for i in range(0, len(s2_ids), BATCH):
        chunk = s2_ids[i:i+BATCH]
        r = requests.post(
            S2_PAPER_BATCH_URL,
            params={'fields': fields},
            json={'ids': chunk},
            headers=headers,
            timeout=60,
        )
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list):
            for req_id, item in zip(chunk, data):
                if not isinstance(item, dict) or 'error' in item:
                    continue
                emb_obj =  item.get('embedding') #emb = (item.get('embedding') or {}).get('specter_v2')

                 # TEMP DEBUG: show one example structure
                # (remove after confirmed)
                print("embedding keys:", emb_obj.keys() if isinstance(emb_obj, dict) else type(emb_obj))
                
                emb = None
                if isinstance(emb_obj, dict):
                    # try likely keys
                    for k in ("specter_v2", "specter2", "vector"):
                        if k in emb_obj:
                            emb = emb_obj.get(k)
                            break
                elif isinstance(emb_obj, list):
                    # sometimes the embedding is directly the vector
                    emb = emb_obj
                
               
                if emb is None:
                    continue
                out[req_id] = {
                    'paperId': item.get('paperId'),
                    'corpusId': item.get('corpusId'),
                    'specter2': emb,
                }
        time.sleep(sleep_s)
    return out

def attach_specter2_to_df(df: pd.DataFrame, s2_api_key: str = '', max_rows: int = 200, sleep_s: float = 0.35) -> pd.DataFrame:
    """Attach SPECTER2 embeddings to a DataFrame with a DOI column.

    Adds columns: S2_paperId, S2_corpusId, SPECTER2 (JSON string), SPECTER2_dim
"""
    if df is None or df.empty or 'DOI' not in df.columns:
        return df
    d = df.copy()
    n = min(int(max_rows), len(d))
    s2_ids = [_doi_to_s2_id(x) for x in d.loc[:n-1, 'DOI'].tolist()]
    s2_ids = [x for x in s2_ids if x]
    if not s2_ids:
        return d
    emb_map = s2_fetch_specter2_embeddings(s2_ids, s2_api_key=s2_api_key, sleep_s=sleep_s)
    d['S2_paperId'] = pd.NA
    d['S2_corpusId'] = pd.NA
    d['SPECTER2'] = pd.NA
    d['SPECTER2_dim'] = pd.NA
    for idx in range(n):
        req_id = _doi_to_s2_id(d.at[idx, 'DOI'])
        item = emb_map.get(req_id)
        if not item:
            continue
        vec = item.get('specter2')
        d.at[idx, 'S2_paperId'] = item.get('paperId')
        d.at[idx, 'S2_corpusId'] = item.get('corpusId')
        d.at[idx, 'SPECTER2'] = json.dumps(vec)
        d.at[idx, 'SPECTER2_dim'] = len(vec) if isinstance(vec, list) else pd.NA
    return d

## ----------------------------
## Semantic Scholar: fetch SPECTER2 embeddings (proof-of-principle)
## ----------------------------

def _doi_to_s2_id(doi_url_or_raw: str) -> str:
    """Convert DOI URL (https://doi.org/...) or raw DOI into S2 ID format DOI:..."""
    if not doi_url_or_raw:
        return ""
    s = str(doi_url_or_raw).strip()
    s = re.sub(r"^https?://doi\.org/", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^doi:", "", s, flags=re.IGNORECASE)
    return f"DOI:{s}" if s else ""


def _extract_specter2_from_embedding_obj(emb_obj):
    """Robust extraction of a vector from various embedding payload shapes."""
    if emb_obj is None:
        return None
    if isinstance(emb_obj, list):
        return emb_obj
    if isinstance(emb_obj, dict):
        for k in ("specter_v2", "specter2", "specter_2", "vector"):
            v = emb_obj.get(k)
            if isinstance(v, list):
                return v
    return None


def s2_fetch_specter2_embeddings(s2_ids: list[str], s2_api_key: str = "", sleep_s: float = 0.35) -> dict:
    """Fetch SPECTER2 embeddings from Semantic Scholar Graph API /paper/batch.

    Returns mapping: requested_id -> {paperId, corpusId, specter2}
    """
    out: dict = {}
    if not s2_ids:
        return out

    headers = {}
    if s2_api_key:
        headers["x-api-key"] = s2_api_key

    # Request full embedding object and extract SPECTER2 vector robustly.
    fields = "paperId,corpusId,embedding"

    BATCH = 500
    for i in range(0, len(s2_ids), BATCH):
        chunk = s2_ids[i:i + BATCH]
        r = requests.post(
            S2_PAPER_BATCH_URL,
            params={"fields": fields},
            json={"ids": chunk},
            headers=headers,
            timeout=60,
        )
        r.raise_for_status()
        data = r.json()

        if isinstance(data, list):
            for req_id, item in zip(chunk, data):
                if not isinstance(item, dict) or item.get("error"):
                    continue
                emb_obj = item.get("embedding")
                vec = _extract_specter2_from_embedding_obj(emb_obj)
                if vec is None:
                    continue
                out[req_id] = {
                    "paperId": item.get("paperId"),
                    "corpusId": item.get("corpusId"),
                    "specter2": vec,
                }

        time.sleep(sleep_s)

    return out


def attach_specter2_to_df(df: pd.DataFrame, s2_api_key: str = "", max_rows: int = 200, sleep_s: float = 0.35) -> pd.DataFrame:
    """Attach SPECTER2 embeddings to DataFrame (proof-of-principle).

    Adds columns (always created):
      - S2_paperId, S2_corpusId, SPECTER2, SPECTER2_dim

    Only the first max_rows are queried to keep it fast.
    """
    if df is None or df.empty:
        return df

    d = df.copy()

    # Always create columns so they remain visible even if fetching fails.
    for col in ("S2_paperId", "S2_corpusId", "SPECTER2", "SPECTER2_dim"):
        if col not in d.columns:
            d[col] = pd.NA

    if "DOI" not in d.columns:
        return d

    n = min(int(max_rows), len(d))
    s2_ids = [_doi_to_s2_id(x) for x in d.loc[: n - 1, "DOI"].tolist()]
    s2_ids = [x for x in s2_ids if x]
    if not s2_ids:
        return d

    emb_map = s2_fetch_specter2_embeddings(s2_ids, s2_api_key=s2_api_key, sleep_s=sleep_s)

    for idx in range(n):
        req_id = _doi_to_s2_id(d.at[idx, "DOI"])
        item = emb_map.get(req_id)
        if not item:
            continue
        vec = item.get("specter2")
        d.at[idx, "S2_paperId"] = item.get("paperId")
        d.at[idx, "S2_corpusId"] = item.get("corpusId")
        d.at[idx, "SPECTER2"] = json.dumps(vec)
        d.at[idx, "SPECTER2_dim"] = len(vec) if isinstance(vec, list) else pd.NA

    return d

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
    require_abstract: bool = True,
    include_xpac: bool = True,
    require_doi: bool = True,
) -> Dict[str, Any]:
    """
    Build query params for /works.

    Notes:
    - Use per_page (underscore) for paging/cursor paging.
    - Use primary_location (host_venue is deprecated).
    - Combine filters with commas = AND.
    """
    #filters = ["has_doi:true", "has_abstract:true"]

    #filters = ["has_doi:true"]
    #if require_abstract:
    #    filters.append("has_abstract:true")
    filters = []
    if require_doi:
        filters.append("has_doi:true")
    if require_abstract:
        filters.append("has_abstract:true")

    #if year_from is not None:
    #    filters.append(f"from_publication_date:{year_from}-01-01")
    #if year_to is not None:
    #    filters.append(f"to_publication_date:{year_to}-12-31")

    if year_from is not None and year_to is not None:
        filters.append(f"publication_year:{year_from}-{year_to}")
    elif year_from is not None:
        filters.append(f"publication_year:{year_from}-9999")
    elif year_to is not None:
        filters.append(f"publication_year:0-{year_to}")

    ####


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
            "ids",
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

    if include_xpac:
        params["include_xpac"] = "true"

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
    require_abstract: bool = True,
    include_xpac: bool = True,
    progress_cb=None,
    status_cb=None,
    fetch_specter2: bool = False,
    s2_api_key: str = "",
    max_embed_rows: int = 200,
    s2_sleep_s: float = 0.35,

    require_doi: bool = True,) -> pd.DataFrame:
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
            require_abstract=require_abstract,
            include_xpac=include_xpac,
            require_doi=require_doi,
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
            #abstract = invert_index_to_text(w.get("abstract_inverted_index"))
            abstract = invert_index_to_text(w.get("abstract_inverted_index"))

            # Always require DOI + title, but make abstract optional via toggle
            if not (doi and title):
                continue

            if require_abstract and not abstract:
                continue

            if not (doi and title and abstract):
                continue
            row = {
                "DOI": doi,
                "Title": title,
                #"Abstract": abstract,
                "Abstract": abstract or "",   # ✅ keep empty string 
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

    df_out = pd.DataFrame(rows)

    if fetch_specter2:
        try:
            df_out = attach_specter2_to_df(df_out, s2_api_key=s2_api_key, max_rows=max_embed_rows, sleep_s=s2_sleep_s)
        except Exception as e:
            if status_cb:
                status_cb(f"SPECTER2 embedding fetch failed: {e}")

    return df_out


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
    #"New use case: fetch papers by **Journal name** + **year range** (e.g., Advanced Science 2025–2026). "
    #"Implementation: resolve journal name → OpenAlex Source ID → filter works by source.id + date range."
    "Searching and dashboarding with OpenAlex sandbox"
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

    
    require_doi = st.checkbox(
        "Require DOI (exclude items without DOI)",
        value=True,
        help="If unchecked, include records without DOIs (may increase counts)."
    )


    include_xpac = st.checkbox(
        "Include xpac (expansion pack) works",
        value=True,
        help="Better coverage, but data quality is lower."
    )

    require_abstract = st.checkbox(
        "Require abstract (exclude items without abstract)",
        value=True,
        help="Turn off to include records even when OpenAlex has no abstract for them."
    )       


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
    n_rows = st.number_input("Rows to collect", min_value=100, max_value=50000, value=5000, step=100)

    st.subheader("API (optional)")
    api_key = st.text_input("OpenAlex API key (optional)", value="", type="password")
    mailto = st.text_input("Contact email (optional)", value="")

    st.subheader("Embeddings (optional)")
    fetch_specter2 = st.checkbox("Fetch SPECTER2 embeddings (Semantic Scholar)", value=False, disabled = True,
        help="Proof-of-principle: fetch SPECTER2 embeddings via Semantic Scholar Graph API for the first N rows.")
    s2_api_key = st.text_input("Semantic Scholar API key (optional)", value="", type="password")
    max_embed_rows = st.number_input("Max rows to embed", min_value=10, max_value=5000, value=200, step=50)
    s2_sleep_s = st.slider("Sleep between S2 batches (s)", min_value=0.0, max_value=2.0, value=0.35, step=0.05)

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
                require_abstract=require_abstract,
                include_xpac=include_xpac,
                require_doi=require_doi,
                fetch_specter2=fetch_specter2,
                s2_api_key=s2_api_key,
                max_embed_rows=int(max_embed_rows),
                s2_sleep_s=float(s2_sleep_s),
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




import plotly.express as px
import pandas as pd

df = st.session_state.df

if df is None or df.empty:
    st.info("Fetch a dataset first.")
    st.stop()

tab_preview, tab_treemap, tab_impact, tab_download = st.tabs([
    "📄 Preview",
    "🟩 Treemap (Topics growth)",
    "🟧 Treemap (Impact)",
    "⬇ Download"
])


# ----------------------------
# Tab 1: Preview
# ----------------------------
with tab_preview:
    st.subheader("Preview")
    st.dataframe(df.head(50), use_container_width=True)

    if "Abstract" in df.columns:
        st.caption(f"Abstract coverage: {(df['Abstract'].astype(str).str.len() > 0).mean():.1%}")

    st.subheader("Quick stats")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Unique DOIs", f"{df['DOI'].nunique():,}" if "DOI" in df.columns else "—")

    if "PublicationYear" in df.columns:
        years = pd.to_numeric(df["PublicationYear"], errors="coerce")
        c3.metric("Year min", f"{int(years.min())}" if years.notna().any() else "—")
        c4.metric("Year max", f"{int(years.max())}" if years.notna().any() else "—")
    else:
        c3.metric("Year min", "—")
        c4.metric("Year max", "—")


# ----------------------------
# Tab 2: Treemap (PrimaryTopic)
# ----------------------------
with tab_treemap:
    st.subheader("Treemap by Primary Topic")
    st.caption("Size = number of papers in dataset. Color = growth over selected years.")

    # Basic validation
    if "PrimaryTopic" not in df.columns or "PublicationYear" not in df.columns:
        st.warning("Need columns 'PrimaryTopic' and 'PublicationYear' in df to build this treemap.")
        st.stop()

    # pick available year bounds from the data
    year_series = pd.to_numeric(df["PublicationYear"], errors="coerce")
    if not year_series.notna().any():
        st.warning("PublicationYear has no valid numeric values.")
        st.stop()

    y_min = int(year_series.min())
    y_max = int(year_series.max())

    colA, colB, colC = st.columns([1, 1, 1])
    with colA:
        start_year = st.number_input(
            "Growth start year",
            min_value=1900, max_value=2100,
            value=max(y_min, y_max - 1),
            step=1
        )
    with colB:
        end_year = st.number_input(
            "Growth end year",
            min_value=1900, max_value=2100,
            value=y_max,
            step=1
        )
    with colC:
        metric = st.selectbox("Color metric", ["CAGR", "SlopePerYear"], index=0)

    colD, colE = st.columns([1, 1])
    with colD:
        smoothing = st.slider(
            "CAGR smoothing",
            min_value=0.0, max_value=5.0,
            value=1.0, step=0.5,
            help="Avoids extreme growth when start-year count is 0."
        )
    with colE:
        min_total = st.number_input(
            "Minimum topic size (N_total)",
            min_value=1, max_value=5000,
            value=5, step=1
        )

    # compute growth table (topic-level)
    topic_tbl = compute_topic_growth_table(
        df=df,
        topic_col="PrimaryTopic",
        year_col="PublicationYear",
        start_year=int(start_year),
        end_year=int(end_year),
        smoothing=float(smoothing),
        min_total=int(min_total),
    )

    if topic_tbl.empty:
        st.info("No topics available after filtering. Try lowering minimum size or adjusting years.")
    else:
        fig = px.treemap(
            topic_tbl,
            path=["Topic"],        # single-level treemap
            values="N_total",      # size
            color=metric,          # growth metric
            hover_data={
                "N_total": True,
                "N_start": True,
                "N_end": True,
                "Delta": True,
                "CAGR": ":.2%",
                "SlopePerYear": ":.2f",
            },
            color_continuous_scale="RdBu_r",
        )
        fig.update_layout(coloraxis_cmid=0)
        fig.update_layout(margin=dict(t=30, l=5, r=5, b=5))
        st.plotly_chart(fig, use_container_width=True)

        
        html_buf = io.StringIO()
        fig.write_html(html_buf, include_plotlyjs="cdn", full_html=True)

        st.download_button(
            label="⬇ Download treemap (HTML)",
            data=html_buf.getvalue().encode("utf-8"),
            file_name="treemap_topics.html",
            mime="text/html",
        )


        with st.expander("Show topic growth table"):
            st.dataframe(topic_tbl.sort_values(metric, ascending=False), use_container_width=True)

# ----------------------------
# Tab: Impact Treemap
# ----------------------------

with tab_impact:
    st.subheader("Treemap by Primary Topic (Impact)")
    st.caption("Size = number of papers. Color = impact (citations per paper).")

    # --- Controls ---
    col1, col2, col3 = st.columns(3)

    with col1:
        impact_mode = st.selectbox(
            "Impact type",
            ["All-time", "IF-window (year X to X-1/X-2)"]
        )

    with col2:
        agg_mode = st.selectbox(
            "Aggregation",
            ["Mean", "Median"]
        )

    with col3:
        min_topic_size = st.number_input(
            "Min papers per topic",
            min_value=1,
            value=5
        )
    
    # detect year columns dynamically ############################################### !!!!!!!!!!!!!!!! This may need to be adjusted to make sure digits are actually years and not something else!!!!!!!!!!!!
    year_cols = [c for c in df.columns if c.isdigit()]
    year_cols = sorted(year_cols)

    if impact_mode == "IF-window (year X to X-1/X-2)":
        if not year_cols:
            st.warning("No citation-by-year columns found in dataset.")
            st.stop()

        col_year = st.selectbox(
            "Evaluation year X",
            year_cols
        )

        # optional warning for latest year
        if col_year == year_cols[-1]:
            st.info("Note: selected year may be incomplete.")

    # --- Build impact table ---
    required_cols = ["PrimaryTopic", "CitedByCount"]

    if not all(c in df.columns for c in required_cols):
        st.warning("Dataset missing required columns for impact calculation.")
        st.stop()

    # drop missing topics
    df_imp = df.dropna(subset=["PrimaryTopic"]).copy()

    grouped = df_imp.groupby("PrimaryTopic")

    # paper counts
    topic_counts = grouped.size().rename("N")

    # aggregation
    # ========================
    # IMPACT CALCULATION
    # ========================

    if impact_mode == "All-time":
        if agg_mode == "Mean":
            topic_impact = grouped["CitedByCount"].mean()
        else:
            topic_impact = grouped["CitedByCount"].median()

    else:
        # IF-WINDOW
        X = int(col_year)

        # required columns
        if col_year not in df_imp.columns:
            st.warning(f"Selected year {X} not available.")
            st.stop()

        if "PublicationYear" not in df_imp.columns:
            st.warning("PublicationYear column required for IF-window.")
            st.stop()

        # filter denominator: papers from X-1 and X-2
        df_window = df_imp[
            df_imp["PublicationYear"].isin([X - 1, X - 2])
        ].copy()

        if df_window.empty:
            st.warning("No papers in publication years X-1 or X-2.")
            st.stop()

        # get citations in year X
        df_window["C_X"] = df_window[col_year].fillna(0)

        grouped = df_window.groupby("PrimaryTopic")

        topic_counts = grouped.size().rename("N")

        if agg_mode == "Mean":
            topic_impact = grouped["C_X"].mean()
        else:
            topic_impact = grouped["C_X"].median()


    impact_table = pd.concat([topic_counts, topic_impact.rename("Impact")], axis=1).reset_index()

    # flag small topics
    impact_table["IsSmall"] = impact_table["N"] < min_topic_size

    # --- Visualization ---
    # Color: use grey for small topics
    impact_table["ColorVal"] = impact_table["Impact"]

    fig = px.treemap(
        impact_table,
        path=["PrimaryTopic"],
        values="N",
        color="ColorVal",
        color_continuous_scale="RdBu_r",
    )

    # --- Manually override small nodes to grey ---
    # (simple hack via color axis range clipping)
    impact_table.loc[impact_table["IsSmall"], "ColorVal"] = None

    fig = px.treemap(
        impact_table,
        path=["PrimaryTopic"],
        values="N",
        color="ColorVal",
        color_continuous_scale="RdBu_r",
    )

    fig.update_traces(marker=dict(line=dict(width=0.5)))

    st.plotly_chart(fig, use_container_width=True)

        # --- Download impact treemap (interactive HTML) ---
    html = fig.to_html(include_plotlyjs="cdn", full_html=True)

    st.download_button(
        label="⬇ Download impact treemap (HTML, interactive)",
        data=html.encode("utf-8"),
        file_name="impact_treemap.html",
        mime="text/html",
    )

    # --- Debug / transparency (optional but useful)
    with st.expander("Show impact table"):
        st.dataframe(impact_table.sort_values("Impact", ascending=False))

# ----------------------------
# Tab 3: Download
# ----------------------------
with tab_download:
    st.subheader("Download")
    st.download_button(
        "Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=f"openalex_sample_{len(df)}.csv",
        mime="text/csv",
    )
