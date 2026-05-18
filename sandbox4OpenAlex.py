# This is _espresso_macchiato.py version

# History : I had a version mess. This version is _dornfelder_SPECTER2 from 5pm 13-May-26 "merged" into stSearchDashOpenAlex_espresso_patched_v2.py
# saved earlier that day but actually more "advanced" - by mistake the whole SPECTER2 code was chucked instead of just GUI disabled
# so now it was added back into "espresso". Also further missing parts were added back
# 
# This version was "merged" by Copilot from the two version and was named stSearchDashOpenAlex_espresso_merged_with_SPECTER2_disabled_plus_TopicID.py
# I copied and renamed this to _espresso_macchiato.py and continued adding chatGPT functionality

# # in the meantime I had given Copilot the _dornfelder_SPECTER2 version to start implementing chatGPT based on my old (2024) Jupyter notebooks working with NatComm
## so that branch is a "cul-de-sac" and I'm going to re-do this (better hopefully, becuase the implementaion in the _dornfelder_specter2 was anyways a bit dumb

# Goals (What Copilot remembers per Lucie):
# Known issue: some journals such as BBA series or Analytical Biochemistry retrieve dramatically fewer records than from WoS or the OpenAlex web interface; under investigation
# Known issue: sometimes Meeting Abstracts are wrongly classified as Articles --> here on the other hand results are blown up with "wrong" results; for now,
# I recommend checking the expected number of records and cleaning downloaded results in excel if necessary. The Meeting Abstracts all started with the word Abstract in the title,
# so for my dataset from J Biological Chemistry I was able to simply chuck them in Excel but filtering for the word Abstract in the title et voila the number of records matches WoS

# SPECTER2 toggle is present but DISABLED for now (deployment safety).

import time
import json
import re
import io
from typing import Optional, Dict, Any, List

import pandas as pd
import requests
import streamlit as st
import plotly.express as px

WORKS_URL = "https://api.openalex.org/works"
SOURCES_URL = "https://api.openalex.org/sources"
SUBFIELDS_URL = "https://api.openalex.org/subfields"

# Semantic Scholar (S2) Graph API (for SPECTER2 embeddings) — code retained but UI is disabled by default
S2_PAPER_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"


# ----------------------------
# Cached taxonomy: Subfields
# ----------------------------
@st.cache_data(show_spinner=False, ttl=24 * 3600)
def fetch_all_subfields(api_key: str = "", mailto: str = "") -> pd.DataFrame:
    """Fetch all OpenAlex subfields (3rd level taxonomy) into a DataFrame.

    Cached for 24h to avoid repeated calls.
    """
    session = requests.Session()
    rows: List[Dict[str, Any]] = []

    cursor = "*"
    while True:
        params: Dict[str, Any] = {
            "per_page": 200,
            "cursor": cursor,
            "select": ",".join(["id", "display_name", "field", "domain"]),
        }
        if api_key:
            params["api_key"] = api_key
        if mailto:
            params["mailto"] = mailto

        r = session.get(SUBFIELDS_URL, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()
        results = data.get("results", [])
        if not results:
            break

        for s in results:
            sid_url = s.get("id") or ""
            sid = sid_url.rsplit("/", 1)[-1] if sid_url else ""
            field = s.get("field") or {}
            domain = s.get("domain") or {}
            rows.append({
                "subfield_id": sid,
                "subfield_id_url": sid_url,
                "subfield": s.get("display_name") or "",
                "field": field.get("display_name") or "",
                "domain": domain.get("display_name") or "",
            })

        cursor = (data.get("meta") or {}).get("next_cursor")
        if not cursor:
            break

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["label"] = df.apply(lambda r: f"{r['subfield']}  —  {r['field']} / {r['domain']}".strip(), axis=1)
    df = df.sort_values(["domain", "field", "subfield"], kind="stable").reset_index(drop=True)
    return df


# ----------------------------
# Helpers
# ----------------------------

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
    pl = work.get("primary_location") or {}
    src = pl.get("source") or {}

# --- Multi-topic helper (CSV-friendly) ---
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


# ----------------------------
# Semantic Scholar: SPECTER2 embeddings (retained for future work)
# NOTE: The UI toggle is disabled (greyed out) to prevent accidental use.
# ----------------------------
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


    return src.get("display_name") or ""


def source_id_short(openalex_source_id_url: str) -> str:
    if not openalex_source_id_url:
        return ""
    return openalex_source_id_url.rsplit("/", 1)[-1]


def find_sources_by_name(journal_query: str, api_key: str = "", mailto: str = "", max_results: int = 25) -> List[Dict[str, Any]]:
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
        ])
    }
    if api_key:
        params["api_key"] = api_key
    if mailto:
        params["mailto"] = mailto

    r = requests.get(SOURCES_URL, params=params, timeout=60)
    r.raise_for_status()
    return r.json().get("results", [])


def add_citations_by_year_columns(df: pd.DataFrame, years: List[int], src_col: str = "CountsByYear", drop_src: bool = True) -> pd.DataFrame:
    """Expand CountsByYear list (list of dicts) into wide year columns."""
    if df is None or df.empty or src_col not in df.columns:
        return df

    out = df.copy()
    for y in years:
        out[str(y)] = 0

    for i, items in enumerate(out[src_col]):
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            y = item.get("year")
            c = item.get("cited_by_count", 0)
            if y in years:
                out.at[i, str(y)] = c

    if drop_src:
        out = out.drop(columns=[src_col])

    return out


def compute_topic_growth_table(
    df: pd.DataFrame,
    topic_col: str = "PrimaryTopic",
    year_col: str = "PublicationYear",
    start_year: int = 2023,
    end_year: int = 2025,
    smoothing: float = 1.0,
    min_total: int = 5,
) -> pd.DataFrame:
    """Aggregate paper counts per topic and compute growth metrics."""
    if df is None or df.empty or topic_col not in df.columns or year_col not in df.columns:
        return pd.DataFrame(columns=["Topic", "N_total", "N_start", "N_end", "Delta", "CAGR", "SlopePerYear"]) 

    d = df[[topic_col, year_col]].copy()
    d[year_col] = pd.to_numeric(d[year_col], errors="coerce").astype("Int64")
    d = d.dropna(subset=[topic_col, year_col])
    d[topic_col] = d[topic_col].astype(str)

    totals = d.groupby(topic_col).size().rename("N_total")
    c_start = d[d[year_col] == int(start_year)].groupby(topic_col).size().rename("N_start")
    c_end = d[d[year_col] == int(end_year)].groupby(topic_col).size().rename("N_end")

    out = pd.concat([totals, c_start, c_end], axis=1).fillna(0)
    out["N_total"] = out["N_total"].astype(int)
    out["N_start"] = out["N_start"].astype(int)
    out["N_end"] = out["N_end"].astype(int)

    out = out[out["N_total"] >= int(min_total)].copy()

    years = max(int(end_year) - int(start_year), 1)
    out["Delta"] = out["N_end"] - out["N_start"]
    out["CAGR"] = ((out["N_end"] + float(smoothing)) / (out["N_start"] + float(smoothing))) ** (1 / years) - 1
    out["SlopePerYear"] = out["Delta"] / years

    out = out.reset_index().rename(columns={topic_col: "Topic"})
    return out


def compute_topic_impact_table_alltime(df: pd.DataFrame, agg: str = "Mean", min_n: int = 5) -> pd.DataFrame:
    """Impact A: all-time citations per paper by topic (mean/median)."""
    if df is None or df.empty or "PrimaryTopic" not in df.columns or "CitedByCount" not in df.columns:
        return pd.DataFrame(columns=["Topic", "N", "Impact", "IsSmall"]) 

    d = df[["PrimaryTopic", "CitedByCount"]].copy()
    d["PrimaryTopic"] = d["PrimaryTopic"].fillna("(Unknown)").astype(str)
    d["CitedByCount"] = pd.to_numeric(d["CitedByCount"], errors="coerce")

    g = d.groupby("PrimaryTopic")
    n = g.size().rename("N")
    if agg == "Median":
        imp = g["CitedByCount"].median()
    else:
        imp = g["CitedByCount"].mean()

    out = pd.concat([n, imp.rename("Impact")], axis=1).reset_index().rename(columns={"PrimaryTopic": "Topic"})
    out["IsSmall"] = out["N"] < int(min_n)
    return out


def compute_topic_impact_table_ifwindow(df: pd.DataFrame, X: int, agg: str = "Mean", min_n: int = 5) -> pd.DataFrame:
    """Impact B: citations in year X to papers published in X-1 and X-2 (mean/median per paper), by topic.

    Requires year columns as strings (e.g., '2025') OR CountsByYear already expanded.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["Topic", "N", "Impact", "IsSmall"]) 

    if "PrimaryTopic" not in df.columns or "PublicationYear" not in df.columns:
        return pd.DataFrame(columns=["Topic", "N", "Impact", "IsSmall"]) 

    colX = str(int(X))
    if colX not in df.columns:
        # cannot compute
        return pd.DataFrame(columns=["Topic", "N", "Impact", "IsSmall"]) 

    d = df[["PrimaryTopic", "PublicationYear", colX]].copy()
    d["PrimaryTopic"] = d["PrimaryTopic"].fillna("(Unknown)").astype(str)
    d["PublicationYear"] = pd.to_numeric(d["PublicationYear"], errors="coerce").astype("Int64")
    d[colX] = pd.to_numeric(d[colX], errors="coerce").fillna(0)

    d = d[d["PublicationYear"].isin([int(X) - 1, int(X) - 2])].copy()
    if d.empty:
        return pd.DataFrame(columns=["Topic", "N", "Impact", "IsSmall"]) 

    g = d.groupby("PrimaryTopic")
    n = g.size().rename("N")
    if agg == "Median":
        imp = g[colX].median()
    else:
        imp = g[colX].mean()

    out = pd.concat([n, imp.rename("Impact")], axis=1).reset_index().rename(columns={"PrimaryTopic": "Topic"})
    out["IsSmall"] = out["N"] < int(min_n)
    return out


# ----------------------------
# OpenAlex fetch
# ----------------------------

def build_works_params(
    cursor: str,
    keyword_query: str,
    year_from: Optional[int],
    year_to: Optional[int],
    api_key: str,
    mailto: str,
    per_page: int,
    source_ids: Optional[List[str]] = None,
    subfield_id: Optional[str] = None,
    require_abstract: bool = True,
    require_doi: bool = True,
    include_xpac: bool = True,
) -> Dict[str, Any]:
    filters: List[str] = []
    if require_doi:
        filters.append("has_doi:true")
    if require_abstract:
        filters.append("has_abstract:true")

    # publication_year filtering (GUI-consistent)
    if year_from is not None and year_to is not None:
        filters.append(f"publication_year:{int(year_from)}-{int(year_to)}")
    elif year_from is not None:
        filters.append(f"publication_year:{int(year_from)}-9999")
    elif year_to is not None:
        filters.append(f"publication_year:0-{int(year_to)}")

    if source_ids:
        sources_val = "|".join([sid for sid in source_ids if sid])
        if sources_val:
            filters.append(f"primary_location.source.id:{sources_val}")

    if subfield_id:
        filters.append(f"primary_topic.subfield.id:{subfield_id}")

    params: Dict[str, Any] = {
        "filter": ",".join(filters),
        "per_page": per_page,
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

    if include_xpac:
        params["include_xpac"] = "true"

    if keyword_query.strip():
        params["search"] = keyword_query.strip()

    if api_key:
        params["api_key"] = api_key
    if mailto:
        params["mailto"] = mailto

    return params


def fetch_works(
    n_rows: int,
    keyword_query: str,
    year_from: Optional[int],
    year_to: Optional[int],
    api_key: str,
    mailto: str,
    sleep_s: float,
    source_ids: Optional[List[str]] = None,
    subfield_id: Optional[str] = None,
    require_abstract: bool = True,
    require_doi: bool = True,
    include_xpac: bool = True,
    fetch_specter2: bool = False,
    s2_api_key: str = "",
    max_embed_rows: int = 200,
    s2_sleep_s: float = 0.35,
    progress_cb=None,
    status_cb=None,
) -> pd.DataFrame:
    session = requests.Session()
    headers = {"User-Agent": "streamlit-openalex-search/espresso-v2"}

    cursor = "*"
    collected = 0
    page_count = 0
    rows: List[Dict[str, Any]] = []

    while collected < n_rows:
        params = build_works_params(
            cursor=cursor,
            keyword_query=keyword_query,
            year_from=year_from,
            year_to=year_to,
            api_key=api_key,
            mailto=mailto,
            per_page=200,
            source_ids=source_ids,
            subfield_id=subfield_id,
            require_abstract=require_abstract,
            require_doi=require_doi,
            include_xpac=include_xpac,
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

            if not title:
                continue
            if require_doi and not doi:
                continue
            if require_abstract and not abstract:
                continue

            row = {
                "DOI": doi,
                "Title": title,
                "Abstract": abstract or "",
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

            pt = w.get("primary_topic") or {}
            if isinstance(pt, dict):
                row["PrimaryTopic"] = pt.get("display_name", "")
                row["PrimaryTopicID"] = (pt.get("id", "") or "").rsplit("/", 1)[-1]  # Txxxx
                sf = pt.get("subfield") or {}
                fld = pt.get("field") or {}
                dom = pt.get("domain") or {}
                row["Subfield"] = sf.get("display_name", "") if isinstance(sf, dict) else ""
                row["Field"] = fld.get("display_name", "") if isinstance(fld, dict) else ""
                row["Domain"] = dom.get("display_name", "") if isinstance(dom, dict) else ""
            else:
                row["PrimaryTopic"] = ""
                row["Subfield"] = ""
                row["Field"] = ""
                row["Domain"] = ""


            # Add Top-N topic tags (from OpenAlex work["topics"]) for CSV-friendly export
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

    # Optional: attach SPECTER2 embeddings (currently disabled in UI)
    if fetch_specter2:
        try:
            if status_cb:
                status_cb(f"Fetching SPECTER2 embeddings for up to {max_embed_rows} rows…")
            df_out = attach_specter2_to_df(df_out, s2_api_key=s2_api_key, max_rows=max_embed_rows, sleep_s=s2_sleep_s)
        except Exception as e:
            if status_cb:
                status_cb(f"SPECTER2 embedding fetch failed: {e}")

    return df_out


# ----------------------------
# Streamlit UI
# ----------------------------


# ----------------------------
# OpenAI / ChatGPT enrichment (safe: only runs on loaded/uploaded data)
# ----------------------------

def openai_chat_completion(api_key: str, model: str, prompt: str, temperature: float = 0.8, timeout_s: int = 120) -> str:
    """Call OpenAI Chat Completions API via HTTPS (no openai package required)."""
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }
    r = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
    r.raise_for_status()
    data = r.json()
    return (data.get("choices") or [{}])[0].get("message", {}).get("content", "").strip()


def parse_key_challenge_review(output: str) -> tuple[str, str]:
    """Parse 2-line output into (challenge, title). Robust to extra blank lines and prefixes."""
    if not output:
        return "", ""
    lines = [ln.strip() for ln in str(output).splitlines() if ln.strip()]
    if not lines:
        return "", ""
    challenge = lines[0]
    title = lines[1] if len(lines) > 1 else ""

    for pref in ("Key challenge:", "Key Challenge:", "Challenge:"):
        if challenge.lower().startswith(pref.lower()):
            challenge = challenge[len(pref):].strip()
            break

    for pref in (
        "Possible Review Title:",
        "Possible review title:",
        "Possible Review Article Title:",
        "Review title:",
        "Title:",
    ):
        if title.lower().startswith(pref.lower()):
            title = title[len(pref):].strip()
            break

    return challenge, title


def build_input_text_from_row(row: pd.Series, mode: str, custom_col: str = "") -> str:
    """Build the text sent to ChatGPT from a dataframe row based on mode."""
    title = str(row.get("Title", "") or row.get("Article Title", "") or "")
    abstract = str(row.get("Abstract", "") or "")

    if mode == "Title":
        return title.strip()
    if mode == "Abstract":
        return abstract.strip()
    if mode == "Title + Abstract":
        if title and abstract:
            return f"{title}\n\n{abstract}".strip()
        return (title or abstract).strip()
    if mode == "Custom column" and custom_col:
        return str(row.get(custom_col, "") or "").strip()

    # fallback
    return (title + "\n\n" + abstract).strip()


def enrich_df_with_chatgpt(
    df: pd.DataFrame,
    top_n: int,
    api_key: str,
    model: str,
    temperature: float,
    prompt_text: str,
    preset_name: str,
    input_mode: str,
    custom_col: str = "",
    progress_cb=None,
) -> pd.DataFrame:
    """Run ChatGPT on top N rows and add result columns to df."""
    if df is None or df.empty:
        return df

    d = df.copy()
    n = max(1, min(int(top_n), len(d)))

    col_ch = "Key challenge identified by ChatGPT"
    col_rt = "Possible Review Article Title suggested by ChatGPT"
    col_other = f"ChatGPT output ({preset_name})"

    if preset_name == "Key challenge / Review":
        if col_ch not in d.columns:
            d[col_ch] = pd.NA
        if col_rt not in d.columns:
            d[col_rt] = pd.NA
    else:
        if col_other not in d.columns:
            d[col_other] = pd.NA

    for i in range(n):
        if progress_cb:
            progress_cb((i + 1) / n)

        row = d.iloc[i]
        input_text = build_input_text_from_row(row, mode=input_mode, custom_col=custom_col)
        if not input_text:
            continue

        prompt = prompt_text.replace("{input_text}", input_text)

        try:
            out = openai_chat_completion(api_key=api_key, model=model, prompt=prompt, temperature=temperature)
        except Exception as e:
            if preset_name == "Key challenge / Review":
                d.at[d.index[i], col_ch] = f"[ERROR] {e}"
                d.at[d.index[i], col_rt] = ""
            else:
                d.at[d.index[i], col_other] = f"[ERROR] {e}"
            continue

        if preset_name == "Key challenge / Review":
            ch, rt = parse_key_challenge_review(out)
            d.at[d.index[i], col_ch] = ch
            d.at[d.index[i], col_rt] = rt
        else:
            d.at[d.index[i], col_other] = out

    return d

st.set_page_config(page_title="Lucie's OpenAlex Search + Dashboard Sandbox (espresso v2)", layout="wide")
st.title("Lucie's OpenAlex Search + Dashboard Sandbox (espresso v2)")
st.warning(
    "🚦 **Usage note:** This app is a draft for exploration and prototyping. "
    "Please avoid fetching tens of thousands of records in one go without an API key. We do not want to get blocked."
    "Start small (e.g., **500–2,000**) and only increase if necessary. "
    "Tip: export once and save to your harddrive, you can also manually clean up and then work from **Upload CSV** to avoid re-querying OpenAlex."
    "I am still trying to understand why these results sometimes DRAMATICALLY differer in count from results from the OpenAlex web interface/other bibliographic databases."
    "One known issue is that OpenAlex sometimes classifies Meeting Abstracts as articles. "
)


with st.sidebar:
    st.header("Inputs")

    data_source = st.radio(
        "Data source",
        ["Fetch from OpenAlex", "Upload CSV", "Use loaded data"],
        index=0,
        help="Upload mode loads a previously exported CSV and does not apply OpenAlex search filters (by design)."
    )

    st.subheader("API (optional)")
    api_key = st.text_input("OpenAlex API key (optional)", value="", type="password")
    mailto = st.text_input("Contact email (optional)", value="")

    uploaded_file = None
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload app-exported CSV", type=["csv"])

    # Fetch-mode controls
    if data_source == "Fetch from OpenAlex":
        st.subheader("Query")
        keyword_query = st.text_area("Keyword / phrase query (Boolean allowed; can be empty)", value="", height=70)

        st.subheader("Journal filter")
        journal_lookup = st.text_input('Journal lookup (e.g., "Nature")', value="")
        find_journals = st.button("Find journals")

        if "source_candidates" not in st.session_state:
            st.session_state.source_candidates = []

        if find_journals:
            try:
                st.session_state.source_candidates = find_sources_by_name(journal_lookup, api_key=api_key, mailto=mailto, max_results=25)
            except Exception as e:
                st.session_state.source_candidates = []
                st.error(f"Journal search error: {e}")

        selected_source_ids: List[str] = []
        if st.session_state.source_candidates:
            options = []
            for s in st.session_state.source_candidates:
                sid = source_id_short(s.get("id", ""))
                name = s.get("display_name", "")
                org = s.get("host_organization_name", "")
                issn_l = s.get("issn_l", "")
                typ = s.get("type", "")
                options.append((sid, f"{name} | {org} | ISSN-L: {issn_l} | type: {typ}"))

            picked = st.multiselect("Select journal(s) (OR across selected journals)", options=options, format_func=lambda x: x[1])
            selected_source_ids = [x[0] for x in picked]

        st.subheader("Subfield filter (OpenAlex taxonomy)")
        subfields_df = fetch_all_subfields(api_key=api_key, mailto=mailto)
        subfield_choice = st.selectbox(
            "Subfield (3rd level) — optional",
            options=["(none)"] + (subfields_df["label"].tolist() if not subfields_df.empty else []),
            index=0,
        )
        subfield_id = None
        if subfield_choice != "(none)" and not subfields_df.empty:
            subfield_id = subfields_df.loc[subfields_df["label"] == subfield_choice, "subfield_id"].iloc[0]

        st.subheader("Year range")
        year_from_val, year_to_val = st.slider("Publication years", min_value=1900, max_value=2100, value=(2023, 2025), step=1)

        st.subheader("Coverage toggles")
        include_xpac = st.checkbox("Include xpac (expansion pack) works", value=True)
        require_doi = st.checkbox("Require DOI", value=True)
        require_abstract = st.checkbox("Require abstract", value=True)

        st.subheader("Sampling")
        n_rows = st.number_input("Rows to collect", min_value=100, max_value=50000, value=5000, step=500)

        st.subheader("Citations by year")
        include_citations_by_year = st.checkbox("Add citations by year columns", value=False)
        cite_year_from = st.number_input("Cite year from", 1900, 2100, 2017)
        cite_year_to = st.number_input("Cite year to", 1900, 2100, 2025)

        # SPECTER2 toggle (disabled for now)
        st.subheader("Embeddings (optional)")
        fetch_specter2 = st.checkbox("Fetch SPECTER2 embeddings (disabled for testing)", value=False, disabled=True)
        s2_api_key = st.text_input("Semantic Scholar API key (disabled)", value="", type="password", disabled=True)
        max_embed_rows = st.number_input("Max rows to embed (disabled)", min_value=10, max_value=5000, value=200, step=50, disabled=True)
        s2_sleep_s = st.slider("Sleep between S2 requests (s) (disabled)", 0.0, 2.0, 0.35, 0.05, disabled=True)

        st.subheader("Politeness")
        sleep_s = st.slider("Sleep between OpenAlex requests (s)", 0.0, 1.0, 0.2, 0.05)

        go = st.button("Fetch dataset", type="primary")

# Session storage
if "df" not in st.session_state:
    st.session_state.df = None

# Upload path
if data_source == "Upload CSV" and uploaded_file is not None:
    try:
        df_up = pd.read_csv(uploaded_file, low_memory=False)
        st.session_state.df = df_up
        st.success(f"Loaded uploaded CSV with {len(df_up):,} rows")
    except Exception as e:
        st.session_state.df = None
        st.error(f"Upload failed: {e}")

# Fetch path
if data_source == "Fetch from OpenAlex" and 'go' in globals() and go:
    progress = st.progress(0.0)
    status = st.empty()

    try:
        with st.spinner("Querying OpenAlex…"):
            df = fetch_works(
                n_rows=int(n_rows),
                keyword_query=keyword_query,
                year_from=int(year_from_val),
                year_to=int(year_to_val),
                api_key=api_key,
                mailto=mailto,
                sleep_s=float(sleep_s),
                source_ids=selected_source_ids or None,
                subfield_id=subfield_id,
                require_abstract=require_abstract,
                require_doi=require_doi,
                include_xpac=include_xpac,
                progress_cb=progress.progress,
                status_cb=status.write,
                fetch_specter2=fetch_specter2,
                s2_api_key=s2_api_key,
                max_embed_rows=int(max_embed_rows),
                s2_sleep_s=float(s2_sleep_s),
            )

            if include_citations_by_year:
                y1, y2 = int(cite_year_from), int(cite_year_to)
                if y1 <= y2:
                    years = list(range(y1, y2 + 1))
                    df = add_citations_by_year_columns(df, years, src_col="CountsByYear", drop_src=True)

        st.session_state.df = df
        status.success(f"Done. Collected {len(df):,} rows.")

    except Exception as e:
        st.session_state.df = None
        status.error(f"Error: {e}")


df = st.session_state.df

if df is None or df.empty:
    st.info("No dataset loaded yet. Fetch from OpenAlex or upload a CSV.")
    st.stop()

# ----------------------------
# Tabs
# ----------------------------
tab_preview, tab_growth, tab_impact, tab_chatgpt, tab_download = st.tabs([
    "📄 Preview",
    "🟩 Treemap (Topics growth)",
    "🟧 Treemap (Impact)",
    "🤖 ChatGPT",
    "⬇ Download",
])

with tab_preview:
    st.subheader("Preview")
    st.dataframe(df.head(50), use_container_width=True)

    if "Abstract" in df.columns:
        abs_cov = (df["Abstract"].fillna("").astype(str).str.strip().str.len() > 0).mean()
        st.caption(f"Abstract coverage: {abs_cov:.1%}")

    # Quick stats
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Unique DOIs", f"{df['DOI'].nunique():,}" if "DOI" in df.columns else "—")
    if "PublicationYear" in df.columns:
        ys = pd.to_numeric(df["PublicationYear"], errors="coerce")
        c3.metric("Year min", f"{int(ys.min())}" if ys.notna().any() else "—")
        c4.metric("Year max", f"{int(ys.max())}" if ys.notna().any() else "—")


with tab_growth:
    st.subheader("Treemap by Primary Topic (Growth)")
    st.caption("Size = number of papers in dataset. Color = growth over selected years.")

    if "PrimaryTopic" not in df.columns or "PublicationYear" not in df.columns:
        st.warning("Need 'PrimaryTopic' and 'PublicationYear' columns.")
    else:
        year_series = pd.to_numeric(df["PublicationYear"], errors="coerce")
        year_series = year_series.dropna().astype(int)
        if year_series.empty:
            st.warning("PublicationYear has no valid numeric values.")
        else:
            y_min, y_max = int(year_series.min()), int(year_series.max())

            colA, colB, colC = st.columns([1, 1, 1])
            with colA:
                start_year = st.number_input("Growth start year", min_value=1900, max_value=2100, value=max(y_min, y_max - 1), step=1)
            with colB:
                end_year = st.number_input("Growth end year", min_value=1900, max_value=2100, value=y_max, step=1)
            with colC:
                metric = st.selectbox("Color metric", ["CAGR", "SlopePerYear"], index=0)

            colD, colE = st.columns([1, 1])
            with colD:
                smoothing = st.slider("CAGR smoothing", 0.0, 5.0, 1.0, 0.5)
            with colE:
                min_total = st.number_input("Minimum topic size (N_total)", min_value=1, max_value=5000, value=5, step=1)

            tbl = compute_topic_growth_table(
                df=df,
                topic_col="PrimaryTopic",
                year_col="PublicationYear",
                start_year=int(start_year),
                end_year=int(end_year),
                smoothing=float(smoothing),
                min_total=int(min_total),
            )

            if tbl.empty:
                st.info("No topics available after filtering.")
            else:
                fig = px.treemap(
                    tbl,
                    path=["Topic"],
                    values="N_total",
                    color=metric,
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
                # Center diverging scale at 0
                fig.update_layout(coloraxis_cmid=0, margin=dict(t=30, l=5, r=5, b=5))
                st.plotly_chart(fig, use_container_width=True)

                # Download interactive HTML (Growth treemap)
                growth_html = fig.to_html(include_plotlyjs="cdn", full_html=True)
                st.download_button(
                    label="⬇ Download growth treemap (HTML, interactive)",
                    data=growth_html.encode("utf-8"),
                    file_name= f"growth_{metric}_{int(start_year)}-{int(end_year)}.html",   #"growth_treemap.html",
                    mime="text/html",
)


with tab_impact:
    st.subheader("Treemap by Primary Topic (Impact)")
    st.caption("Size = number of papers. Color = impact (citations per paper).")

    if "PrimaryTopic" not in df.columns:
        st.warning("Need 'PrimaryTopic' column.")
    else:
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        with col1:
            impact_mode = st.selectbox("Impact type", ["All-time", "IF-window (year X to X-1/X-2)"], index=0)
        with col2:
            agg_mode = st.selectbox("Aggregation", ["Mean", "Median"], index=0)
        with col3:
            min_topic_size = st.number_input("Min papers per topic", min_value=1, value=5, step=1)
        with col4:
            grey_small = st.checkbox("Grey small/undefined topics", value=True)

        tbl = None
        if impact_mode == "All-time":
            tbl = compute_topic_impact_table_alltime(df, agg=agg_mode, min_n=int(min_topic_size))
        else:
            # Evaluation year X must be explicitly chosen; use available year columns in df
            year_cols = sorted([c for c in df.columns if str(c).isdigit()])
            if not year_cols:
                st.info("No citation-by-year columns present. Enable 'Add citations by year columns' when fetching, or upload a file that includes year columns.")
                tbl = pd.DataFrame(columns=["Topic", "N", "Impact", "IsSmall"])
            else:
                X = int(st.selectbox("Evaluation year X", options=year_cols, index=max(0, len(year_cols) - 2)))
                tbl = compute_topic_impact_table_ifwindow(df, X=X, agg=agg_mode, min_n=int(min_topic_size))

        if tbl is None or tbl.empty:
            st.info("No impact values available for the chosen settings.")
        else:
            # Use None for greyed topics if requested
            plot_tbl = tbl.copy()
            if grey_small and "IsSmall" in plot_tbl.columns:
                plot_tbl.loc[plot_tbl["IsSmall"], "Impact"] = None

            fig = px.treemap(
                plot_tbl,
                path=["Topic"],
                values="N",
                color="Impact",
                color_continuous_scale="RdBu_r",
                hover_data={"N": True, "Impact": True},
            )
            fig.update_layout(margin=dict(t=30, l=5, r=5, b=5))
            st.plotly_chart(fig, use_container_width=True)

            # Download interactive HTML only
            html = fig.to_html(include_plotlyjs="cdn", full_html=True)
            st.download_button(
                label="⬇ Download impact treemap (HTML, interactive)",
                data=html.encode("utf-8"),
                file_name="impact_treemap.html",
                mime="text/html",
            )




with tab_chatgpt:
    st.subheader("ChatGPT enrichment")

    # Safety: do not allow ChatGPT calls while in Fetch-from-OpenAlex mode
    if 'data_source' in globals() and data_source == "Fetch from OpenAlex":
        st.warning(
            "ChatGPT enrichment is disabled while 'Data source' is set to **Fetch from OpenAlex**. "
            "Switch to **Upload CSV** or **Use loaded data** first (to avoid accidental large API spend)."
        )
        st.stop()

    st.caption("Runs on the current loaded dataframe only (uploaded or previously fetched).")

    # --- Controls ---
    colA, colB, colC = st.columns([1, 1, 1])
    with colA:
        openai_api_key = st.text_input("OpenAI API key", value="", type="password")
    with colB:
        openai_model = st.text_input("Model", value="gpt-4o-mini")
    with colC:
        openai_temp = st.slider("Temperature", 0.0, 1.2, 0.8, 0.1)

    preset = st.selectbox(
        "Prompt preset",
        ["Key challenge / Review", "(placeholder) Prompt 2", "(placeholder) Prompt 3", "Write your own"],
        index=0,
    )

    PRESET_KEY_CHALLENGE = (
        "Identify the key challenge in the following text and provide it without any line breaks, "
        "then insert a line break and on a new line provide a title for a possible review article "
        "addressing this challenge without any further line breaks.\n\n"
        "{input_text}"
    )

    PRESET_2 = (
        "Provide a one-sentence plain-language summary of the following text (single line, no line breaks).\n\n"
        "{input_text}"
    )

    PRESET_3 = (
        "Suggest three potential applications or implications (single line; separate items with '; ').\n\n"
        "{input_text}"
    )

    if preset == "Key challenge / Review":
        prompt_text = st.text_area("Prompt", value=PRESET_KEY_CHALLENGE, height=140)
    elif preset == "(placeholder) Prompt 2":
        prompt_text = st.text_area("Prompt", value=PRESET_2, height=140)
    elif preset == "(placeholder) Prompt 3":
        prompt_text = st.text_area("Prompt", value=PRESET_3, height=140)
    else:
        prompt_text = st.text_area("Prompt", value="", height=140, placeholder="Write your prompt here. Must include {input_text}.")

    # What to send
    input_mode = st.selectbox("Send to ChatGPT", ["Abstract", "Title", "Title + Abstract", "Custom column"], index=0)
    custom_col = ""
    if input_mode == "Custom column":
        text_cols = [c for c in df.columns if c not in ("SPECTER2",) ]
        custom_col = st.selectbox("Column", options=text_cols, index=0 if text_cols else 0)

    top_n = st.number_input("Top N records (test)", min_value=1, max_value=500, value=10, step=1)

    run = st.button("Run ChatGPT enrichment", type="primary")

    if run:
        if not openai_api_key.strip():
            st.error("OpenAI API key is required.")
        elif not prompt_text.strip() or "{input_text}" not in prompt_text:
            st.error("Prompt must be non-empty and include the placeholder {input_text}.")
        else:
            prog = st.progress(0.0)
            try:
                df2 = enrich_df_with_chatgpt(
                    df=df,
                    top_n=int(top_n),
                    api_key=openai_api_key.strip(),
                    model=openai_model.strip(),
                    temperature=float(openai_temp),
                    prompt_text=prompt_text,
                    preset_name=preset if preset != "Write your own" else "Custom",
                    input_mode=input_mode,
                    custom_col=custom_col,
                    progress_cb=prog.progress,
                )
                st.session_state.df = df2
                df = df2
                st.success(f"ChatGPT enrichment completed for Top {int(top_n)} rows.")
                ####
                st.download_button(
                    label="⬇ Download enriched Top N (CSV)",
                    data=df.head(int(top_n)).to_csv(index=False).encode("utf-8"),
                    file_name=f"chatgpt_enriched_top{int(top_n)}.csv",
                    mime="text/csv",
                )
                
                ####

                # --- Show results immediately in the UI ---
                st.markdown("### Preview of enriched rows (Top N)")

                if preset == "Key challenge / Review":
                    show_cols = [
                        c for c in [
                            "DOI", "Title", "Article Title",
                            "Key challenge identified by ChatGPT",
                            "Possible Review Article Title suggested by ChatGPT"
                        ]
                        if c in df.columns
                    ]
                else:
                    out_col = f"ChatGPT output ({preset if preset != 'Write your own' else 'Custom'})"
                    show_cols = [c for c in ["DOI", "Title", "Article Title", out_col] if c in df.columns]

                st.dataframe(df.head(int(top_n))[show_cols], use_container_width=True)
                ###
                with st.expander("Show input text used for the first enriched record"):
                    st.write(build_input_text_from_row(df.iloc[0], mode=input_mode, custom_col=custom_col))
                ###

            except Exception as e:
                st.error(f"ChatGPT enrichment failed: {e}")
            finally:
                prog.empty()

with tab_download:
    st.subheader("Download")
    st.download_button(
        label="Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=f"openalex_dataset_{len(df)}.csv",
        mime="text/csv",
    )
