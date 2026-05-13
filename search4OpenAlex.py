# stSearchDashOpenAlex_espresso_patched_v2.py
#
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


# ----------------------------
# Streamlit UI
# ----------------------------
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
        ["Fetch from OpenAlex", "Upload CSV"],
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
tab_preview, tab_growth, tab_impact, tab_download = st.tabs([
    "📄 Preview",
    "🟩 Treemap (Topics growth)",
    "🟧 Treemap (Impact)",
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


with tab_download:
    st.subheader("Download")
    st.download_button(
        label="Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=f"openalex_dataset_{len(df)}.csv",
        mime="text/csv",
    )
