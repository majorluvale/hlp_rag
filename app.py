*"""
HLP RAG Agent — Enhanced Edition
Housing, Land & Property AI Assistant
"""

import json
import asyncio
import os
import re
import requests
import pandas as pd
from operator import itemgetter

import streamlit as st
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers.openai_tools import JsonOutputToolsParser
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables.config import RunnableConfig
from langchain_chroma import Chroma
from langchain_core.tools.retriever import create_retriever_tool
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from typing import List
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, AIMessage
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from langchain.tools import tool
from bs4 import BeautifulSoup

# ─────────────────────────────────────────────
# ENV
# ─────────────────────────────────────────────
GROQ_API_KEY        = os.getenv("GROQ_API_KEY")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_BASE_URL   = os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")

langfuse_handler = CallbackHandler()

# ─────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="HLP Intelligence Hub",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL CSS — Premium dark humanitarian theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;1,300&family=DM+Mono:wght@400;500&family=Playfair+Display:wght@600;700&display=swap');

/* ── Root tokens ── */
:root {
  --un-blue:     #009EDB;
  --un-blue-dim: #005A8C;
  --accent:      #FF6B35;
  --accent-soft: #FF6B3520;
  --bg-base:     #0D1117;
  --bg-card:     #161B22;
  --bg-input:    #1C2128;
  --border:      #30363D;
  --text-primary:#E6EDF3;
  --text-muted:  #7D8590;
  --text-dim:    #4D5560;
  --success:     #3FB950;
  --warning:     #D29922;
  --radius:      12px;
  --radius-lg:   20px;
}

/* ── Global reset ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, .stApp {
  background: var(--bg-base) !important;
  color: var(--text-primary) !important;
  font-family: 'DM Sans', sans-serif !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: var(--bg-card) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

/* ── Main area padding ── */
.main .block-container {
  padding: 2rem 2.5rem 6rem !important;
  max-width: 900px !important;
  margin: 0 auto;
}

/* ── Header ── */
.hlp-header {
  display: flex;
  align-items: center;
  gap: 1.2rem;
  padding: 1.8rem 0 0.5rem;
  border-bottom: 1px solid var(--border);
  margin-bottom: 1.5rem;
}
.hlp-logo {
  width: 52px; height: 52px;
  background: linear-gradient(135deg, var(--un-blue), var(--un-blue-dim));
  border-radius: 14px;
  display: flex; align-items: center; justify-content: center;
  font-size: 1.6rem;
  box-shadow: 0 4px 20px rgba(0,158,219,0.3);
}
.hlp-title { line-height: 1.2; }
.hlp-title h1 {
  font-family: 'Playfair Display', serif !important;
  font-size: 1.7rem !important;
  font-weight: 700 !important;
  color: var(--text-primary) !important;
  margin: 0 !important; padding: 0 !important;
}
.hlp-title span {
  font-size: 0.78rem;
  color: var(--un-blue);
  font-weight: 500;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}
.hlp-badge {
  margin-left: auto;
  background: var(--accent-soft);
  color: var(--accent);
  font-size: 0.72rem;
  font-weight: 600;
  padding: 4px 12px;
  border-radius: 20px;
  border: 1px solid var(--accent);
  letter-spacing: 0.06em;
  text-transform: uppercase;
}

/* ── Status pills ── */
.status-bar {
  display: flex; gap: 0.6rem; flex-wrap: wrap;
  margin-bottom: 1.2rem;
}
.pill {
  font-size: 0.72rem; font-weight: 500;
  padding: 3px 10px; border-radius: 20px;
  display: inline-flex; align-items: center; gap: 5px;
}
.pill-green  { background:#3FB95015; color:var(--success); border:1px solid #3FB95040; }
.pill-blue   { background:#009EDB15; color:var(--un-blue); border:1px solid #009EDB40; }
.pill-orange { background:#FF6B3515; color:var(--accent);  border:1px solid #FF6B3540; }
.dot { width:6px; height:6px; border-radius:50%; background:currentColor; }

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
  background: transparent !important;
  border: none !important;
  padding: 0 !important;
  margin-bottom: 1rem !important;
}

/* User bubble */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) .stMarkdown {
  background: linear-gradient(135deg, #1B2A40, #162030) !important;
  border: 1px solid #2A4060 !important;
  border-radius: 18px 18px 4px 18px !important;
  padding: 12px 18px !important;
  max-width: 78% !important;
  margin-left: auto !important;
  font-size: 0.93rem;
}

/* Assistant bubble */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) .stMarkdown {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 4px 18px 18px 18px !important;
  padding: 14px 20px !important;
  max-width: 88% !important;
  margin-right: auto !important;
  font-size: 0.93rem;
  line-height: 1.65;
}

/* Hide avatars for cleaner look */
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"] {
  display: none !important;
}

/* Text colors inside bubbles */
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] span {
  color: var(--text-primary) !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] {
  background: var(--bg-input) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-lg) !important;
}
[data-testid="stChatInput"]:focus-within {
  border-color: var(--un-blue) !important;
  box-shadow: 0 0 0 3px rgba(0,158,219,0.15) !important;
}
[data-testid="stChatInput"] textarea {
  color: var(--text-primary) !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: 0.95rem !important;
}

/* ── Buttons ── */
.stButton > button {
  background: var(--bg-card) !important;
  color: var(--text-primary) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: 0.85rem !important;
  transition: all 0.15s ease;
}
.stButton > button:hover {
  border-color: var(--un-blue) !important;
  background: rgba(0,158,219,0.08) !important;
  color: var(--un-blue) !important;
}

/* ── Metrics card (for planning data) ── */
.metric-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
  gap: 0.8rem;
  margin: 1rem 0;
}
.metric-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1rem 1.2rem;
  transition: border-color 0.2s;
}
.metric-card:hover { border-color: var(--un-blue); }
.metric-label {
  font-size: 0.72rem;
  font-weight: 600;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.07em;
  margin-bottom: 0.3rem;
}
.metric-value {
  font-family: 'DM Mono', monospace;
  font-size: 1.4rem;
  font-weight: 500;
  color: var(--un-blue);
}
.metric-sub {
  font-size: 0.72rem;
  color: var(--text-dim);
  margin-top: 2px;
}

/* ── Section divider ── */
.divider {
  border: none;
  border-top: 1px solid var(--border);
  margin: 1.5rem 0;
}

/* ── Thinking indicator ── */
.thinking {
  display: inline-flex; gap: 4px; align-items: center;
  padding: 8px 14px;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 20px;
  font-size: 0.8rem; color: var(--text-muted);
  font-style: italic;
}
.thinking span {
  width: 5px; height: 5px; border-radius: 50%;
  background: var(--un-blue);
  animation: blink 1.4s infinite;
}
.thinking span:nth-child(2) { animation-delay: 0.2s; }
.thinking span:nth-child(3) { animation-delay: 0.4s; }
@keyframes blink {
  0%, 80%, 100% { opacity: 0.2; }
  40% { opacity: 1; }
}

/* ── Sidebar sections ── */
.sidebar-section {
  background: var(--bg-base);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1rem;
  margin-bottom: 0.8rem;
}
.sidebar-title {
  font-size: 0.75rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--text-muted);
  margin-bottom: 0.6rem;
}

/* ── Selectbox, number input ── */
.stSelectbox div[data-baseweb="select"] > div,
.stNumberInput input {
  background: var(--bg-input) !important;
  border-color: var(--border) !important;
  color: var(--text-primary) !important;
  font-family: 'DM Sans', sans-serif !important;
  border-radius: 8px !important;
}

/* ── Expander ── */
details summary {
  color: var(--un-blue) !important;
  font-size: 0.85rem !important;
  font-weight: 500;
}

/* ── Source tag ── */
.source-tag {
  display: inline-block;
  font-size: 0.7rem;
  font-weight: 500;
  padding: 2px 8px;
  border-radius: 4px;
  margin: 2px;
  background: rgba(0,158,219,0.1);
  color: var(--un-blue);
  border: 1px solid rgba(0,158,219,0.25);
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPERS — Planning data (from your getPlanData)
# ─────────────────────────────────────────────

def _get_global_cluster_info(cluster_id: int):
    url = "https://api.hpc.tools/v1/public/global-cluster"
    try:
        resp = requests.get(url, timeout=10).json()["data"]
        code = next((x["code"] for x in resp if x["id"] == cluster_id), None)
        name = next((x["name"] for x in resp if x["id"] == cluster_id), None)
        return code, name
    except Exception:
        return None, None


def fetch_planning_data(year: int = 2025, global_cluster_id: int = 14) -> pd.DataFrame:
    code, cluster_name = _get_global_cluster_info(global_cluster_id)
    if not code:
        return pd.DataFrame()

    url = "https://api.hpc.tools/v2/public/planSummary"
    params = {
        "year": year,
        "includeIndicators": True,
        "includeCaseloads": True,
        "includeFinancials": True,
        "includeDisaggregatedData": True,
    }
    try:
        r = requests.get(url, params=params, timeout=20)
        data = r.json()["data"]["planData"]
    except Exception:
        return pd.DataFrame()

    plans = pd.json_normalize(data, max_level=1)[[
        "planId", "planYear", "name", "planType", "isReleased", "planCostingType",
        "planFocusCountry.name", "planFocusCountry.iso3",
    ]].rename(columns={
        "planFocusCountry.name": "countryName",
        "planFocusCountry.iso3": "countryISO3",
        "name": "planName",
    })

    caseloads = pd.json_normalize(data, record_path=["caseloads"], meta=["planId"])
    expanded  = caseloads.explode("measurements")
    expanded["measurements"] = expanded["measurements"].apply(lambda x: {} if pd.isna(x) else x)
    cumreach = pd.json_normalize(expanded["measurements"])
    if "cumulativeReach" in cumreach.columns:
        df = caseloads.join(cumreach["cumulativeReach"])
    else:
        caseloads["cumulativeReach"] = 0
        df = caseloads

    df = df[df["availableGlobalClusterCode"] == code]
    drop_cols = [c for c in [
        "caseloadId","caseloadCustomRef","caseloadType","caseloadDescription",
        "availableGlobalClusterCode","entityId","totalPopulation","affected","measurements"
    ] if c in df.columns]
    df.drop(drop_cols, axis=1, inplace=True)

    for col in ["inNeed", "target", "cumulativeReach"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Financials
    requirements = [p for p in data if p.get("financialData", {}).get("requirements", {}).get("breakdown")]
    if requirements:
        req = pd.json_normalize(
            requirements, meta=["planId"], max_level=3,
            record_path=["financialData", "requirements", "breakdown", "byGlobalCluster"],
        )
        req = req[req["globalClusterId"] == global_cluster_id]
    else:
        req = pd.DataFrame(columns=["planId", "requirements"])

    try:
        funded = pd.json_normalize(
            data, meta=["planId"], max_level=3,
            record_path=["financialData", "funding", "breakdown", "byGlobalCluster"],
        )
        funded = funded[funded["globalClusterId"] == global_cluster_id]
        funded_data = req.merge(funded, how="outer", on="planId")
    except Exception:
        funded_data = req

    df = df.merge(funded_data, how="outer", on="planId")
    df.fillna(0, inplace=True)
    df = df.merge(plans, on="planId", how="left")
    df.rename(columns={
        "inNeed":          "peopleInNeed",
        "target":          "peopleTargeted",
        "cumulativeReach": "peopleReached",
        "requirements":    "requiredFunds",
        "funding":         "fundedAmount",
    }, inplace=True)

    keep = [c for c in [
        "planId","planYear","countryName","countryISO3","planName","planType",
        "isReleased","planCostingType","peopleInNeed","peopleTargeted","peopleReached",
        "requiredFunds","fundedAmount","cashTransferFunding",
    ] if c in df.columns]
    return df[keep]


def fmt_num(n) -> str:
    try:
        n = float(n)
        if n >= 1_000_000:
            return f"{n/1_000_000:.1f}M"
        if n >= 1_000:
            return f"{n/1_000:.0f}K"
        return str(int(n))
    except Exception:
        return "—"


def fmt_usd(n) -> str:
    try:
        n = float(n)
        if n >= 1_000_000_000:
            return f"${n/1_000_000_000:.2f}B"
        if n >= 1_000_000:
            return f"${n/1_000_000:.1f}M"
        return f"${n:,.0f}"
    except Exception:
        return "—"


# ─────────────────────────────────────────────
# TOOLS
# ─────────────────────────────────────────────

@tool
def chiffre_cle_pays(nom_pays: str, context_humanitaire: str) -> str:
    """Affiche le contexte humanitaire HLP par pays.
    Args:
        nom_pays: Nom du pays
        context_humanitaire: Contexte humanitaire.
    """
    return f"Contexte HLP pour {nom_pays}: {context_humanitaire}"


@tool
def get_hlp_planning_data(year: int = 2025, country_filter: str = "") -> str:
    """Retrieve HLP AoR planning data: People in Need, People Targeted, People Reached,
    funding requirements and funding received from OCHA HPC API.
    Args:
        year: The plan year (2019–2025). Defaults to 2025.
        country_filter: Optional country name to filter results (e.g. 'Ukraine', 'Somalia').
    """
    df = fetch_planning_data(year=year, global_cluster_id=14)
    if df.empty:
        return f"No planning data found for year {year}."

    if country_filter:
        mask = df["countryName"].str.contains(country_filter, case=False, na=False)
        df = df[mask]
        if df.empty:
            return f"No HLP planning data found for '{country_filter}' in {year}."

    total_pin      = df["peopleInNeed"].sum()    if "peopleInNeed"    in df.columns else 0
    total_targeted = df["peopleTargeted"].sum()  if "peopleTargeted"  in df.columns else 0
    total_reached  = df["peopleReached"].sum()   if "peopleReached"   in df.columns else 0
    total_req      = df["requiredFunds"].sum()   if "requiredFunds"   in df.columns else 0
    total_funded   = df["fundedAmount"].sum()    if "fundedAmount"    in df.columns else 0

    coverage = (total_funded / total_req * 100) if total_req > 0 else 0

    result = f"""
HLP AoR Planning Data — {year}{' / ' + country_filter if country_filter else ' (Global)'}

AGGREGATE FIGURES:
• People in Need:    {fmt_num(total_pin)}
• People Targeted:  {fmt_num(total_targeted)}
• People Reached:   {fmt_num(total_reached)}
• Funding Required: {fmt_usd(total_req)}
• Funding Received: {fmt_usd(total_funded)}
• Funding Coverage: {coverage:.1f}%

COUNTRY BREAKDOWN ({len(df)} plans):
"""
    for _, row in df.iterrows():
        country = row.get("countryName", "Unknown")
        pin     = fmt_num(row.get("peopleInNeed", 0))
        target  = fmt_num(row.get("peopleTargeted", 0))
        reached = fmt_num(row.get("peopleReached", 0))
        req_f   = fmt_usd(row.get("requiredFunds", 0))
        funded  = fmt_usd(row.get("fundedAmount", 0))
        result += f"  {country}: PIN={pin}, Target={target}, Reached={reached}, Req={req_f}, Funded={funded}\n"

    return result.strip()


@tool
def search_hlp_resources(query: str) -> str:
    """Search HLP documents on the Global Protection Cluster website and NRC.
    Useful for policy documents, guidelines, and technical resources on HLP/LTP.
    Args:
        query: Search terms related to HLP topics.
    """
    results = []
    headers = {"User-Agent": "Mozilla/5.0 (compatible; HLPBot/1.0)"}

    # GPC HLP Resources
    try:
        url = f"https://www.globalprotectioncluster.org/?s={requests.utils.quote(query)}&post_type=resource"
        r   = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        items = soup.select("article h2 a, .resource-title a, .entry-title a")[:5]
        for item in items:
            title = item.get_text(strip=True)
            href  = item.get("href", "")
            if title and href:
                results.append(f"[GPC] {title} — {href}")
    except Exception as e:
        results.append(f"[GPC] Search unavailable: {e}")

    # NRC Search
    try:
        url = f"https://www.nrc.no/search/?q={requests.utils.quote(query)}&type=publication"
        r   = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        items = soup.select(".search-result__title a, .article-list__title a")[:5]
        for item in items:
            title = item.get_text(strip=True)
            href  = item.get("href", "")
            if href and not href.startswith("http"):
                href = "https://www.nrc.no" + href
            if title and href:
                results.append(f"[NRC] {title} — {href}")
    except Exception as e:
        results.append(f"[NRC] Search unavailable: {e}")

    if not results:
        return f"No results found for '{query}' on GPC or NRC."
    return "\n".join(results)


@tool
def search_reliefweb(query: str, limit: int = 5) -> str:
    """Search ReliefWeb for humanitarian reports, situation reports and HLP-related content.
    Args:
        query: Search terms (e.g. 'housing land property Ukraine 2024').
        limit: Number of results to return (default 5).
    """
    try:
        url = "https://api.reliefweb.int/v1/reports"
        params = {
            "appname": "hlp-agent",
            "query[value]": query,
            "query[operator]": "AND",
            "fields[include][]": ["title", "url", "date.original", "source.name", "body-html"],
            "limit": limit,
            "sort[]": "date:desc",
        }
        r = requests.get(url, params=params, timeout=12)
        data = r.json().get("data", [])
        if not data:
            return f"No ReliefWeb results found for '{query}'."

        results = []
        for item in data:
            fields = item.get("fields", {})
            title  = fields.get("title", "Untitled")
            url_rw = fields.get("url", "")
            date   = fields.get("date", {}).get("original", "")[:10]
            src    = ", ".join(s.get("name","") for s in fields.get("source", []))
            body   = BeautifulSoup(fields.get("body-html",""), "html.parser").get_text()[:400]
            results.append(f"[{date}] {title} ({src})\n  URL: {url_rw}\n  Excerpt: {body}...")
        return "\n\n".join(results)
    except Exception as e:
        return f"ReliefWeb search error: {e}"


@tool
def scrape_hpc_learning(query: str) -> str:
    """Scrape OCHA HPC Collective Learning knowledge base for operational guidance,
    lessons learned, and best practices on humanitarian coordination including HLP.
    Args:
        query: Topic or keywords to search.
    """
    results = []
    headers = {"User-Agent": "Mozilla/5.0 (compatible; HLPBot/1.0)"}
    try:
        url = f"https://collective.humanitarianresponse.info/?search={requests.utils.quote(query)}"
        r   = requests.get(url, headers=headers, timeout=12)
        soup = BeautifulSoup(r.text, "html.parser")
        items = soup.select("h2 a, h3 a, .views-field-title a")[:6]
        for item in items:
            title = item.get_text(strip=True)
            href  = item.get("href", "")
            if href and not href.startswith("http"):
                href = "https://collective.humanitarianresponse.info" + href
            if title:
                results.append(f"[HPC Learning] {title} — {href}")
    except Exception as e:
        results.append(f"HPC search unavailable: {e}")

    # Also try humanitarianresponse.info directly
    try:
        url = f"https://www.humanitarianresponse.info/en/search?query={requests.utils.quote(query)}"
        r   = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        items = soup.select(".search-result h3 a, .views-row h3 a")[:4]
        for item in items:
            title = item.get_text(strip=True)
            href  = item.get("href", "")
            if title:
                results.append(f"[HR.info] {title} — {href}")
    except Exception:
        pass

    return "\n".join(results) if results else f"No HPC learning results for '{query}'."


@tool
def fetch_webpage_content(url: str) -> str:
    """Fetch and extract text content from a web page URL to read its full content.
    Useful for reading linked HLP documents, reports, guidelines from GPC, NRC, IOM, IFRC, OCHA.
    Args:
        url: Full URL of the page to fetch (must start with https://).
    """
    allowed_domains = [
        "globalprotectioncluster.org", "nrc.no", "reliefweb.int",
        "iom.int", "ifrc.org", "humanitarianresponse.info",
        "hpc.tools", "unocha.org", "unhcr.org", "ohchr.org",
        "collective.humanitarianresponse.info",
    ]
    domain = url.split("/")[2] if url.startswith("http") else ""
    if not any(d in domain for d in allowed_domains):
        return f"Domain not in allowlist. Allowed: {', '.join(allowed_domains)}"
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; HLPBot/1.0)"}
        r = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(r.text, "html.parser")
        # Remove nav/footer noise
        for tag in soup(["nav","footer","header","script","style","aside"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        # Clean blank lines
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        clean = "\n".join(lines)[:4000]
        return f"Content from {url}:\n\n{clean}"
    except Exception as e:
        return f"Failed to fetch {url}: {e}"


@tool
def search_iom_ifrc(query: str) -> str:
    """Search IOM (International Organization for Migration) and IFRC
    (International Federation of Red Cross) websites for HLP-related content.
    Args:
        query: Search terms.
    """
    results = []
    headers = {"User-Agent": "Mozilla/5.0 (compatible; HLPBot/1.0)"}

    # IOM
    try:
        url = f"https://www.iom.int/search?search_api_fulltext={requests.utils.quote(query)}"
        r   = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        items = soup.select(".view-search-results h3 a, .search-result__title a")[:4]
        for item in items:
            title = item.get_text(strip=True)
            href  = item.get("href", "")
            if href and not href.startswith("http"):
                href = "https://www.iom.int" + href
            if title:
                results.append(f"[IOM] {title} — {href}")
    except Exception as e:
        results.append(f"[IOM] Search unavailable: {e}")

    # IFRC
    try:
        url = f"https://www.ifrc.org/search?keywords={requests.utils.quote(query)}"
        r   = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        items = soup.select(".search-result a, .card__title a")[:4]
        for item in items:
            title = item.get_text(strip=True)
            href  = item.get("href", "")
            if href and not href.startswith("http"):
                href = "https://www.ifrc.org" + href
            if title:
                results.append(f"[IFRC] {title} — {href}")
    except Exception as e:
        results.append(f"[IFRC] Search unavailable: {e}")

    return "\n".join(results) if results else f"No IOM/IFRC results for '{query}'."


# ─────────────────────────────────────────────
# EMBEDDINGS & VECTOR STORE
# ─────────────────────────────────────────────

@st.cache_resource
def load_vector_store():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return Chroma(
        collection_name="hlp_aor",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db",
    )


vector_store = load_vector_store()

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4},
)

retriever_tool = create_retriever_tool(
    retriever,
    "hlprag",
    "HLP knowledge base — answers questions on Housing, Land and Property from trained documents.",
    document_separator="\n\n",
)

tools = [
    retriever_tool,
    chiffre_cle_pays,
    get_hlp_planning_data,
    search_hlp_resources,
    search_reliefweb,
    scrape_hpc_learning,
    fetch_webpage_content,
    search_iom_ifrc,
]

# ─────────────────────────────────────────────
# LLM
# ─────────────────────────────────────────────

@st.cache_resource
def load_llm():
    llm = ChatOpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
        model="openai/gpt-oss-120b",
        temperature=0.4,
        max_tokens=1500,
    )
    return llm.bind_tools(tools)


llm = load_llm()

# ─────────────────────────────────────────────
# PROMPT
# ─────────────────────────────────────────────

prompt = PromptTemplate.from_template("""
You are the HLP Intelligence Hub — an expert AI assistant specialized in
Housing, Land, and Property (HLP/LTP/LTB) in humanitarian contexts.

Your capabilities:
- Answer HLP-related questions using your knowledge base (hlprag tool)
- Retrieve real-time HLP planning data: PIN, People Targeted, People Reached,
  funding requirements and received (get_hlp_planning_data tool)
- Search HLP resources on GPC, NRC, ReliefWeb, IOM, IFRC, HPC Learning
- Read full web page content when needed for deeper answers
- Provide country-specific HLP context

Key context:
- Humanitarian Reset: SLSCC (Shelter, Land and Site Coordination Cluster) integrates Shelter, CCCM, HLP AoR
- Global Cluster Leads: IFRC and IOM; NRC is Global POLR for HLP
- POLR = Provider of Last Resort
- HLP priorities: legal assistance, eviction prevention, restitution, due diligence,
  gender and women's land rights, recovery and reconstruction, IM & analysis
- Always maintain links with Protection Cluster; ensure protection mainstreaming
- Cluster principles: predictability, accountability, flexibility, transparency

Instructions:
- Answer ONLY HLP-related questions
- Respond in the same language as the question
- Always cite sources when using web tools (include URLs)
- For planning data questions, use get_hlp_planning_data tool
- For recent reports or news, use search_reliefweb or search_hlp_resources
- When you find a relevant URL, use fetch_webpage_content to read it
- Format numbers clearly (M = millions, K = thousands, $ for USD)
- If you cannot find the answer, say so politely and suggest where to look
- Never fabricate statistics; always use real data from tools

Conversation history: {chat_history}
Question: {input}
Context from tools: {context}
Response:
""")

# ─────────────────────────────────────────────
# IN-MEMORY HISTORY
# ─────────────────────────────────────────────

store = {}


class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    messages: List[BaseMessage] = Field(default_factory=list)

    def add_messages(self, messages: List[BaseMessage]) -> None:
        self.messages.extend(messages)

    def clear(self) -> None:
        self.messages = []


# ─────────────────────────────────────────────
# AGENT
# ─────────────────────────────────────────────

class Agent:

    def __init__(self, session_id: str):
        self.session_id     = session_id
        self.tools          = tools
        self.llm            = llm
        self.prompt         = prompt
        self.call_tool_list = RunnableLambda(self.call_tool).map()
        self.chain          = self.prompt | self.llm | self.route
        self.session_history = None
        self.config = RunnableConfig(session_id=self.session_id)
        self.conversation = RunnableWithMessageHistory(
            self.chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        ).with_config(RunnableConfig(
            run_name="hlp-intelligence-hub",
            callbacks=[langfuse_handler],
        ))

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = InMemoryHistory()
        self.session_history = store[session_id]
        return self.session_history

    def call_tool(self, tool_invocation: dict) -> Runnable:
        tool_map = {t.name: t for t in tools}
        tool = tool_map.get(tool_invocation.get("type", ""))
        if not tool:
            return RunnablePassthrough.assign(output=lambda _: "Tool not found.")
        return RunnablePassthrough.assign(output=itemgetter("args") | tool)

    def parse_context(self, get_mo_output: list) -> str:
        self.session_history.add_message(
            AIMessage(json.dumps(get_mo_output, indent=2))
        )
        return {
            "context": json.dumps(get_mo_output, indent=2),
            "input":   self.question,
            "chat_history": self.session_history.messages,
        }

    def route(self, message: AIMessage):
        if not message.tool_calls:
            return StrOutputParser()
        return (
            JsonOutputToolsParser()
            | self.call_tool_list
            | self.parse_context
            | self.chain
        )

    async def astream(self, question: str):
        self.question = question
        async for event in self.conversation.astream_events(
            {"input": self.question, "chat_history": [], "context": ""},
            config=self.config,
            version="v2",
        ):
            if event["event"] == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if chunk and hasattr(chunk, "content"):
                    yield chunk.content


# ─────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────

# ── Session state ──
if "history"    not in st.session_state: st.session_state.history    = []
if "session_id" not in st.session_state: st.session_state.session_id = "default"
if "plan_data"  not in st.session_state: st.session_state.plan_data  = None

# ── Sidebar ──
with st.sidebar:
    st.markdown("""
    <div style='padding:1rem 0 0.5rem'>
      <div style='font-family:"Playfair Display",serif;font-size:1.15rem;
                  font-weight:700;color:#E6EDF3;margin-bottom:4px'>
        HLP Intelligence Hub
      </div>
      <div style='font-size:0.72rem;color:#009EDB;letter-spacing:0.08em;
                  text-transform:uppercase;font-weight:600'>
        Powered by NRC · OCHA · Global Cluster
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Planning data widget
    st.markdown('<div class="sidebar-title">📊 Planning Data Explorer</div>', unsafe_allow_html=True)

    selected_year    = st.selectbox("Year", [2025, 2024, 2023, 2022, 2021, 2020, 2019], index=0)
    country_filter   = st.text_input("Filter by country", placeholder="e.g. Ukraine, Somalia…")
    load_data_btn    = st.button("⬇  Load Planning Data", use_container_width=True)

    if load_data_btn:
        with st.spinner("Fetching data from OCHA HPC API…"):
            df = fetch_planning_data(year=selected_year, global_cluster_id=14)
            if country_filter:
                mask = df["countryName"].str.contains(country_filter, case=False, na=False)
                df = df[mask]
            st.session_state.plan_data = df
        if not df.empty:
            st.success(f"{len(df)} plan(s) loaded ✓")
        else:
            st.warning("No data found.")

    st.divider()

    # Tools status
    st.markdown('<div class="sidebar-title">🔧 Active Tools</div>', unsafe_allow_html=True)
    for t_name, t_desc in [
        ("🗄  Knowledge Base", "HLP RAG (local)"),
        ("📊  Planning Data", "OCHA HPC API"),
        ("🌐  GPC / NRC Search", "Web scraper"),
        ("📰  ReliefWeb", "Live API"),
        ("🏫  HPC Learning", "OCHA scraper"),
        ("🌍  IOM / IFRC", "Web scraper"),
        ("📄  Document Reader", "URL fetcher"),
    ]:
        st.markdown(f"""
        <div style='display:flex;justify-content:space-between;align-items:center;
                    padding:4px 0;border-bottom:1px solid #30363D;font-size:0.8rem'>
          <span style='color:#E6EDF3'>{t_name}</span>
          <span style='color:#3FB950;font-size:0.7rem'>● active</span>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # Observability link
    st.markdown('<div class="sidebar-title">🔭 Observability</div>', unsafe_allow_html=True)
    langfuse_url = LANGFUSE_BASE_URL or "https://cloud.langfuse.com"
    st.markdown(f"""
    <a href="{langfuse_url}" target="_blank"
       style='display:block;text-align:center;padding:8px;
              background:#161B22;border:1px solid #30363D;border-radius:8px;
              color:#009EDB;font-size:0.8rem;text-decoration:none;
              font-weight:500;transition:all .15s'>
      🔗 Open Langfuse Dashboard
    </a>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.72rem;color:#7D8590;margin-top:6px;line-height:1.5'>
      Evaluate responses, add feedback scores, and correct answers — no retraining needed.
    </div>""", unsafe_allow_html=True)

    st.divider()

    if st.button("🗑  Clear conversation", use_container_width=True):
        st.session_state.history = []
        st.session_state.session_id = f"session_{len(store)}"
        store.clear()
        st.rerun()

# ── Main area ──

# Header
st.markdown("""
<div class="hlp-header">
  <div class="hlp-logo">🏠</div>
  <div class="hlp-title">
    <h1>HLP Intelligence Hub</h1>
    <span>Housing · Land · Property — Humanitarian AI Assistant</span>
  </div>
  <div class="hlp-badge">Beta v2</div>
</div>
""", unsafe_allow_html=True)

# Status pills
st.markdown("""
<div class="status-bar">
  <span class="pill pill-green"><span class="dot"></span>Knowledge base ready</span>
  <span class="pill pill-blue"><span class="dot"></span>OCHA HPC API connected</span>
  <span class="pill pill-orange"><span class="dot"></span>7 tools active</span>
  <span class="pill pill-green"><span class="dot"></span>Langfuse observability</span>
</div>
""", unsafe_allow_html=True)

# Planning data dashboard (if loaded)
if st.session_state.plan_data is not None and not st.session_state.plan_data.empty:
    df = st.session_state.plan_data
    total_pin  = df["peopleInNeed"].sum()   if "peopleInNeed"   in df.columns else 0
    total_tgt  = df["peopleTargeted"].sum() if "peopleTargeted" in df.columns else 0
    total_rch  = df["peopleReached"].sum()  if "peopleReached"  in df.columns else 0
    total_req  = df["requiredFunds"].sum()  if "requiredFunds"  in df.columns else 0
    total_fund = df["fundedAmount"].sum()   if "fundedAmount"   in df.columns else 0
    cov        = total_fund / total_req * 100 if total_req > 0 else 0

    st.markdown(f"""
    <div style='margin-bottom:0.5rem;font-size:0.78rem;font-weight:600;
                color:#7D8590;text-transform:uppercase;letter-spacing:0.07em'>
      📊 HLP AoR Planning Data — {selected_year}
    </div>
    <div class="metric-grid">
      <div class="metric-card">
        <div class="metric-label">People in Need</div>
        <div class="metric-value">{fmt_num(total_pin)}</div>
        <div class="metric-sub">PIN</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">People Targeted</div>
        <div class="metric-value">{fmt_num(total_tgt)}</div>
        <div class="metric-sub">Targeted</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">People Reached</div>
        <div class="metric-value">{fmt_num(total_rch)}</div>
        <div class="metric-sub">Cumulative reach</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Funding Required</div>
        <div class="metric-value">{fmt_usd(total_req)}</div>
        <div class="metric-sub">Requirements</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Funding Received</div>
        <div class="metric-value">{fmt_usd(total_fund)}</div>
        <div class="metric-sub">{cov:.1f}% coverage</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander(f"📋 Country detail ({len(df)} plans)"):
        disp_cols = [c for c in ["countryName","planType","peopleInNeed","peopleTargeted",
                                  "peopleReached","requiredFunds","fundedAmount"] if c in df.columns]
        st.dataframe(
            df[disp_cols].rename(columns={
                "countryName":"Country","planType":"Type",
                "peopleInNeed":"PIN","peopleTargeted":"Targeted",
                "peopleReached":"Reached","requiredFunds":"Required ($)",
                "fundedAmount":"Funded ($)",
            }),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

# Suggested prompts (only when chat is empty)
if not st.session_state.history:
    st.markdown("""
    <div style='font-size:0.8rem;color:#7D8590;margin-bottom:0.8rem;font-weight:500'>
      💡 Try asking…
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(2)
    suggestions = [
        "What is the HLP AoR's role in the humanitarian reset?",
        "Show me HLP planning data for Somalia 2024",
        "What are the latest NRC resources on eviction prevention?",
        "Find recent ReliefWeb reports on land rights in Ukraine",
    ]
    for i, sug in enumerate(suggestions):
        with cols[i % 2]:
            if st.button(sug, key=f"sug_{i}", use_container_width=True):
                st.session_state._prefill = sug
                st.rerun()

# Chat messages
chat_container = st.container()
with chat_container:
    for msg in st.session_state.history:
        st.chat_message(msg["role"]).write(msg["content"])

# Chat input
prefill = getattr(st.session_state, "_prefill", "")
if prefill:
    del st.session_state._prefill

user_input = st.chat_input("Ask anything about Housing, Land & Property…") or prefill

if user_input:
    st.session_state.history.append({"role": "user", "content": user_input})
    with chat_container:
        st.chat_message("user").write(user_input)

    agent   = Agent(st.session_state.session_id)
    placeholder     = chat_container.empty()
    response_chunks = []

    async def get_response():
        async for chunk in agent.astream(user_input):
            response_chunks.append(chunk)
            placeholder.chat_message("assistant").write("".join(response_chunks))

    try:
        loop = asyncio.get_running_loop()
        loop.create_task(get_response())
    except RuntimeError:
        asyncio.run(get_response())

    final = "".join(response_chunks)
    st.session_state.history.append({"role": "assistant", "content": final})
