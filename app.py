import streamlit as st
import requests
import feedparser
import pytz
import torch
import json
import pathlib
import plotly.graph_objects as go
from datetime import datetime, timedelta, timezone
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="Shantanu's Market Update", page_icon="📈", layout="wide")

IST            = pytz.timezone("Asia/Kolkata")
MODEL_NAME     = "ProsusAI/finbert"
NEWS_BIAS_PATH = pathlib.Path("/tmp/news_bias.json")  # Feature 6 bridge file

NEWSAPI_KEY    = st.secrets.get("NEWSAPI_KEY", "")
FINNHUB_KEY    = st.secrets.get("FINNHUB_KEY", "")
FREENEWS_KEY   = st.secrets.get("FREENEWS_KEY", "")
OWNER_PASSWORD = st.secrets.get("OWNER_PASSWORD", "")

# Upstash Redis — used for cross-session persistence of score_history and
# news_bias.  Free tier (10k req/day, 256 MB) is sufficient.
# Add to .streamlit/secrets.toml:
#   UPSTASH_REDIS_URL   = "https://xxxx.upstash.io"
#   UPSTASH_REDIS_TOKEN = "your-token"
REDIS_URL   = st.secrets.get("UPSTASH_REDIS_URL", "")
REDIS_TOKEN = st.secrets.get("UPSTASH_REDIS_TOKEN", "")

# ── Model (loaded once, shared across all sessions) ───────────────────────────

@st.cache_resource
def load_model():
    # HF_TOKEN avoids unauthenticated rate-limits on cold-start downloads.
    # Add to .streamlit/secrets.toml:  HF_TOKEN = "hf_xxxx"
    hf_token = st.secrets.get("HF_TOKEN", None)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)
    mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, token=hf_token)
    mdl.eval()
    return tok, mdl

tokenizer, model = load_model()

SCORE_MAP = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
# ProsusAI/FinBERT config.json: {0: positive, 1: negative, 2: neutral}
# Derived directly from the loaded model so it stays correct if the model
# is ever swapped out.
ID2LABEL  = {int(k): v for k, v in model.config.id2label.items()}

# ── Free rules-based override layer ──────────────────────────────────────────
# FinBERT was trained on company-level text and mis-scores geopolitical /
# macro headlines because conflict vocabulary dominates sentiment cues.
# e.g. "Street indexes jump, Trump says strikes against Iran canceled"
#      → FinBERT sees "strikes"+"Iran" and outputs negative.
#
# Fix (no API needed, no cost): scan the headline for unambiguous market-
# direction phrases AFTER FinBERT scores it.  If a strong override phrase is
# present, replace FinBERT's label.  The "scorer" field records what happened
# so the UI can badge overridden headlines with a ⚡ marker.
#
# Design notes:
#  • Only MULTI-WORD or unambiguous phrases are used — single words like
#    "fall" or "rise" are too noisy on their own.
#  • Phrases are matched on word boundaries (space/start/end) to avoid
#    accidental substring hits.
#  • BULL phrases take priority over BEAR phrases if both somehow appear
#    (rare, but possible in mixed headlines like "indexes jump despite crash fears").

import re as _re

def _wb(phrase: str) -> _re.Pattern:
    """Compile a word-boundary-aware regex for a phrase."""
    return _re.compile(r'(?<!\w)' + _re.escape(phrase) + r'(?!\w)', _re.IGNORECASE)

# ── BULLISH override phrases ──────────────────────────────────────────────────
_BULL_PHRASES: list[_re.Pattern] = [p for p in map(_wb, [
    # explicit index / market direction
    "indexes jump", "indices jump", "index jumps", "market jumps",
    "stocks jump", "shares jump", "markets rally", "market rallies",
    "indexes rise", "indices rise", "stocks rise", "shares rise",
    "indexes surge", "indices surge", "stocks surge", "shares surge",
    "market surges", "indexes gain", "indices gain", "stocks gain",
    "bull run", "market up", "stocks up", "broad rally",
    "record high", "all-time high", "52-week high", "multi-year high",
    "strong rally", "sharp rally", "market bounces", "stocks bounce",
    # de-escalation / deal keywords
    "strikes canceled", "strikes called off", "attack canceled",
    "ceasefire", "peace deal", "trade deal signed", "deal reached",
    "sanctions lifted", "tensions ease", "de-escalation",
    "war averted", "conflict averted", "crisis averted",
    # monetary / policy tailwind
    "rate cut", "rate cuts", "rates cut", "interest rate cut",
    "stimulus approved", "stimulus package", "bailout approved",
    "fed pivots", "rbi rate cut", "dovish",
    # corporate positive
    "beats estimates", "beats expectations", "earnings beat",
    "strong earnings", "record profit", "order win", "contract win",
    "merger approved", "acquisition approved", "strong guidance",
    "upgrade to buy", "price target raised",
])]

# ── BEARISH override phrases ──────────────────────────────────────────────────
_BEAR_PHRASES: list[_re.Pattern] = [p for p in map(_wb, [
    # explicit index / market direction
    "indexes fall", "indices fall", "stocks fall", "shares fall",
    "market falls", "indexes crash", "stocks crash", "market crashes",
    "indexes plunge", "stocks plunge", "shares plunge", "market plunges",
    "indexes tumble", "stocks tumble", "indices tumble",
    "sell-off", "selloff", "broad sell-off",
    "circuit breaker", "trading halt", "lower circuit",
    "record low", "52-week low", "multi-year low",
    # escalation / negative macro
    "war declared", "military strike", "invasion begins",
    "sanctions imposed", "trade war escalates",
    "rate hike", "rates hiked", "hawkish surprise",
    "recession confirmed", "recession fears", "stagflation",
    # corporate negative
    "misses estimates", "misses expectations", "earnings miss",
    "profit warning", "guidance cut", "downgrade to sell",
    "price target cut", "layoffs", "job cuts", "bankruptcy",
    "default risk", "insolvency",
])]

def rules_override(title: str, finbert_label: str) -> tuple[str, str]:
    """
    Check title against override phrase lists.
    Returns (final_label, scorer) where scorer is:
      'finbert'       — no override triggered
      'rules-bull'    — bullish override applied
      'rules-bear'    — bearish override applied
    """
    tl = title.lower()
    # Bull phrases take priority (de-escalation + market jump combo)
    for pat in _BULL_PHRASES:
        if pat.search(tl):
            return "positive", "rules-bull"
    for pat in _BEAR_PHRASES:
        if pat.search(tl):
            return "negative", "rules-bear"
    return finbert_label, "finbert"

# ── Feature 5: sector keyword map ────────────────────────────────────────────

SECTOR_KEYWORDS = {
    "Banking":  ["hdfc", "sbi", "kotak", "axis bank", "icici", "rbi", "npa",
                 "credit", "bank", "nbfc", "microfinance", "loan", "rate cut", "repo"],
    "IT":       ["infosys", "tcs", "wipro", "hcl", "tech mahindra", "it sector",
                 "software", "nasdaq", "tech", "digital", "cloud"],
    "Energy":   ["reliance", "ongc", "oil", "petroleum", "gas", "crude",
                 "bpcl", "iocl", "coal india", "ntpc", "power", "solar"],
    "Pharma":   ["sun pharma", "cipla", "drreddy", "dr. reddy", "fda", "api",
                 "drug", "biocon", "pharmaceutical", "healthcare", "vaccine"],
    "Auto":     ["maruti", "tata motors", "m&m", "bajaj auto", "ev",
                 "automobile", "hero motocorp", "tvs", "electric vehicle"],
    "FMCG":     ["hul", "dabur", "nestle", "itc", "fmcg", "britannia",
                 "godrej", "consumer", "marico", "colgate"],
    "Metals":   ["tata steel", "jsw", "steel", "aluminium", "copper",
                 "hindalco", "vedanta", "mining", "iron ore", "zinc"],
    "Infra":    ["l&t", "adani", "infrastructure", "cement", "construction",
                 "ambuja", "ultratech", "capex", "order win"],
    "Finance":  ["sebi", "ipo", "fii", "dii", "mutual fund", "amc",
                 "futures", "options", "derivative"],
    "NIFTY":    ["nifty", "sensex", "index", "market rally",
                 "market fall", "bull run", "bear market", "circuit breaker"],
}

# ── Time helpers ──────────────────────────────────────────────────────────────

def now_ist():
    return datetime.now(IST)

def fmt_ist(dt=None):
    return (dt or now_ist()).strftime("%d %b %Y, %I:%M %p IST")

def is_market_hours() -> bool:
    n = now_ist()
    if n.weekday() >= 5:
        return False
    mins = n.hour * 60 + n.minute
    return 9 * 60 + 15 <= mins <= 15 * 60 + 30

def session_label() -> str:
    n = now_ist()
    if n.weekday() >= 5:        return "Weekend"
    mins = n.hour * 60 + n.minute
    if mins < 9 * 60:           return "Pre-Open"
    if mins < 9 * 60 + 15:     return "Pre-Market"
    if mins <= 15 * 60 + 30:   return "🟢 Market Open"
    if mins <= 16 * 60:        return "Post-Market"
    return "After Hours"

def market_refresh_interval() -> int:
    """
    Feature 2: auto-detect optimal refresh cadence by session phase.
    First/last 30 min → 2 min (high volatility windows).
    Mid-session       → 5 min.
    Outside hours     → 20 min.  Weekends → 30 min.
    """
    n = now_ist()
    if n.weekday() >= 5:
        return 30
    mins    = n.hour * 60 + n.minute
    open_m  = 9 * 60 + 15
    close_m = 15 * 60 + 30
    if not (open_m <= mins <= close_m):
        return 20
    if (mins - open_m) < 30 or (close_m - mins) < 30:
        return 2
    return 5

# ── Redis helpers ─────────────────────────────────────────────────────────────

def _redis_hdrs() -> dict:
    return {"Authorization": f"Bearer {REDIS_TOKEN}"}

def load_history_from_redis() -> list:
    """
    Pull score_history from Upstash on cold load so the timeline chart is
    never blank after a restart, sleep-wake cycle, or new tab.
    Returns a list of (IST-aware datetime, float, str) tuples, or [] on any error.
    """
    if not (REDIS_URL and REDIS_TOKEN):
        return []
    try:
        r = requests.get(
            f"{REDIS_URL}/get/news_sentiment_history",
            headers=_redis_hdrs(), timeout=5,
        )
        raw = r.json().get("result")
        if not raw:
            return []
        rows = json.loads(raw)           # [[iso_str, score, label], ...]
        result = []
        for iso, score, label in rows:
            dt = datetime.fromisoformat(iso)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            result.append((dt.astimezone(IST), float(score), str(label)))
        return result
    except Exception:
        return []

def save_history_to_redis(history: list):
    """
    Persist the last 12 h of score_history to Upstash.
    Key expires after 13 h so overnight stale data is cleaned up automatically.
    """
    if not (REDIS_URL and REDIS_TOKEN) or not history:
        return
    rows = [[t.isoformat(), s, l] for t, s, l in history]
    try:
        # Upstash REST: SETEX <key> <seconds> <value>
        requests.post(
            f"{REDIS_URL}/setex/news_sentiment_history/46800/"
            + requests.utils.quote(json.dumps(rows), safe=""),
            headers=_redis_hdrs(), timeout=5,
        )
    except Exception:
        pass

# ── Date parsing ──────────────────────────────────────────────────────────────

def parse_pub_dt(s: str):
    if not s:
        return None
    s      = s.strip()
    s_norm = s.replace(" GMT", " +0000")   # strptime doesn't parse "GMT"
    formats = [
        "%Y-%m-%dT%H:%M:%SZ",   "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%S",   "%a, %d %b %Y %H:%M:%S %z",
    ]
    for src in (s_norm, s):
        for fmt in formats:
            try:
                dt = datetime.strptime(src, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(IST)
            except (ValueError, TypeError):
                pass
    try:
        from dateutil import parser as dp
        dt = dp.parse(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(IST)
    except Exception:
        return None

def within_last_24h(article: dict) -> bool:
    dt = parse_pub_dt(article.get("published", ""))
    return dt is not None and dt >= now_ist() - timedelta(hours=24)

# ── Feature 5: sector tagger ──────────────────────────────────────────────────

def tag_sector(title: str) -> str:
    t = title.lower()
    for sector, kws in SECTOR_KEYWORDS.items():
        if any(kw in t for kw in kws):
            return sector
    return "General"

# ── Data fetching ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def fetch_newsapi():
    if not NEWSAPI_KEY:
        return []
    try:
        r = requests.get(
            "https://newsapi.org/v2/top-headlines",
            params={"country": "in", "category": "business",
                    "pageSize": 50, "apiKey": NEWSAPI_KEY},
            timeout=15,
        )
        r.raise_for_status()
        return [
            {"title": a.get("title", ""),
             "source": a.get("source", {}).get("name", "NewsAPI"),
             "url": a.get("url", ""), "published": a.get("publishedAt", ""),
             "feed": "NewsAPI"}
            for a in r.json().get("articles", [])
            if a.get("title") and "[Removed]" not in a.get("title", "")
        ]
    except Exception:
        return []

@st.cache_data(ttl=300, show_spinner=False)
def fetch_finnhub():
    if not FINNHUB_KEY:
        return []
    try:
        r = requests.get(
            "https://finnhub.io/api/v1/news",
            params={"category": "general", "token": FINNHUB_KEY},
            timeout=15,
        )
        r.raise_for_status()
        data  = r.json()
        items = data if isinstance(data, list) else []
        return [
            {"title": a["headline"],
             "source": a.get("source", "Finnhub"),
             "url": a.get("url", ""),
             "published": datetime.fromtimestamp(a["datetime"], tz=timezone.utc).isoformat(),
             "feed": "Finnhub"}
            for a in items[:40]
            if a.get("headline")
        ]
    except Exception:
        return []

@st.cache_data(ttl=300, show_spinner=False)
def fetch_rss():
    feeds = [
        # General Indian business / market feeds
        ("RBI",              "https://www.rbi.org.in/RSS/RBIRSSFeed.aspx?Id=316"),
        ("PIB",              "https://pib.gov.in/RSSNewsFeed.aspx?ModID=6"),
        ("ET Markets",       "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"),
        ("Moneycontrol",     "https://www.moneycontrol.com/rss/marketreports.xml"),
        ("LiveMint",         "https://www.livemint.com/rss/markets"),
        ("Business Std",     "https://www.business-standard.com/rss/markets-106.rss"),
        ("Hindu BizLine",    "https://www.thehindubusinessline.com/markets/feeder/default.rss"),
        ("Financial Exp",    "https://www.financialexpress.com/market/feed/"),
        ("NDTV Profit",      "https://feeds.feedburner.com/ndtvprofit-latest"),
        # ── Feature 1: NSE/BSE exchange filings — highest-signal source ──────
        ("NSE Board Mtgs",   "https://nsearchives.nseindia.com/content/RSS/boardMeetings.xml"),
        ("NSE Qtly Results", "https://nsearchives.nseindia.com/content/RSS/quarterlyResults.xml"),
        ("BSE Corp Actions", "https://www.bseindia.com/Rss/RssXml.aspx?Type=31"),
    ]

    # Per-feed timeout: fetch via requests (hard 8 s limit) then parse bytes.
    # feedparser.parse(url) uses urllib with NO timeout — one slow server
    # (NSE/BSE archives are known offenders) can hang the thread forever.
    # This was the root cause of the stuck spinner.
    _HEADERS      = {"User-Agent": "Mozilla/5.0 (compatible; feedparser/6.0)"}
    _FEED_TIMEOUT = 8    # seconds per individual feed HTTP request
    _POOL_TIMEOUT = 20   # seconds for ALL feeds combined

    def _fetch_one(name_url):
        name, url = name_url
        try:
            r    = requests.get(url, timeout=_FEED_TIMEOUT, headers=_HEADERS)
            r.raise_for_status()
            feed = feedparser.parse(r.content)   # parse raw bytes, NOT the URL
            items = []
            for e in feed.entries[:12]:
                title = getattr(e, "title", "").strip()
                if title:
                    items.append({
                        "title":     title,
                        "source":    name,
                        "url":       getattr(e, "link", ""),
                        "published": getattr(e, "published",
                                             getattr(e, "updated", "")),
                        "feed":      "RSS",
                    })
            return items
        except Exception:
            return []   # any timeout or HTTP error: skip silently

    from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeout
    results = []
    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = {pool.submit(_fetch_one, f): f for f in feeds}
        try:
            for fut in as_completed(futures, timeout=_POOL_TIMEOUT):
                try:
                    results.extend(fut.result())
                except Exception:
                    pass
        except FuturesTimeout:
            # Collect whatever finished within the budget; drop the rest.
            for fut in futures:
                if fut.done() and not fut.cancelled():
                    try:
                        results.extend(fut.result())
                    except Exception:
                        pass
    return results

@st.cache_data(ttl=300, show_spinner=False)
def fetch_freenews():
    if not FREENEWS_KEY:
        return []
    try:
        r = requests.get(
            "https://freenewsapi.io/api/v1/news",
            params={"country": "IN", "language": "en",
                    "category": "business", "pageSize": 20,
                    "apiKey": FREENEWS_KEY},
            timeout=15,
        )
        r.raise_for_status()
        return [
            {"title": a.get("title", ""),
             "source": a.get("source", {}).get("name", "FreeNews"),
             "url": a.get("url", ""), "published": a.get("publishedAt", ""),
             "feed": "FreeNewsAPI"}
            for a in r.json().get("articles", [])
            if a.get("title")
        ]
    except Exception:
        return []

# ── Pipeline ──────────────────────────────────────────────────────────────────

def dedupe(items: list) -> list:
    seen, out = set(), []
    for a in items:
        k = a["title"][:70].lower()
        if k not in seen:
            seen.add(k)
            out.append(a)
    return out

def load_all() -> list:
    # Run all four sources concurrently instead of sequentially.
    # Each is already @st.cache_data so a cache-hit returns instantly;
    # on a cache-miss all four network calls fire at the same time.
    from concurrent.futures import ThreadPoolExecutor
    sources = [fetch_newsapi, fetch_finnhub, fetch_rss, fetch_freenews]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(fn) for fn in sources]
        raw = []
        for fut in futures:
            try:
                raw.extend(fut.result())
            except Exception:
                pass
    raw = dedupe(raw)
    raw = [a for a in raw if within_last_24h(a)]
    raw.sort(
        key=lambda x: parse_pub_dt(x.get("published", ""))
                      or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )
    return raw[:150]

BATCH_SIZE = 16   # tune up if you have more RAM; 16 is safe on 2 GB

def score_articles(articles: list) -> list:
    """
    Two-pass scoring — entirely free, no external API:

    Pass 1 — FinBERT batch inference (unchanged from original).
    Pass 2 — rules_override(): scans each headline for unambiguous
              market-direction phrases and replaces FinBERT's label when
              a match is found.  The 'scorer' field records which pass
              produced the final label so the UI can badge overrides.
    """
    titles  = [a["title"][:512] for a in articles]
    results = []   # (label, conf, scorer) per article

    with torch.no_grad():
        for i in range(0, len(titles), BATCH_SIZE):
            batch  = titles[i : i + BATCH_SIZE]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding=True,
            )
            logits = model(**inputs).logits
            probs  = torch.softmax(logits, dim=-1)
            idxs   = torch.argmax(probs, dim=-1)
            for j in range(len(batch)):
                idx   = int(idxs[j].item())
                label = ID2LABEL[idx]
                conf  = float(probs[j][idx].item())
                results.append([label, round(conf, 3), "finbert"])

    # ── Pass 2: free rules-based override ────────────────────────────────────
    for i, (title, (label, conf, _)) in enumerate(zip(titles, results)):
        final_label, scorer = rules_override(title, label)
        if scorer != "finbert":               # override fired
            results[i] = [final_label, conf, scorer]

    # ── Assemble final article dicts ──────────────────────────────────────────
    scored = []
    for a, (label, conf, scorer) in zip(articles, results):
        scored.append({
            **a,
            "sentiment":  label,
            "confidence": conf,
            "score":      SCORE_MAP[label],
            "sector":     tag_sector(a["title"]),
            "scorer":     scorer,
        })
    return scored

@st.cache_data(ttl=300, show_spinner=False)
def get_fresh_data():
    """Full fetch → dedupe → 24-h filter → FinBERT score pipeline.
    @st.cache_data(ttl=300) ensures FinBERT inference runs at most once per
    5-minute window — not on every Streamlit rerun."""
    arts    = score_articles(load_all())
    scores  = [a["score"] for a in arts]
    avg     = round(sum(scores) / len(scores), 3) if scores else 0.0
    overall = "Bullish" if avg > 0.15 else ("Bearish" if avg < -0.15 else "Neutral")
    return arts, avg, overall

# ── Feature 6: news-bias bridge ───────────────────────────────────────────────

def write_news_bias(arts: list, avg: float, overall: str) -> dict:
    """
    Persists the current market-sentiment snapshot in two places:

    1. /tmp/news_bias.json  — works for co-located processes (local / same container).
    2. Upstash Redis key "news_bias" (TTL 10 min) — survives container recycles and
       works across Streamlit Cloud, so the options dashboard always finds fresh data.

    ── Reading from Redis in your options dashboard ─────────────────────────────
        import requests, json, os
        def get_news_bias() -> dict:
            url   = os.environ["UPSTASH_REDIS_URL"]
            token = os.environ["UPSTASH_REDIS_TOKEN"]
            try:
                r = requests.get(f"{url}/get/news_bias",
                                 headers={"Authorization": f"Bearer {token}"},
                                 timeout=3)
                raw = r.json().get("result")
                return json.loads(raw) if raw else {}
            except Exception:
                return {}

        bias = get_news_bias()
        news_score  = bias.get("avg_score", 0.0)      # float -1.0 … +1.0
        news_label  = bias.get("overall", "Neutral")  # Bullish/Neutral/Bearish
        sector_map  = bias.get("sector_sentiment", {}) # {sector: score}
        bull_pct    = bias.get("bull_pct", 0.0)
        bear_pct    = bias.get("bear_pct", 0.0)
    ─────────────────────────────────────────────────────────────────────────────
    """
    n      = len(arts)
    bull_n = sum(1 for a in arts if a["sentiment"] == "positive")
    bear_n = sum(1 for a in arts if a["sentiment"] == "negative")

    smap: dict = {}
    for a in arts:
        smap.setdefault(a.get("sector", "General"), []).append(a["score"])
    sector_sentiment = {s: round(sum(v) / len(v), 3) for s, v in smap.items() if v}

    payload = {
        "avg_score":        avg,
        "overall":          overall,
        "bull_pct":         round(bull_n / n * 100, 1) if n else 0.0,
        "bear_pct":         round(bear_n / n * 100, 1) if n else 0.0,
        "article_count":    n,
        "sector_sentiment": sector_sentiment,
        "market_session":   session_label(),
        "updated_ist":      fmt_ist(),
    }

    # ── 1. Local file (co-located processes) ──────────────────────────────────
    try:
        NEWS_BIAS_PATH.write_text(json.dumps(payload, indent=2))
    except Exception:
        pass

    # ── 2. Upstash Redis (cross-container / Streamlit Cloud) ──────────────────
    if REDIS_URL and REDIS_TOKEN:
        try:
            requests.post(
                f"{REDIS_URL}/setex/news_bias/600/"
                + requests.utils.quote(json.dumps(payload), safe=""),
                headers=_redis_hdrs(), timeout=5,
            )
        except Exception:
            pass

    return payload

# ── Session-state bootstrap ───────────────────────────────────────────────────

_defaults = {
    "owner_ok":           False,
    "prev_score":         0.0,
    "prev_sentiment":     None,
    "articles":           [],
    "avg_score":          0.0,
    "overall":            "Neutral",
    "change":             "—",
    "last_updated":       fmt_ist(),
    "last_refresh_count": -1,
    "score_history":      load_history_from_redis(),  # seeded from Redis on cold load
    "bias_payload":       {},     # Feature 6: last written JSON payload
    "auto_refresh_chk":   True,   # Feature 2: auto-mode on by default
    "manual_interval":    5,      # Feature 2: manual override value (minutes)
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## Owner Mode")
    pwd = st.text_input("Password", type="password")
    if st.button("Unlock"):
        st.session_state.owner_ok = bool(OWNER_PASSWORD) and pwd == OWNER_PASSWORD

    if st.session_state.owner_ok:
        st.success("Owner mode enabled")

        # Feature 2: auto vs manual refresh controls
        st.markdown("**Refresh Interval**")
        st.checkbox("Auto (market-hours aware)", key="auto_refresh_chk")
        if not st.session_state.auto_refresh_chk:
            _opts = [2, 5, 10, 15, 30, 60]
            _cur  = st.session_state.manual_interval
            st.session_state.manual_interval = st.selectbox(
                "Manual frequency (minutes)", _opts,
                index=_opts.index(_cur) if _cur in _opts else 1,
            )
        else:
            st.caption(
                f"Auto-detected: **{market_refresh_interval()} min** · {session_label()}"
            )

        if st.button("SOS Manual Refresh"):
            # Clearing cache + resetting counter causes needs_load=True on rerun.
            # get_fresh_data() re-runs fresh; no double-scoring since cache was cleared.
            st.cache_data.clear()
            st.session_state.last_refresh_count = -1
            st.rerun()

        # Feature 6: live bridge payload viewer
        if st.session_state.bias_payload:
            with st.expander("📡 Dashboard Bridge Payload", expanded=False):
                st.code(
                    json.dumps(st.session_state.bias_payload, indent=2),
                    language="json",
                )
                st.caption(f"Written to `{NEWS_BIAS_PATH}`")
    else:
        st.info("Read-only mode")

# ── Feature 2: resolve effective refresh interval (after sidebar widget state) ──

auto_interval = market_refresh_interval()
eff_interval  = (
    auto_interval
    if st.session_state.auto_refresh_chk
    else st.session_state.manual_interval
)

# ── Auto-refresh ──────────────────────────────────────────────────────────────
# refresh_count increments on every timer tick; comparing to last stored value
# is the only reliable way to detect that a real tick (not a user interaction
# rerun) has fired.

refresh_count = st_autorefresh(
    interval=eff_interval * 60 * 1000,
    key="refresh_timer",
)

needs_load = (
    not st.session_state.articles                               # initial load
    or refresh_count != st.session_state.last_refresh_count    # timer tick
)

if needs_load:
    with st.spinner("Fetching and scoring market news — last 24 hours…"):
        arts, avg, overall = get_fresh_data()

    is_first = not st.session_state.articles
    delta    = round(avg - st.session_state.prev_score, 3)

    if is_first:
        change = "Initial load"
    elif st.session_state.prev_sentiment and overall != st.session_state.prev_sentiment:
        change = f"{st.session_state.prev_sentiment} → {overall} ({delta:+.3f})"
    else:
        change = f"{delta:+.3f}"

    # Feature 3: append to rolling timeline; keep last 12 hours only
    st.session_state.score_history.append((now_ist(), avg, overall))
    _cutoff = now_ist() - timedelta(hours=12)
    st.session_state.score_history = [
        row for row in st.session_state.score_history
        if row[0] >= _cutoff
    ]

    st.session_state.articles           = arts
    st.session_state.avg_score          = avg
    st.session_state.overall            = overall
    st.session_state.change             = change
    st.session_state.prev_score         = avg
    st.session_state.prev_sentiment     = overall
    st.session_state.last_updated       = fmt_ist()
    st.session_state.last_refresh_count = refresh_count

    # Feature 6: write bridge JSON + mirror to Redis after every data refresh
    st.session_state.bias_payload = write_news_bias(arts, avg, overall)

    # Persist rolling history to Redis so cold loads start with data
    save_history_to_redis(st.session_state.score_history)

# ── CSS ───────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  .stApp { background: white; color: #111; }
  [data-testid="stSidebar"] { background: #fafafa; border-right: 1px solid #e5e7eb; }
  .card {
    background: white; border: 1px solid #e5e7eb;
    border-left-width: 5px; border-radius: 12px;
    padding: 14px 16px; margin-bottom: 10px;
  }
  .bull  { border-left-color: #22c55e; }
  .bear  { border-left-color: #ef4444; }
  .neut  { border-left-color: #f59e0b; }
  .hot-panel {
    background: #fff7ed; border: 2px solid #f97316;
    border-radius: 12px; padding: 12px 16px; margin-bottom: 14px;
  }
  .badge {
    display: inline-block; padding: 2px 8px;
    border-radius: 999px; font-size: 12px; font-weight: 700;
  }
  .badge-bull   { background: #ecfdf5; color: #15803d; }
  .badge-bear   { background: #fef2f2; color: #b91c1c; }
  .badge-neut   { background: #fffbeb; color: #b45309; }
  .badge-sector { background: #f0f9ff; color: #0369a1; }
  .badge-override { background: #fefce8; color: #854d0e; }
</style>
""", unsafe_allow_html=True)

# ── Local convenience refs ────────────────────────────────────────────────────

articles = st.session_state.articles
avg      = st.session_state.avg_score
overall  = st.session_state.overall
bull     = [a for a in articles if a["sentiment"] == "positive"]
bear     = [a for a in articles if a["sentiment"] == "negative"]
neut     = [a for a in articles if a["sentiment"] == "neutral"]

# ── Header & metrics ──────────────────────────────────────────────────────────

st.markdown("### Shantanu's Market Update")

col_t, col_s = st.columns([7, 1])
with col_t:
    st.markdown(f"**LAST UPDATED AT:** {st.session_state.last_updated}")
    st.caption(
        f"Last 24 h · Auto-refresh every **{eff_interval} min** · {session_label()}"
    )
with col_s:
    mkt_clr = "#22c55e" if is_market_hours() else "#6b7280"
    st.markdown(
        f"<div style='text-align:right;margin-top:6px;'>"
        f"<span style='color:{mkt_clr};font-weight:700;font-size:13px;'>"
        f"{session_label()}</span></div>",
        unsafe_allow_html=True,
    )

c1, c2, c3, c4 = st.columns(4)
c1.metric("Overall",  overall)
c2.metric("Score",    f"{avg:+.3f}")
c3.metric("Change",   st.session_state.change)
c4.metric("Articles", len(articles))

# ── Feature 4: High-conviction alert panel ────────────────────────────────────
# Shows only articles where FinBERT confidence > 82% AND published in the last
# 20 minutes — the "stop-what-you're-doing" items during market hours.

_hot_cutoff = now_ist() - timedelta(minutes=20)
hot_news    = [
    a for a in articles
    if a["confidence"] > 0.82
    and (parse_pub_dt(a.get("published", ""))
         or datetime.min.replace(tzinfo=timezone.utc)) >= _hot_cutoff
]

if hot_news:
    st.markdown(
        f"<div class='hot-panel'>"
        f"<b>🚨 HIGH-CONVICTION ALERTS</b>&nbsp;&nbsp;"
        f"<span style='font-size:13px;color:#92400e'>"
        f"{len(hot_news)} item(s) · published last 20 min · FinBERT conf &gt; 82%"
        f"</span></div>",
        unsafe_allow_html=True,
    )
    for a in hot_news:
        cls    = ("bull" if a["sentiment"] == "positive"
                  else "bear" if a["sentiment"] == "negative" else "neut")
        icon   = "▲" if cls == "bull" else "▼" if cls == "bear" else "●"
        ts     = parse_pub_dt(a.get("published", ""))
        ts_txt = ts.strftime("%I:%M %p IST") if ts else "—"
        ovr    = ("<span class='badge badge-override' style='margin-left:6px;'"
                  " title='FinBERT overridden'>⚡ rules</span>"
                  if a.get("scorer", "finbert") != "finbert" else "")
        # Single-line concatenation — no blank lines that confuse st.markdown
        html = (
            f'<div class="card {cls}" style="margin-bottom:6px;">'
            f'<a href="{a["url"]}" target="_blank"><b>{a["title"]}</b></a><br>'
            f'<span class="badge badge-{cls}">{icon} {a["sentiment"]}</span>'
            f'<span class="badge badge-sector" style="margin-left:6px;">{a.get("sector","General")}</span>'
            + ovr +
            f'<span style="margin-left:8px;color:#6b7280">'
            f'{a["source"]} · {ts_txt} · conf {int(a["confidence"]*100)}%'
            f'</span></div>'
        )
        st.markdown(html, unsafe_allow_html=True)

# ── Features 3 & 5: Analytics charts row ─────────────────────────────────────

ch_l, ch_r = st.columns(2)

# ── Feature 3: Rolling intraday sentiment timeline ────────────────────────────

with ch_l:
    history = st.session_state.score_history
    if len(history) >= 2:
        _times  = [t for t, _, _ in history]
        _scores = [s for _, s, _ in history]
        _labels = [l for _, _, l in history]
        _clr    = ["#22c55e" if s > 0.15 else "#ef4444" if s < -0.15 else "#f59e0b"
                   for s in _scores]

        fig_tl = go.Figure()
        # Coloured background bands for bullish/bearish zones
        fig_tl.add_hrect(y0=0.15,  y1=1.0,  fillcolor="#dcfce7", opacity=0.18, line_width=0)
        fig_tl.add_hrect(y0=-1.0,  y1=-0.15, fillcolor="#fee2e2", opacity=0.18, line_width=0)
        # Threshold reference lines
        fig_tl.add_hline(y=0,      line_dash="dot", line_color="#9ca3af", line_width=1)
        fig_tl.add_hline(y=0.15,   line_dash="dot", line_color="#22c55e", line_width=0.8)
        fig_tl.add_hline(y=-0.15,  line_dash="dot", line_color="#ef4444", line_width=0.8)
        fig_tl.add_trace(go.Scatter(
            x=_times, y=_scores,
            mode="lines+markers",
            line=dict(color="#6366f1", width=2.5),
            marker=dict(color=_clr, size=9,
                        line=dict(color="white", width=1.5)),
            text=_labels,
            hovertemplate="%{x|%H:%M IST}<br>Score: %{y:.3f} · %{text}<extra></extra>",
        ))
        fig_tl.update_layout(
            height=240,
            margin=dict(l=0, r=10, t=32, b=0),
            plot_bgcolor="white", paper_bgcolor="white",
            title=dict(text="Intraday Sentiment Score  (12 h rolling)",
                       font=dict(size=13, color="#374151")),
            xaxis=dict(showgrid=False, title=""),
            yaxis=dict(showgrid=True, gridcolor="#f3f4f6",
                       range=[-1.05, 1.05], title="Score"),
        )
        st.plotly_chart(fig_tl, use_container_width=True)
    else:
        st.info("Sentiment timeline builds after 2+ refreshes.")

# ── Feature 5: Sector sentiment bar chart ────────────────────────────────────

with ch_r:
    _smap: dict = {}
    for a in articles:
        _sec = a.get("sector", "General")
        _smap.setdefault(_sec, {"scores": [], "count": 0})
        _smap[_sec]["scores"].append(a["score"])
        _smap[_sec]["count"] += 1

    if _smap:
        _sects = sorted(
            _smap.keys(),
            key=lambda x: sum(_smap[x]["scores"]) / len(_smap[x]["scores"]),
            reverse=True,
        )
        _avgs  = [round(sum(_smap[s]["scores"]) / len(_smap[s]["scores"]), 3)
                  for s in _sects]
        _cnts  = [_smap[s]["count"] for s in _sects]
        _bclr  = ["#22c55e" if v > 0.15 else "#ef4444" if v < -0.15 else "#f59e0b"
                  for v in _avgs]

        fig_sc = go.Figure(go.Bar(
            x=_sects, y=_avgs,
            marker_color=_bclr,
            text=[f"{v:+.2f} ({c})" for v, c in zip(_avgs, _cnts)],
            textposition="outside",
            hovertemplate="%{x}<br>Avg Score: %{y:.3f}<extra></extra>",
        ))
        fig_sc.add_hline(y=0, line_dash="dot", line_color="#9ca3af", line_width=1)
        fig_sc.update_layout(
            height=240,
            margin=dict(l=0, r=10, t=32, b=0),
            plot_bgcolor="white", paper_bgcolor="white",
            title=dict(text="Sector Sentiment Breakdown",
                       font=dict(size=13, color="#374151")),
            xaxis=dict(showgrid=False, tickangle=-25),
            yaxis=dict(showgrid=True, gridcolor="#f3f4f6",
                       range=[-1.2, 1.4], title="Score"),
        )
        st.plotly_chart(fig_sc, use_container_width=True)

# ── Headlines with sector filter ──────────────────────────────────────────────

col_h, col_f = st.columns([5, 2])
with col_h:
    st.write("### Latest Headlines")
with col_f:
    _sectors_present = sorted(set(a.get("sector", "General") for a in articles))
    sel_sector = st.selectbox(
        "Sector filter",
        ["All Sectors"] + _sectors_present,
        key="sector_filter",
        label_visibility="collapsed",
    )

filtered = (
    articles if sel_sector == "All Sectors"
    else [a for a in articles if a.get("sector") == sel_sector]
)
f_bull = [a for a in filtered if a["sentiment"] == "positive"]
f_bear = [a for a in filtered if a["sentiment"] == "negative"]
f_neut = [a for a in filtered if a["sentiment"] == "neutral"]

tabs = st.tabs([
    f"All ({len(filtered)})",
    f"Bullish ({len(f_bull)})",
    f"Neutral ({len(f_neut)})",
    f"Bearish ({len(f_bear)})",
])

def render(items: list):
    if not items:
        st.caption("No articles in this view.")
        return
    for a in items:
        cls    = ("bull" if a["sentiment"] == "positive"
                  else "bear" if a["sentiment"] == "negative" else "neut")
        icon   = "▲" if cls == "bull" else "▼" if cls == "bear" else "●"
        ts     = parse_pub_dt(a.get("published", ""))
        ts_txt = ts.strftime("%d %b %Y, %I:%M %p IST") if ts else "Time unavailable"
        sector = a.get("sector", "General")
        ovr    = ("<span class='badge badge-override' style='margin-left:6px;'"
                  " title='FinBERT overridden'>⚡ rules</span>"
                  if a.get("scorer", "finbert") != "finbert" else "")
        # Single-line concatenation — no blank lines that confuse st.markdown
        html = (
            f'<div class="card {cls}">'
            f'<a href="{a["url"]}" target="_blank"><b>{a["title"]}</b></a><br>'
            f'<span class="badge badge-{cls}">{icon} {a["sentiment"]}</span>'
            f'<span class="badge badge-sector" style="margin-left:6px;">{sector}</span>'
            + ovr +
            f'<span style="margin-left:8px;color:#6b7280">'
            f'{a["source"]} · {a["feed"]} · {ts_txt} · conf {int(a["confidence"]*100)}%'
            f'</span></div>'
        )
        st.markdown(html, unsafe_allow_html=True)

with tabs[0]: render(filtered)
with tabs[1]: render(f_bull)
with tabs[2]: render(f_neut)
with tabs[3]: render(f_bear)
