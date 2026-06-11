import streamlit as st
import requests
import feedparser
import pytz
import torch
from datetime import datetime, timedelta, timezone
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="Shantanu's Market Update", page_icon="📈", layout="wide")

IST        = pytz.timezone("Asia/Kolkata")
MODEL_NAME = "ProsusAI/finbert"

NEWSAPI_KEY    = st.secrets.get("NEWSAPI_KEY", "")
FINNHUB_KEY    = st.secrets.get("FINNHUB_KEY", "")
FREENEWS_KEY   = st.secrets.get("FREENEWS_KEY", "")
OWNER_PASSWORD = st.secrets.get("OWNER_PASSWORD", "")

# ── Model (loaded once, shared across all sessions) ───────────────────────────

@st.cache_resource
def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    mdl.eval()
    return tok, mdl

tokenizer, model = load_model()

SCORE_MAP = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
ID2LABEL  = {0: "negative", 1: "neutral", 2: "positive"}

# ── Time helpers ──────────────────────────────────────────────────────────────

def now_ist():
    return datetime.now(IST)

def fmt_ist(dt=None):
    dt = dt or now_ist()
    return dt.strftime("%d %b %Y, %I:%M %p IST")

def parse_pub_dt(s: str):
    """
    Robustly convert a publication-date string to an IST-aware datetime.

    Bugs fixed vs original:
    • Removed the s[:25] slice that truncated RFC-2822 dates like
      "Thu, 11 Jun 2026 07:30:00 +0530" (31 chars) to an unparseable stub.
    • Normalised "GMT" → "+0000" because strptime does not recognise "GMT"
      as a timezone name.
    • Added dateutil fallback that handles virtually every real-world format.
    """
    if not s:
        return None
    s = s.strip()
    # strptime cannot handle the literal string "GMT" as a %z value
    s_norm = s.replace(" GMT", " +0000")

    formats = [
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%S",
        "%a, %d %b %Y %H:%M:%S %z",
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

    # Last resort: python-dateutil handles almost any format
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

# ── Data sources ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def fetch_newsapi():
    if not NEWSAPI_KEY:
        return []
    try:
        r = requests.get(
            "https://newsapi.org/v2/top-headlines",
            params={
                "country": "in",
                "category": "business",
                "pageSize": 50,          # was 15 — too few for 24 h
                "apiKey": NEWSAPI_KEY,
            },
            timeout=15,
        )
        r.raise_for_status()
        return [
            {
                "title":     a.get("title", ""),
                "source":    a.get("source", {}).get("name", "NewsAPI"),
                "url":       a.get("url", ""),
                "published": a.get("publishedAt", ""),
                "feed":      "NewsAPI",
            }
            for a in r.json().get("articles", [])
            # Free-tier NewsAPI returns placeholder "[Removed]" titles for
            # paywalled articles — filter them out
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
        # Bug fixed: r.json() was called twice (two JSON deserialisations)
        data  = r.json()
        items = data if isinstance(data, list) else []
        out   = []
        for a in items[:40]:                 # was 15
            if a.get("headline"):
                out.append({
                    "title":     a["headline"],
                    "source":    a.get("source", "Finnhub"),
                    "url":       a.get("url", ""),
                    # Use UTC explicitly; IST is a DST-aware zone and .isoformat()
                    # from pytz-localised datetimes is fine, but UTC is cleaner here
                    "published": datetime.fromtimestamp(
                        a["datetime"], tz=timezone.utc
                    ).isoformat(),
                    "feed":      "Finnhub",
                })
        return out
    except Exception:
        return []

@st.cache_data(ttl=300, show_spinner=False)
def fetch_rss():
    feeds = [
        ("RBI",           "https://www.rbi.org.in/RSS/RBIRSSFeed.aspx?Id=316"),
        ("PIB",           "https://pib.gov.in/RSSNewsFeed.aspx?ModID=6"),
        ("ET Markets",    "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"),
        ("Moneycontrol",  "https://www.moneycontrol.com/rss/marketreports.xml"),
        ("LiveMint",      "https://www.livemint.com/rss/markets"),
        # Added for better 24-h Indian market coverage
        ("Business Std",  "https://www.business-standard.com/rss/markets-106.rss"),
        ("Hindu BizLine", "https://www.thehindubusinessline.com/markets/feeder/default.rss"),
        ("Financial Exp", "https://www.financialexpress.com/market/feed/"),
        ("NDTV Profit",   "https://feeds.feedburner.com/ndtvprofit-latest"),
    ]
    results = []
    for name, url in feeds:
        try:
            feed = feedparser.parse(url)
            for e in feed.entries[:12]:     # was 5 — too few for 24 h
                title = getattr(e, "title", "").strip()
                if title:
                    results.append({
                        "title":     title,
                        "source":    name,
                        "url":       getattr(e, "link", ""),
                        "published": getattr(e, "published",
                                             getattr(e, "updated", "")),
                        "feed":      "RSS",
                    })
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
            params={
                "country":  "IN",
                "language": "en",
                "category": "business",
                "pageSize": 20,
                "apiKey":   FREENEWS_KEY,
            },
            timeout=15,
        )
        r.raise_for_status()
        return [
            {
                "title":     a.get("title", ""),
                "source":    a.get("source", {}).get("name", "FreeNews"),
                "url":       a.get("url", ""),
                "published": a.get("publishedAt", ""),
                "feed":      "FreeNewsAPI",
            }
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
    raw = fetch_newsapi() + fetch_finnhub() + fetch_rss() + fetch_freenews()
    raw = dedupe(raw)
    raw = [a for a in raw if within_last_24h(a)]
    raw.sort(
        # Bug fixed: datetime.min.replace(tzinfo=IST) is wrong — pytz zones must
        # never be attached via .replace(); use a plain UTC sentinel instead.
        key=lambda x: parse_pub_dt(x.get("published", ""))
                      or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )
    return raw[:150]                         # was 60

def score_articles(articles: list) -> list:
    scored = []
    with torch.no_grad():
        for a in articles:
            inputs = tokenizer(
                a["title"][:512],
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            logits = model(**inputs).logits[0]
            probs  = torch.softmax(logits, dim=-1)
            idx    = int(torch.argmax(probs).item())
            label  = ID2LABEL[idx]
            conf   = float(probs[idx].item())
            # Bug fixed: original code mutated dicts that came from @st.cache_data
            # functions, causing aliasing bugs on subsequent cached reads.
            # Now we always create a new dict.
            scored.append({
                **a,
                "sentiment":  label,
                "confidence": round(conf, 3),
                "score":      SCORE_MAP[label],
            })
    return scored

@st.cache_data(ttl=300, show_spinner=False)
def get_fresh_data():
    """
    Full pipeline: fetch → dedupe → 24-h filter → FinBERT score.
    Result cached for 5 min so FinBERT inference only runs on cache-miss,
    not on every Streamlit rerun.
    """
    arts    = score_articles(load_all())
    scores  = [a["score"] for a in arts]
    avg     = round(sum(scores) / len(scores), 3) if scores else 0.0
    overall = "Bullish" if avg > 0.15 else ("Bearish" if avg < -0.15 else "Neutral")
    return arts, avg, overall

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
    "refresh_minutes":    10,
    "last_refresh_count": -1,   # tracks st_autorefresh counter to detect real ticks
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
        st.session_state.refresh_minutes = st.selectbox(
            "Auto-refresh frequency (minutes)",
            [5, 10, 15, 30, 60],
            index=[5, 10, 15, 30, 60].index(st.session_state.refresh_minutes),
        )
        if st.button("SOS Manual Refresh"):
            st.cache_data.clear()
            arts, avg, overall = get_fresh_data()
            delta = round(avg - st.session_state.prev_score, 3)
            if st.session_state.prev_sentiment and overall != st.session_state.prev_sentiment:
                change = f"{st.session_state.prev_sentiment} → {overall} ({delta:+.3f})"
            else:
                change = f"{delta:+.3f}"
            st.session_state.articles           = arts
            st.session_state.avg_score          = avg
            st.session_state.overall            = overall
            st.session_state.change             = change
            st.session_state.prev_score         = avg
            st.session_state.prev_sentiment     = overall
            st.session_state.last_updated       = fmt_ist()
            st.session_state.last_refresh_count = -1   # reset so next tick is fresh
            st.rerun()
    else:
        st.info("Read-only mode")

# ── Auto-refresh ──────────────────────────────────────────────────────────────
# Bug fixed: original code ignored the return value of st_autorefresh, so a
# timed rerun re-rendered stale session-state data instead of fetching fresh
# articles.  We now compare the counter to detect a real tick.

refresh_count = st_autorefresh(
    interval=st.session_state.refresh_minutes * 60 * 1000,
    key="refresh_timer",
)

needs_load = (
    not st.session_state.articles                               # initial page load
    or refresh_count != st.session_state.last_refresh_count    # auto-refresh tick
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

    st.session_state.articles           = arts
    st.session_state.avg_score          = avg
    st.session_state.overall            = overall
    st.session_state.change             = change
    st.session_state.prev_score         = avg
    st.session_state.prev_sentiment     = overall
    st.session_state.last_updated       = fmt_ist()
    st.session_state.last_refresh_count = refresh_count

# ── CSS ───────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  .stApp { background: white; color: #111; }
  [data-testid="stSidebar"] { background: #fafafa; border-right: 1px solid #e5e7eb; }
  .card {
    background: white;
    border: 1px solid #e5e7eb;
    border-left-width: 5px;
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 10px;
  }
  .bull { border-left-color: #22c55e; }
  .bear { border-left-color: #ef4444; }
  .neut { border-left-color: #f59e0b; }
  .badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 700;
  }
  .badge-bull { background: #ecfdf5; color: #15803d; }
  .badge-bear { background: #fef2f2; color: #b91c1c; }
  .badge-neut { background: #fffbeb; color: #b45309; }
</style>
""", unsafe_allow_html=True)

# ── Main UI ───────────────────────────────────────────────────────────────────

articles = st.session_state.articles
avg      = st.session_state.avg_score
overall  = st.session_state.overall

bull = [a for a in articles if a["sentiment"] == "positive"]
bear = [a for a in articles if a["sentiment"] == "negative"]
neut = [a for a in articles if a["sentiment"] == "neutral"]

st.markdown("### Shantanu's Market Update")
st.markdown(f"**LAST UPDATED AT:** {st.session_state.last_updated}")
st.caption("All timestamps shown in IST · Showing last 24 hours of news")

c1, c2, c3 = st.columns(3)
c1.metric("Overall", overall)
c2.metric("Score",   f"{avg:+.3f}")
c3.metric("Change",  st.session_state.change)

st.write("### Latest Headlines")

tabs = st.tabs([
    f"All ({len(articles)})",
    f"Bullish ({len(bull)})",
    f"Neutral ({len(neut)})",
    f"Bearish ({len(bear)})",
])

def render(items):
    for a in items:
        cls   = "bull" if a["sentiment"] == "positive" else ("bear" if a["sentiment"] == "negative" else "neut")
        badge = "badge-bull" if cls == "bull" else ("badge-bear" if cls == "bear" else "badge-neut")
        icon  = "▲" if cls == "bull" else ("▼" if cls == "bear" else "●")
        ts    = parse_pub_dt(a.get("published", ""))
        ts_txt = ts.strftime("%d %b %Y, %I:%M %p IST") if ts else "Time unavailable"
        st.markdown(f"""
        <div class="card {cls}">
          <a href="{a['url']}" target="_blank"><b>{a['title']}</b></a><br>
          <span class="badge {badge}">{icon} {a['sentiment']}</span>
          <span style="margin-left:8px;color:#6b7280">
            {a['source']} · {a['feed']} · {ts_txt} · conf {int(a['confidence'] * 100)}%
          </span>
        </div>
        """, unsafe_allow_html=True)

with tabs[0]: render(articles)
with tabs[1]: render(bull)
with tabs[2]: render(neut)
with tabs[3]: render(bear)
