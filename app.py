import streamlit as st
import requests, feedparser, pytz, torch
from datetime import datetime, timedelta, timezone
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="Shantanu's Market Update", page_icon="📈", layout="wide")

IST = pytz.timezone("Asia/Kolkata")
MODEL_NAME = "ProsusAI/finbert"

NEWSAPI_KEY = st.secrets.get("NEWSAPI_KEY", "")
FINNHUB_KEY = st.secrets.get("FINNHUB_KEY", "")
FREENEWS_KEY = st.secrets.get("FREENEWS_KEY", "")
OWNER_PASSWORD = st.secrets.get("OWNER_PASSWORD", "")

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

SCORE_MAP = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}


def now_ist():
    return datetime.now(IST)


def fmt_ist(dt=None):
    dt = dt or now_ist()
    return dt.strftime("%d %b %Y, %I:%M %p IST")


def parse_pub_dt(s):
    if not s:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S", "%a, %d %b %Y %H:%M:%S %z", "%a, %d %b %Y %H:%M:%S GMT"):
        try:
            dt = datetime.strptime(s[:25].strip(), fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(IST)
        except:
            pass
    return None


def within_last_3_hours(article):
    dt = parse_pub_dt(article.get("published", ""))
    return dt is not None and dt >= now_ist() - timedelta(hours=3)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_newsapi():
    try:
        r = requests.get(
            "https://newsapi.org/v2/top-headlines",
            params={"country": "in", "category": "business", "pageSize": 15, "apiKey": NEWSAPI_KEY},
            timeout=15,
        )
        items = r.json().get("articles", [])
        return [
            {"title": a.get("title", ""), "source": a.get("source", {}).get("name", "NewsAPI"),
             "url": a.get("url", ""), "published": a.get("publishedAt", ""), "feed": "NewsAPI"}
            for a in items if a.get("title")
        ]
    except:
        return []


@st.cache_data(ttl=300, show_spinner=False)
def fetch_finnhub():
    try:
        r = requests.get(
            "https://finnhub.io/api/v1/news",
            params={"category": "general", "token": FINNHUB_KEY},
            timeout=15,
        )
        items = r.json() if isinstance(r.json(), list) else []
        out = []
        for a in items[:15]:
            if a.get("headline"):
                out.append({
                    "title": a["headline"],
                    "source": a.get("source", "Finnhub"),
                    "url": a.get("url", ""),
                    "published": datetime.fromtimestamp(a["datetime"], tz=IST).isoformat(),
                    "feed": "Finnhub",
                })
        return out
    except:
        return []


@st.cache_data(ttl=300, show_spinner=False)
def fetch_rss():
    feeds = [
        ("RBI", "https://www.rbi.org.in/RSS/RBIRSSFeed.aspx?Id=316"),
        ("PIB", "https://pib.gov.in/RSSNewsFeed.aspx?ModID=6"),
        ("ET Markets", "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"),
        ("Moneycontrol", "https://www.moneycontrol.com/rss/marketreports.xml"),
        ("LiveMint", "https://www.livemint.com/rss/markets"),
    ]
    results = []
    for name, url in feeds:
        try:
            feed = feedparser.parse(url)
            for e in feed.entries[:5]:
                results.append({
                    "title": getattr(e, "title", ""),
                    "source": name,
                    "url": getattr(e, "link", ""),
                    "published": getattr(e, "published", getattr(e, "updated", "")),
                    "feed": "RSS",
                })
        except:
            pass
    return results


@st.cache_data(ttl=300, show_spinner=False)
def fetch_freenews():
    try:
        r = requests.get(
            "https://freenewsapi.io/api/v1/news",
            params={"country": "IN", "language": "en", "category": "business", "pageSize": 10, "apiKey": FREENEWS_KEY},
            timeout=15,
        )
        items = r.json().get("articles", [])
        return [
            {"title": a.get("title", ""), "source": a.get("source", {}).get("name", "FreeNews"),
             "url": a.get("url", ""), "published": a.get("publishedAt", ""), "feed": "FreeNewsAPI"}
            for a in items if a.get("title")
        ]
    except:
        return []


def dedupe(items):
    seen, out = set(), []
    for a in items:
        k = a["title"][:70].lower()
        if k not in seen:
            seen.add(k)
            out.append(a)
    return out


def load_all():
    raw = fetch_newsapi() + fetch_finnhub() + fetch_rss() + fetch_freenews()
    raw = dedupe(raw)
    raw = [a for a in raw if within_last_3_hours(a)]
    raw.sort(key=lambda x: parse_pub_dt(x.get("published", "")) or datetime.min.replace(tzinfo=IST), reverse=True)
    return raw[:60]


def score_articles(articles):
    scored = []
    with torch.no_grad():
        for a in articles:
            inputs = tokenizer(a["title"][:512], return_tensors="pt", truncation=True, max_length=512)
            logits = model(**inputs).logits[0]
            probs = torch.softmax(logits, dim=-1)
            idx = int(torch.argmax(probs).item())
            label = ID2LABEL[idx]
            conf = float(probs[idx].item())
            a["sentiment"] = label
            a["confidence"] = round(conf, 3)
            a["score"] = SCORE_MAP[label]
            scored.append(a)
    return scored


def refresh_logic():
    arts = score_articles(load_all())
    scores = [a["score"] for a in arts]
    avg = round(sum(scores) / len(scores), 3) if scores else 0.0
    if avg > 0.15:
        overall = "Bullish"
    elif avg < -0.15:
        overall = "Bearish"
    else:
        overall = "Neutral"
    return arts, avg, overall


if "owner_ok" not in st.session_state:
    st.session_state.owner_ok = False
if "prev_score" not in st.session_state:
    st.session_state.prev_score = 0.0
    st.session_state.prev_sentiment = None
    st.session_state.articles = []
    st.session_state.avg_score = 0.0
    st.session_state.overall = "Neutral"
    st.session_state.change = "—"
    st.session_state.last_updated = fmt_ist()
if "refresh_minutes" not in st.session_state:
    st.session_state.refresh_minutes = 10

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
            arts, avg, overall = refresh_logic()
            delta = round(avg - st.session_state.prev_score, 3)
            if st.session_state.prev_sentiment and overall != st.session_state.prev_sentiment:
                change = f"{st.session_state.prev_sentiment} → {overall} ({delta:+.3f})"
            else:
                change = f"{delta:+.3f}"
            st.session_state.articles = arts
            st.session_state.avg_score = avg
            st.session_state.overall = overall
            st.session_state.change = change
            st.session_state.prev_score = avg
            st.session_state.prev_sentiment = overall
            st.session_state.last_updated = fmt_ist()
            st.rerun()
    else:
        st.info("Read-only mode")

st_autorefresh(interval=st.session_state.refresh_minutes * 60 * 1000, key="refresh_timer")

st.markdown("""
<style>
  .stApp { background: white; color: #111; }
  [data-testid="stSidebar"] { background: #fafafa; border-right: 1px solid #e5e7eb; }
  .card { background: white; border: 1px solid #e5e7eb; border-left-width: 5px; border-radius: 12px; padding: 14px 16px; margin-bottom: 10px; }
  .bull { border-left-color: #22c55e; }
  .bear { border-left-color: #ef4444; }
  .neut { border-left-color: #f59e0b; }
  .badge { display:inline-block; padding: 2px 8px; border-radius: 999px; font-size: 12px; font-weight: 700; }
  .badge-bull { background:#ecfdf5; color:#15803d; }
  .badge-bear { background:#fef2f2; color:#b91c1c; }
  .badge-neut { background:#fffbeb; color:#b45309; }
</style>
""", unsafe_allow_html=True)

if not st.session_state.articles:
    with st.spinner(f"LAST UPDATED AT: {st.session_state.last_updated}"):
        arts, avg, overall = refresh_logic()
        st.session_state.articles = arts
        st.session_state.avg_score = avg
        st.session_state.overall = overall
        st.session_state.change = "Initial load"
        st.session_state.prev_score = avg
        st.session_state.prev_sentiment = overall
        st.session_state.last_updated = fmt_ist()

articles = st.session_state.articles
avg = st.session_state.avg_score
overall = st.session_state.overall

bull = [a for a in articles if a["sentiment"] == "positive"]
bear = [a for a in articles if a["sentiment"] == "negative"]
neut = [a for a in articles if a["sentiment"] == "neutral"]

st.markdown("### Shantanu's Market Update")
st.markdown(f"**LAST UPDATED AT:** {st.session_state.last_updated}")
st.caption("All timestamps, notifications, and refresh events are shown in IST.")

c1, c2, c3 = st.columns(3)
c1.metric("Overall", overall)
c2.metric("Score", f"{avg:+.3f}")
c3.metric("Change", st.session_state.change)

st.write("### Latest Headlines")

tabs = st.tabs([f"All ({len(articles)})", f"Bullish ({len(bull)})", f"Neutral ({len(neut)})", f"Bearish ({len(bear)})"])

def render(items):
    for a in items:
        cls = "bull" if a["sentiment"] == "positive" else "bear" if a["sentiment"] == "negative" else "neut"
        badge = "badge-bull" if cls == "bull" else "badge-bear" if cls == "bear" else "badge-neut"
        icon = "▲" if cls == "bull" else "▼" if cls == "bear" else "●"
        ts = parse_pub_dt(a.get("published", ""))
        ts_txt = ts.strftime("%d %b %Y, %I:%M %p IST") if ts else "Time unavailable"
        st.markdown(f"""
        <div class="card {cls}">
          <a href="{a['url']}" target="_blank"><b>{a['title']}</b></a><br>
          <span class="badge {badge}">{icon} {a['sentiment']}</span>
          <span style="margin-left:8px;color:#6b7280">{a['source']} · {a['feed']} · {ts_txt} · conf {int(a['confidence']*100)}%</span>
        </div>
        """, unsafe_allow_html=True)

with tabs[0]:
    render(articles)
with tabs[1]:
    render(bull)
with tabs[2]:
    render(neut)
with tabs[3]:
    render(bear)
