import requests, feedparser, pytz
from datetime import datetime
import streamlit as st

IST = pytz.timezone("Asia/Kolkata")

def _newsapi_key():    return st.secrets.get("NEWSAPI_KEY", "")
def _finnhub_key():    return st.secrets.get("FINNHUB_KEY", "")
def _freenews_key():   return st.secrets.get("FREENEWS_KEY", "")

def fetch_newsapi():
    try:
        r = requests.get("https://newsapi.org/v2/top-headlines", params={
            "country": "in", "category": "business",
            "pageSize": 15, "apiKey": _newsapi_key()
        }, timeout=10)
        return [{"title": a["title"], "source": a["source"]["name"],
                 "url": a["url"], "published": a.get("publishedAt",""),
                 "feed": "NewsAPI"}
                for a in r.json().get("articles", []) if a.get("title")]
    except Exception as e:
        st.warning(f"NewsAPI: {e}"); return []

def fetch_finnhub():
    try:
        r = requests.get("https://finnhub.io/api/v1/news",
            params={"category": "general", "token": _finnhub_key()}, timeout=10)
        items = r.json() if isinstance(r.json(), list) else []
        return [{"title": a["headline"], "source": a.get("source","Finnhub"),
                 "url": a.get("url",""),
                 "published": datetime.fromtimestamp(a["datetime"], tz=IST).strftime("%Y-%m-%dT%H:%M:%S"),
                 "feed": "Finnhub"}
                for a in items[:15] if a.get("headline")]
    except Exception as e:
        st.warning(f"Finnhub: {e}"); return []

def fetch_rss():
    feeds = [
        ("RBI",          "https://www.rbi.org.in/RSS/RBIRSSFeed.aspx?Id=316"),
        ("PIB",          "https://pib.gov.in/RSSNewsFeed.aspx?ModID=6"),
        ("ET Markets",   "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"),
        ("Moneycontrol", "https://www.moneycontrol.com/rss/marketreports.xml"),
        ("LiveMint",     "https://www.livemint.com/rss/markets"),
    ]
    results = []
    for name, url in feeds:
        try:
            for e in feedparser.parse(url).entries[:5]:
                results.append({"title": e.title, "source": name,
                                 "url": e.link,
                                 "published": e.get("published", e.get("updated","")),
                                 "feed": "RSS"})
        except: pass
    return results

def fetch_freenews():
    try:
        r = requests.get("https://freenewsapi.io/api/v1/news", params={
            "country": "IN", "language": "en",
            "category": "business", "pageSize": 10,
            "apiKey": _freenews_key()
        }, timeout=10)
        return [{"title": a["title"],
                 "source": a.get("source", {}).get("name","FreeNews"),
                 "url": a.get("url",""), "published": a.get("publishedAt",""),
                 "feed": "FreeNewsAPI"}
                for a in r.json().get("articles", []) if a.get("title")]
    except Exception as e:
        st.warning(f"FreeNewsAPI: {e}"); return []

def parse_date(s):
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S",
                "%a, %d %b %Y %H:%M:%S %z", "%a, %d %b %Y %H:%M:%S GMT"):
        try: return datetime.strptime(s[:25].strip(), fmt)
        except: pass
    return datetime.min

def fetch_all():
    raw = fetch_newsapi() + fetch_finnhub() + fetch_rss() + fetch_freenews()
    seen, unique = set(), []
    for a in raw:
        k = a["title"][:60].lower()
        if k not in seen:
            seen.add(k); unique.append(a)
    unique.sort(key=lambda x: parse_date(x.get("published","")), reverse=True)
    return unique[:60]