import streamlit as st
import pytz
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
from news_fetcher import fetch_all
from sentiment import score_articles

st.set_page_config(
    page_title="Nifty Sentiment Monitor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

IST = pytz.timezone("Asia/Kolkata")

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #0d1117; color: #e6edf3; }
  [data-testid="stSidebar"] { background: #161b22; border-right: 1px solid #30363d; }
  .block-container { padding-top: 1rem; }

  .sent-box { border-radius: 12px; padding: 20px; text-align: center;
              margin-bottom: 12px; border: 2px solid; }
  .sent-bull { background:#0d2b1f; border-color:#3fb950; }
  .sent-bear { background:#2a1010; border-color:#f85149; }
  .sent-neut { background:#2a2210; border-color:#d29922; }
  .sent-label-bull { font-size:2rem; font-weight:800; color:#3fb950; }
  .sent-label-bear { font-size:2rem; font-weight:800; color:#f85149; }
  .sent-label-neut { font-size:2rem; font-weight:800; color:#d29922; }

  .news-card { background:#161b22; border:1px solid #30363d; border-radius:10px;
               padding:12px 16px; margin-bottom:8px; border-left:4px solid; }
  .card-bull { border-left-color:#3fb950; }
  .card-bear { border-left-color:#f85149; }
  .card-neut { border-left-color:#d29922; }

  .badge { display:inline-block; font-size:.72rem; font-weight:700;
           padding:2px 8px; border-radius:10px; margin-right:6px; }
  .badge-bull { background:#0d2b1f; color:#3fb950; border:1px solid #1e4731; }
  .badge-bear { background:#2a1010; color:#f85149; border:1px solid #5a1f1f; }
  .badge-neut { background:#2a2210; color:#d29922; border:1px solid #5a4820; }

  .feed-tag { font-size:.68rem; background:#21262d; color:#8b949e;
              padding:2px 7px; border-radius:4px; border:1px solid #30363d; }
  .meta-row  { font-size:.75rem; color:#8b949e; margin-top:4px; }
  .headline  { font-size:.9rem; font-weight:500; color:#e6edf3; }
  .change-box { background:#1c2128; border:1px solid #30363d; border-radius:8px;
                padding:10px 14px; font-size:.82rem; color:#8b949e; margin-bottom:12px; }
</style>
""", unsafe_allow_html=True)

# ── Market hours check ────────────────────────────────────────────────────────
def is_market_hours():
    now = datetime.now(IST)
    return now.weekday() < 5 and (
        now.replace(hour=9, minute=15, second=0) <= now <=
        now.replace(hour=15, minute=30, second=0)
    )

interval_ms = 10 * 60 * 1000 if is_market_hours() else 30 * 60 * 1000
interval_label = "10 min (Market Hours)" if is_market_hours() else "30 min (Off-Hours)"

# ── Auto-refresh ──────────────────────────────────────────────────────────────
st_autorefresh(interval=interval_ms, key="auto_refresh")  # [web:48][web:52]

# ── Session state for sentiment history ──────────────────────────────────────
if "prev_sentiment" not in st.session_state:
    st.session_state.prev_sentiment = None
    st.session_state.prev_score     = 0.0

# ── Data load with spinner ────────────────────────────────────────────────────
@st.cache_data(ttl=600, show_spinner=False)
def load_data():
    articles = fetch_all()
    return score_articles(articles)

with st.spinner("🔄 Fetching & scoring news with FinBERT..."):
    articles = load_data()

# ── Compute overall sentiment ─────────────────────────────────────────────────
scores = [a["score"] for a in articles]
avg    = round(sum(scores) / len(scores), 3) if scores else 0.0
if avg > 0.15:    overall = "Bullish"
elif avg < -0.15: overall = "Bearish"
else:             overall = "Neutral"

delta = round(avg - st.session_state.prev_score, 3)
if st.session_state.prev_sentiment is None:
    change_str = "Initial load"
elif overall != st.session_state.prev_sentiment:
    arrow = "▲" if delta > 0 else "▼"
    change_str = f"{st.session_state.prev_sentiment} → {overall} ({arrow}{abs(delta):.3f})"
elif abs(delta) > 0.005:
    arrow = "▲" if delta > 0 else "▼"
    change_str = f"Unchanged label ({arrow}{abs(delta):.3f})"
else:
    change_str = "No significant change"

st.session_state.prev_sentiment = overall
st.session_state.prev_score     = avg

# ── Breakdown counts ──────────────────────────────────────────────────────────
bull_arts = [a for a in articles if a["sentiment"] == "positive"]
bear_arts = [a for a in articles if a["sentiment"] == "negative"]
neut_arts = [a for a in articles if a["sentiment"] == "neutral"]
total     = len(articles) or 1

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 Nifty Sentiment")

    box_cls   = {"Bullish":"sent-bull","Bearish":"sent-bear","Neutral":"sent-neut"}[overall]
    label_cls = {"Bullish":"sent-label-bull","Bearish":"sent-label-bear","Neutral":"sent-label-neut"}[overall]
    st.markdown(f"""
    <div class="sent-box {box_cls}">
      <div class="{label_cls}">{overall}</div>
      <div style="color:#8b949e;font-size:.85rem">Score: {avg:+.3f}</div>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="change-box">
      🔄 <b>Change:</b> {change_str}
    </div>""", unsafe_allow_html=True)

    # Gauge bar
    pct = int((avg + 1) / 2 * 100)
    st.markdown("**Sentiment Gauge**")
    st.markdown(f"""
    <div style="height:10px;border-radius:5px;background:linear-gradient(to right,#f85149,#555,#3fb950);position:relative;margin-bottom:16px">
      <div style="width:12px;height:12px;background:#fff;border-radius:50%;position:absolute;top:-1px;left:{pct}%;transform:translateX(-50%)"></div>
    </div>
    <div style="display:flex;justify-content:space-between;font-size:.7rem;color:#8b949e;margin-top:-12px;margin-bottom:16px">
      <span>Bearish</span><span>Neutral</span><span>Bullish</span>
    </div>""", unsafe_allow_html=True)

    # Breakdown metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("🟢 Bull", len(bull_arts))
    c2.metric("🟡 Neut", len(neut_arts))
    c3.metric("🔴 Bear", len(bear_arts))

    st.progress(len(bull_arts) / total, text=f"Bullish {len(bull_arts)/total*100:.0f}%")
    st.progress(len(neut_arts) / total, text=f"Neutral {len(neut_arts)/total*100:.0f}%")
    st.progress(len(bear_arts) / total, text=f"Bearish {len(bear_arts)/total*100:.0f}%")

    st.divider()

    market_status = "🟢 Market Open" if is_market_hours() else "🔴 Market Closed"
    st.markdown(f"**{market_status}**")
    st.caption(f"Auto-refresh: **{interval_label}**")
    st.caption(f"Last loaded: **{datetime.now(IST).strftime('%d %b %Y, %H:%M:%S IST')}**")
    st.caption(f"Articles analysed: **{len(articles)}**")

    st.divider()
    if st.button("🔄 Manual Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ── MAIN PANEL ────────────────────────────────────────────────────────────────
st.markdown("### 📰 Latest Headlines")

tab_all, tab_bull, tab_neut, tab_bear, tab_rss, tab_fin = st.tabs([
    f"All ({len(articles)})",
    f"🟢 Bullish ({len(bull_arts)})",
    f"🟡 Neutral ({len(neut_arts)})",
    f"🔴 Bearish ({len(bear_arts)})",
    "📡 RBI/PIB/RSS",
    "📊 Finnhub"
])

def render_articles(arts):
    if not arts:
        st.info("No articles in this category.")
        return
    for a in arts:
        s   = a["sentiment"]
        cls = {"positive":"bull","negative":"bear","neutral":"neut"}[s]
        icon = {"positive":"▲","negative":"▼","neutral":"●"}[s]
        conf = int(a["confidence"] * 100)
        pub  = a.get("published","")[:16].replace("T"," ")
        st.markdown(f"""
        <div class="news-card card-{cls}">
          <a href="{a['url']}" target="_blank" class="headline">{a['title']}</a>
          <div class="meta-row">
            <span class="badge badge-{cls}">{icon} {s}</span>
            <span class="feed-tag">{a['feed']}</span>
            &nbsp;{a['source']}&nbsp;·&nbsp;🕐 {pub}&nbsp;·&nbsp;conf: {conf}%
          </div>
        </div>""", unsafe_allow_html=True)

with tab_all:  render_articles(articles)
with tab_bull: render_articles(bull_arts)
with tab_neut: render_articles(neut_arts)
with tab_bear: render_articles(bear_arts)
with tab_rss:  render_articles([a for a in articles if a["feed"] == "RSS"])
with tab_fin:  render_articles([a for a in articles if a["feed"] == "Finnhub"])