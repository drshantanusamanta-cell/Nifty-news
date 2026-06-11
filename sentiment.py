from transformers import pipeline

_classifier = None

def get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            truncation=True,
            max_length=512
        )
    return _classifier

SCORE_MAP = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}

def score_articles(articles):
    clf = get_classifier()
    for a in articles:
        try:
            result = clf(a["title"])[0]
            label  = result["label"].lower()
            a["sentiment"]  = label
            a["confidence"] = round(result["score"], 3)
            a["score"]      = SCORE_MAP.get(label, 0.0)
        except:
            a["sentiment"]  = "neutral"
            a["confidence"] = 0.0
            a["score"]      = 0.0
    return articles