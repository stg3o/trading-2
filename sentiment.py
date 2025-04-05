import requests

def get_coin_sentiment(symbol="bitcoin"):
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{symbol.lower()}"
        res = requests.get(url)
        res.raise_for_status()
        data = res.json()

        up = data.get("sentiment_votes_up_percentage", 0)
        down = data.get("sentiment_votes_down_percentage", 0)

        label = "Neutral"
        if up - down > 20:
            label = "Positive"
        elif down - up > 20:
            label = "Negative"

        summary = f"{up:.1f}% of users voted positive, {down:.1f}% negative."

        return {
            "score": up - down,
            "label": label,
            "summary": summary
        }

    except Exception as e:
        return {
            "score": None,
            "label": "Error",
            "summary": f"Failed to fetch sentiment: {str(e)}"
        }
