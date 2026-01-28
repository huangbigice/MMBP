import requests
import json
from datetime import datetime

def get_news():
    stock_id = 2330
    url = f'https://ess.api.cnyes.com/ess/api/v1/news/keyword?q={stock_id}&limit=5&page=1'
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/117.0 Safari/537.36"
    }

    r = requests.get(url, headers=headers, timeout=6)

    # 先做基礎回應檢查
    if r.status_code != 200 or not r.text.strip().startswith("{"):
        return []

    json_data = json.loads(r.text)

    news_list = []
    for item in json_data.get('data', {}).get('items', []):
        news_id = item.get("newsId")
        title = item.get("title")
        publish_at = item.get("publishAt")
        if not publish_at:
            continue
        utc_time = datetime.utcfromtimestamp(publish_at)
        news_time = utc_time.strftime('%Y-%m-%d %H:%M:%S')
        news_list.append({
            "title": title,
            "url": f"https://news.cnyes.com/news/id/{news_id}",
            "date": news_time
        })

    return news_list
