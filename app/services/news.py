# app/services/news.py
import feedparser
import logging
import requests
from datetime import datetime
from dateutil.parser import parse as date_parse
from dateutil.relativedelta import relativedelta
from dateutil.tz import tzutc
import re

logger = logging.getLogger(__name__)

def clean_summary(text):
    # Remove HTML tags
    clean = re.sub(r'<[^>]*>', '', text)
    # Remove extra whitespace
    clean = re.sub(r'\s+', ' ', clean).strip()
    return clean

def get_crypto_news(coin=None, limit=10):
    feeds = [
        "https://www.coindesk.com/arc/outboundfeeds/rss/?outputType=xml",
        "https://cointelegraph.com/rss",
        "https://thedefiant.io/feed",
        "https://blockworks.co/feed"
    ]
    news_items = []
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36',
        'Accept': 'application/rss+xml, application/xml, text/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Referer': 'https://www.coindesk.com/'
    }
    for url in feeds:
        logger.info(f"Fetching RSS feed: {url}")
        try:
            r = requests.get(url, timeout=10, headers=headers)
            r.raise_for_status()
            # Force UTF-8 decoding to handle charset mismatches
            content = r.content.decode('utf-8', errors='replace').encode('utf-8')
            feed = feedparser.parse(content)
            if feed.bozo:
                logger.warning(f"Bozo flag set for feed {url}: {feed.bozo_exception}, but proceeding if entries available")
            if 'entries' in feed:
                logger.info(f"Retrieved {len(feed.entries)} entries from {url}")
                for entry in feed.entries:
                    title = entry.get('title', '')
                    link = entry.get('link', '')
                    published = entry.get('published', entry.get('updated', entry.get('pubDate', '')))
                    summary = entry.get('summary', entry.get('description', ''))
                    summary = clean_summary(summary)
                    if coin and coin.upper() not in (title + summary).upper():
                        continue
                    news_items.append({
                        'title': title,
                        'link': link,
                        'published': published,
                        'summary': summary
                    })
        except Exception as e:
            logger.error(f"Failed to fetch or parse {url}: {str(e)}")
            continue
    def parse_date(date_str):
        date_str = date_str.strip()
        if not date_str:
            logger.warning("Empty date string, using current time")
            return datetime.now()
        try:
            parsed = date_parse(date_str, fuzzy=True)
            now = datetime.now(tzutc())
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=tzutc())
            if parsed > now:
                parsed = parsed + relativedelta(years=-1)
            return parsed.replace(tzinfo=None)
        except Exception as e:
            logger.warning(f"Failed to parse date: {date_str}, error: {str(e)}, using current time")
            return datetime.now()
    news_items.sort(key=lambda x: parse_date(x['published']), reverse=True)
    return news_items[:limit]