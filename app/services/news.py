# app/services/news.py
import feedparser
from datetime import datetime
import logging

# Configure logging
logger = logging.getLogger(__name__)

def parse_date(date_str):
    try:
        return datetime.strptime(date_str, '%a, %d %b %Y %H:%M:%S %z')
    except ValueError:
        logger.warning(f"Failed to parse date: {date_str}, using current time")
        return datetime.now()

def get_crypto_news(coin=None, limit=10):
    feeds = [
        'https://www.coindesk.com/arc/outboundfeeds/rss/',
        'https://cointelegraph.com/rss',
        'https://bitcoinmagazine.com/feed',
        'https://thedefiant.io/feed',  # Added for diversity
        'https://blockworks.co/feed',  # Added for diversity
    ]
    news_items = []
    for feed_url in feeds:
        try:
            logger.info(f"Fetching RSS feed: {feed_url}")
            feed = feedparser.parse(feed_url)
            if feed.bozo:
                logger.error(f"Failed to parse feed {feed_url}: {feed.bozo_exception}")
                continue
            for entry in feed.entries:
                title = entry.get('title', 'No title')
                summary = entry.get('summary', 'No summary')
                if coin:
                    coin_lower = coin.lower()
                    if coin_lower not in title.lower() and coin_lower not in summary.lower():
                        continue
                news_items.append({
                    'title': title,
                    'link': entry.get('link', ''),
                    'published': entry.get('published', datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
                    'summary': summary,
                })
            logger.info(f"Retrieved {len(feed.entries)} entries from {feed_url}")
        except Exception as e:
            logger.error(f"Error fetching feed {feed_url}: {e}")
    # Sort by published date descending
    news_items.sort(key=lambda x: parse_date(x['published']), reverse=True)
    # If no coin-specific results, return general crypto news as fallback
    if not news_items and coin:
        logger.info(f"No news found for {coin}, fetching general news")
        return get_crypto_news(limit=limit)  # Recursive call without coin
    logger.info(f"Returning {len(news_items)} news items for coin: {coin or 'all'}")
    return news_items[:limit]