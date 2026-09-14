"""
Pulse - Auto US News Poster for Facebook (text-only, reliable baseline)
-------------------------------------------------------------------------
Fetches fresh general US news from Google News RSS (free, no API key),
asks Groq's free LLM to write a detailed, strictly-accurate summary with
hashtags, and posts it as text to the Facebook Page. Tracks posted article
URLs so nothing is ever repeated.
"""

import os
import re
import json
import time
import hashlib
import requests
import feedparser

FB_PAGE_ID = os.environ["FB_PAGE_ID"]
FB_PAGE_ACCESS_TOKEN = os.environ["FB_PAGE_ACCESS_TOKEN"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]

DEDUPE_FILE = "posted_urls.json"
MAX_POSTS_PER_RUN = 1
GROQ_MODEL = "openai/gpt-oss-120b"

RSS_FEEDS = [
    "https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/headlines/section/geo/United%20States?hl=en-US&gl=US&ceid=US:en",
]


def load_posted():
    if os.path.exists(DEDUPE_FILE):
        with open(DEDUPE_FILE, "r") as f:
            return set(json.load(f))
    return set()


def save_posted(posted):
    with open(DEDUPE_FILE, "w") as f:
        json.dump(sorted(posted), f, indent=2)


def url_hash(url):
    return hashlib.sha256(url.encode()).hexdigest()


def fetch_candidates(posted):
    candidates = []
    seen_this_run = set()

    for feed_url in RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_url)
        except Exception as e:
            print(f"Failed to fetch {feed_url}: {e}")
            continue

        for entry in feed.entries:
            link = entry.get("link")
            if not link:
                continue
            h = url_hash(link)
            if h in posted or h in seen_this_run:
                continue
            seen_this_run.add(h)

            candidates.append({
                "hash": h,
                "url": link,
                "title": (entry.get("title") or "").strip(),
                "summary": entry.get("summary") or entry.get("description") or "",
                "published": entry.get("published_parsed"),
            })

    candidates.sort(key=lambda c: c["published"] or time.gmtime(0), reverse=True)
    return candidates


def generate_post(title, summary, url):
    prompt = f"""You are the social media editor for a US news Facebook page called "Pulse".

Write a detailed but strictly accurate Facebook post explaining what happened in this story.
Use only the facts given below - do not invent names, numbers, or details not present in the source.
Aim for 4-6 informative sentences covering the who/what/when/where if available.
End with 2-3 relevant hashtags, then a new line: "Read more: {url}"

Headline: {title}
Source summary: {summary}

Return ONLY the post text - no preamble, no quotation marks."""

    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": GROQ_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.5,
            "max_tokens": 400,
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def post_text_to_facebook(message):
    resp = requests.post(
        f"https://graph.facebook.com/{FB_PAGE_ID}/feed",
        data={
            "message": message,
            "access_token": FB_PAGE_ACCESS_TOKEN,
        },
        timeout=30,
    )
    if resp.status_code != 200:
        print("Facebook text post error:", resp.text)
        resp.raise_for_status()
    return resp.json()


def main():
    posted = load_posted()
    candidates = fetch_candidates(posted)

    if not candidates:
        print("No new articles found this run.")
        return

    posted_count = 0
    for c in candidates:
        if posted_count >= MAX_POSTS_PER_RUN:
            break
        try:
            caption = generate_post(c["title"], c["summary"], c["url"])
            result = post_text_to_facebook(caption)
            print(f"Posted: {c['title']} -> {result}")
            posted.add(c["hash"])
            posted_count += 1
        except Exception as e:
            print(f"Failed to post '{c['title']}': {e}")
            continue

    save_posted(posted)


if __name__ == "__main__":
    main()
