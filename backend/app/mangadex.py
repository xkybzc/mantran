"""Minimal MangaDex API client: look up titles, list chapters, download pages.

Follows the API's usage rules: a descriptive User-Agent, staying under the
global rate limit (about 5 requests per second), and reporting image fetches
from MangaDex@Home servers back to the network.
"""

from __future__ import annotations

import re
import threading
import time
from urllib.parse import urlparse

import requests

API_URL = "https://api.mangadex.org"
REPORT_URL = "https://api.mangadex.network/report"
USER_AGENT = "Mantran/0.3 (personal manga translator)"
MIN_REQUEST_INTERVAL = 0.25
CONTENT_RATINGS = ("safe", "suggestive", "erotica", "pornographic")

_UUID = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
_LINK = re.compile(rf"mangadex\.org/(title|manga|chapter)/({_UUID})")

_session = requests.Session()
_session.headers["User-Agent"] = USER_AGENT
_throttle_lock = threading.Lock()
_last_request = 0.0


class MangaDexError(Exception):
    pass


def parse_link(link: str) -> tuple[str, str]:
    """Return ``(kind, id)`` for a MangaDex link, where kind is "manga" or "chapter".

    A bare UUID returns kind "unknown".
    """
    link = link.strip()
    match = _LINK.search(link)
    if match:
        kind = "chapter" if match.group(1) == "chapter" else "manga"
        return kind, match.group(2).lower()
    if re.fullmatch(_UUID, link):
        return "unknown", link.lower()
    raise MangaDexError("That doesn't look like a MangaDex title or chapter link.")


def _wait_for_slot():
    global _last_request
    with _throttle_lock:
        delay = _last_request + MIN_REQUEST_INTERVAL - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        _last_request = time.monotonic()


def _get(path: str, params=None, retries=2) -> dict:
    for attempt in range(retries + 1):
        _wait_for_slot()
        try:
            response = _session.get(f"{API_URL}{path}", params=params, timeout=20)
        except requests.RequestException as exc:
            if attempt == retries:
                raise MangaDexError(f"Could not reach MangaDex: {exc}") from exc
            time.sleep(1 + attempt)
            continue

        if response.status_code == 429 and attempt < retries:
            retry_at = response.headers.get("X-RateLimit-Retry-After")
            time.sleep(max(1.0, float(retry_at) - time.time()) if retry_at else 2.0)
            continue
        if response.status_code == 404:
            raise MangaDexError("MangaDex has no title or chapter with that link.")
        if response.status_code >= 400:
            raise MangaDexError(f"MangaDex returned an error ({response.status_code}).")
        return response.json()
    raise MangaDexError("MangaDex is rate limiting requests; try again in a minute.")


def _pick_title(attributes: dict) -> str:
    titles = attributes.get("title") or {}
    for key in ("en", "ja-ro", "ko-ro", "zh-ro"):
        if titles.get(key):
            return titles[key]
    for alt in attributes.get("altTitles") or []:
        if alt.get("en"):
            return alt["en"]
    return next(iter(titles.values()), "Untitled")


def _relationship(item: dict, kind: str) -> dict | None:
    return next((rel for rel in item.get("relationships", []) if rel.get("type") == kind), None)


def get_manga(manga_id: str) -> dict:
    data = _get(f"/manga/{manga_id}")["data"]
    attributes = data["attributes"]
    languages = [code for code in attributes.get("availableTranslatedLanguages") or [] if code]
    return {
        "id": data["id"],
        "title": _pick_title(attributes),
        "original_language": attributes.get("originalLanguage"),
        "languages": sorted(set(languages)),
    }


def _chapter_summary(item: dict) -> dict:
    attributes = item["attributes"]
    group = _relationship(item, "scanlation_group")
    return {
        "id": item["id"],
        "volume": attributes.get("volume"),
        "chapter": attributes.get("chapter"),
        "title": attributes.get("title") or "",
        "language": attributes.get("translatedLanguage"),
        "pages": attributes.get("pages") or 0,
        "group": ((group or {}).get("attributes") or {}).get("name") or "",
        "external_url": attributes.get("externalUrl"),
    }


def _chapter_sort_key(chapter: dict):
    try:
        number = float(chapter["chapter"])
    except (TypeError, ValueError):
        number = float("inf")
    return number, chapter["group"].lower()


def list_chapters(manga_id: str, language: str) -> list[dict]:
    """All readable chapters of a title in one language, in reading order."""
    chapters, offset, limit = [], 0, 500
    while True:
        payload = _get(f"/manga/{manga_id}/feed", params={
            "translatedLanguage[]": [language],
            "contentRating[]": list(CONTENT_RATINGS),
            "includes[]": ["scanlation_group"],
            "includeExternalUrl": 0,
            "includeEmptyPages": 0,
            "includeFuturePublishAt": 0,
            "order[chapter]": "asc",
            "limit": limit,
            "offset": offset,
        })
        chapters.extend(_chapter_summary(item) for item in payload["data"])
        offset += limit
        if offset >= payload.get("total", 0) or offset >= 10_000:
            break
    return sorted((c for c in chapters if c["pages"] > 0), key=_chapter_sort_key)


def get_chapter(chapter_id: str) -> dict:
    payload = _get(f"/chapter/{chapter_id}", params={"includes[]": ["scanlation_group", "manga"]})
    data = payload["data"]
    chapter = _chapter_summary(data)
    manga = _relationship(data, "manga") or {}
    chapter["manga_id"] = manga.get("id")
    chapter["manga_title"] = _pick_title(manga.get("attributes") or {}) if manga.get("attributes") else ""
    return chapter


def chapter_label(chapter: dict) -> str:
    parts = []
    if chapter.get("volume"):
        parts.append(f"Vol. {chapter['volume']}")
    parts.append(f"Ch. {chapter['chapter']}" if chapter.get("chapter") else "Oneshot")
    label = " ".join(parts)
    return f"{label} - {chapter['title']}" if chapter.get("title") else label


def get_page_urls(chapter_id: str, data_saver=False) -> list[str]:
    """Image URLs for a chapter's pages. They stay valid for roughly 15 minutes."""
    payload = _get(f"/at-home/server/{chapter_id}")
    base, info = payload["baseUrl"], payload["chapter"]
    if data_saver:
        return [f"{base}/data-saver/{info['hash']}/{name}" for name in info["dataSaver"]]
    return [f"{base}/data/{info['hash']}/{name}" for name in info["data"]]


def _report(url: str, success: bool, size: int, started: float, cached: bool):
    # MangaDex@Home nodes are volunteer servers; the network asks clients to
    # report each fetch. Images served from mangadex.org itself are exempt.
    if urlparse(url).hostname.endswith("mangadex.org"):
        return
    try:
        _session.post(REPORT_URL, json={
            "url": url,
            "success": success,
            "bytes": size,
            "duration": int((time.monotonic() - started) * 1000),
            "cached": cached,
        }, timeout=10)
    except requests.RequestException:
        pass


def download_page(url: str) -> bytes:
    started = time.monotonic()
    try:
        response = _session.get(url, timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        _report(url, False, 0, started, False)
        raise MangaDexError(f"Could not download page: {exc}") from exc
    cached = response.headers.get("X-Cache", "").upper().startswith("HIT")
    _report(url, True, len(response.content), started, cached)
    return response.content
