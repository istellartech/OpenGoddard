"""Check bilingual routes, local links, anchors, and media after astro build.

Run from website with ``npm run check:site``. Uses only the Python standard
library so the same checks run locally and in the GitHub Pages build job.
"""

import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
BASE = "/OpenGoddard/"


class Page(HTMLParser):
    def __init__(self, path):
        super().__init__()
        self.lang = None
        self.ids = set()
        self.links = []
        self.language_options = set()
        self.missing_media = False
        self.japanese_ui = False
        self.feed(path.read_text())

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "html":
            self.lang = attrs.get("lang")
        if tag == "button" and re.search(
            r"[\u3040-\u30ff\u3400-\u9fff]",
            attrs.get("title", "") + attrs.get("aria-label", ""),
        ):
            self.japanese_ui = True
        if "id" in attrs:
            self.ids.add(attrs["id"])
        if "og-missing" in attrs.get("class", "").split():
            self.missing_media = True
        if tag == "option" and attrs.get("value", "").startswith(BASE):
            self.language_options.add(attrs["value"])
        if tag in ("a", "img", "source", "video", "script", "link"):
            for key in ("href", "src", "poster"):
                if attrs.get(key):
                    self.links.append(attrs[key])


def main():
    errors = []
    pages = {p.relative_to(DIST).as_posix(): Page(p) for p in DIST.rglob("*.html")}
    if not pages:
        sys.exit("No built pages; run npm run build first.")
    docs = ROOT / "src/content/docs"
    japanese = {
        p.relative_to(docs)
        for p in docs.rglob("*.mdx")
        if "en" not in p.relative_to(docs).parts
    }
    english = {p.relative_to(docs / "en") for p in (docs / "en").rglob("*.mdx")}
    for path in sorted(japanese ^ english):
        errors.append(f"Translation has no counterpart: {path}")

    # Include generated API modules: every public page must have a language twin.
    for name, page in pages.items():
        if name == "404.html":
            continue
        is_english = name.startswith("en/")
        twin = name[3:] if is_english else "en/" + name
        expected_lang = "en" if is_english else "ja"
        if page.lang != expected_lang:
            errors.append(f"{name}: expected lang={expected_lang}, got {page.lang}")
        if is_english and page.japanese_ui:
            errors.append(f"{name}: English page has untranslated button labels")
        if twin not in pages:
            errors.append(f"{name}: missing language counterpart {twin}")
        own_url = BASE + (
            name.removesuffix("index.html") if name.endswith("index.html") else name
        )
        twin_url = BASE + (
            twin.removesuffix("index.html") if twin.endswith("index.html") else twin
        )
        if page.language_options != {own_url, twin_url}:
            errors.append(f"{name}: language selector does not point to matching pages")
        if page.missing_media:
            errors.append(f"{name}: a figure or video has not been generated")

    for name, page in pages.items():
        url = "https://site.invalid" + BASE + name.removesuffix("index.html")
        for link in page.links:
            target = urlsplit(urljoin(url, link))
            if target.netloc != "site.invalid" or target.scheme not in (
                "http",
                "https",
            ):
                continue
            if not target.path.startswith(BASE):
                errors.append(f"{name}: local link omits the site base: {link}")
                continue
            relative = unquote(target.path[len(BASE) :])
            file = DIST / relative
            if file.is_dir():
                file /= "index.html"
            if not file.is_file():
                errors.append(f"{name}: missing target: {link}")
                continue
            key = file.relative_to(DIST).as_posix()
            if (
                target.fragment
                and key in pages
                and unquote(target.fragment) not in pages[key].ids
            ):
                errors.append(f"{name}: missing anchor: {link}")
            # Page links stay in the current language. The language picker uses
            # option elements, not anchors, and is checked separately above.
            if (
                name != "404.html"
                and key in pages
                and name.startswith("en/") != key.startswith("en/")
            ):
                errors.append(f"{name}: link changes language: {link}")
            if (
                name.startswith("en/")
                and "/media/" in target.path
                and not target.path.startswith(BASE + "media/en/")
            ):
                errors.append(f"{name}: English page uses Japanese media: {link}")

    search = DIST / "pagefind/pagefind-entry.json"
    languages = json.loads(search.read_text())["languages"] if search.exists() else {}
    if not {"ja", "en"} <= languages.keys():
        errors.append("Search index must include Japanese and English")
    if errors:
        sys.exit("\n".join(sorted(set(errors))))
    print(
        f"Checked {len(pages)} pages: language pairs, selectors, links, anchors, media, and search (ja/en)."
    )


if __name__ == "__main__":
    main()
