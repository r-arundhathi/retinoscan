"""
EyeScan non-retinal dataset collector
-------------------------------------
Creates/extends image_validation_dataset/non_retinal without changing
the existing retinal dataset or the existing COCO natural_images folder.

Source: Wikimedia Commons public API. The script records source page URLs
and available license metadata in source_manifest.csv.

Run from C:\\retina_project:
    python collect_non_retinal.py

Review every downloaded image before training. Search results can contain
irrelevant images; delete any image that is not genuinely in its folder's
category. The script never downloads PDFs or executes downloaded content.
"""

import csv
import hashlib
import html
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    raise SystemExit("Pillow is required. Install it with: python -m pip install Pillow")

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "image_validation_dataset" / "non_retinal"
MANIFEST = BASE / "source_manifest.csv"

TARGETS = {"documents": 300, "screenshots": 300, "charts": 200, "objects": 200}
SEARCH_TERMS = {
    "documents": ["scanned document page", "scientific journal page scan",
                  "printed report page", "book page scan", "invoice document scan",
                  "newspaper page scan", "form document scan"],
    "screenshots": ["computer software screenshot", "web browser screenshot",
                    "desktop application screenshot", "mobile application screenshot",
                    "computer interface screenshot"],
    "charts": ["bar chart", "line chart graph", "pie chart infographic",
               "flowchart diagram", "technical block diagram", "architecture diagram"],
    "objects": ["everyday objects photograph", "household objects photograph",
                "vehicle photograph", "book laptop phone photograph",
                "tools equipment photograph"],
}
API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "EyeScanDatasetCollector/1.0 (educational dataset preparation)"
ALLOWED = {".jpg", ".jpeg", ".png", ".webp"}
MAX_BYTES = 8 * 1024 * 1024
TIMEOUT = 30


def api_json(params):
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as response:
        return json.loads(response.read().decode("utf-8"))


def image_files(folder):
    return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in ALLOWED]


def existing_hashes(folder):
    hashes = set()
    for p in image_files(folder):
        try:
            hashes.add(hashlib.sha256(p.read_bytes()).hexdigest())
        except OSError:
            pass
    return hashes


def load_manifest_rows():
    if not MANIFEST.exists():
        return []
    with MANIFEST.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def save_manifest(rows):
    fields = ["category", "filename", "source_page", "license", "artist"]
    with MANIFEST.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def safe_filename(text):
    text = re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_")
    return (text[:70] or "commons_image")


def download_image(url, destination):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as response:
        data = response.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        return None
    temp = destination.with_suffix(destination.suffix + ".part")
    temp.write_bytes(data)
    try:
        with Image.open(temp) as im:
            im.verify()
        with Image.open(temp) as im:
            if im.width < 100 or im.height < 100:
                return None
            ext = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}.get((im.format or "").upper())
            if not ext:
                return None
        final = destination.with_suffix(ext)
        temp.replace(final)
        return final
    except Exception:
        return None
    finally:
        if temp.exists():
            temp.unlink(missing_ok=True)


def collect_category(category, target, rows):
    folder = BASE / category
    folder.mkdir(parents=True, exist_ok=True)
    hashes = existing_hashes(folder)
    count = len(image_files(folder))
    print(f"\n[{category}] existing: {count}; target: {target}")
    if count >= target:
        print("Target already reached; leaving this folder unchanged.")
        return rows

    seen_titles = set()
    for term in SEARCH_TERMS[category]:
        if count >= target:
            break
        print(f"Searching Commons: {term}")
        params = {
            "action": "query", "format": "json", "generator": "search",
            "gsrsearch": term, "gsrnamespace": 6, "gsrlimit": 50,
            "prop": "imageinfo", "iiprop": "url|extmetadata", "iiurlwidth": 700,
        }
        try:
            payload = api_json(params)
        except Exception as exc:
            print(f"Search request failed: {exc}")
            continue

        pages = (payload.get("query") or {}).get("pages") or {}
        for page in pages.values():
            if count >= target:
                break
            title = page.get("title", "")
            if title in seen_titles:
                continue
            seen_titles.add(title)
            info_list = page.get("imageinfo") or []
            if not info_list:
                continue
            info = info_list[0]
            image_url = info.get("thumburl") or info.get("url")
            source_page = info.get("descriptionurl", "")
            if not image_url:
                continue
            suffix = Path(urllib.parse.urlparse(image_url).path).suffix.lower()
            if suffix and suffix not in ALLOWED:
                continue

            base_name = safe_filename(Path(title).stem)
            destination = folder / f"{base_name}.jpg"
            n = 1
            while destination.exists():
                destination = folder / f"{base_name}_{n}.jpg"
                n += 1
            try:
                saved = download_image(image_url, destination)
            except Exception:
                saved = None
            if not saved:
                continue

            digest = hashlib.sha256(saved.read_bytes()).hexdigest()
            if digest in hashes:
                saved.unlink(missing_ok=True)
                continue
            hashes.add(digest)
            extmeta = info.get("extmetadata") or {}
            license_name = html.unescape(re.sub("<[^>]+>", "", (extmeta.get("LicenseShortName") or {}).get("value", "")))
            artist = html.unescape(re.sub("<[^>]+>", "", (extmeta.get("Artist") or {}).get("value", "")))[:300]
            rows.append({"category": category, "filename": saved.name,
                         "source_page": source_page, "license": license_name, "artist": artist})
            count += 1
            print(f"  saved {count}/{target}: {saved.name}")
            time.sleep(0.15)

    print(f"[{category}] finished with {count}/{target} images.")
    return rows


def main():
    BASE.mkdir(parents=True, exist_ok=True)
    for name in ["documents", "screenshots", "charts", "natural_images", "objects"]:
        (BASE / name).mkdir(parents=True, exist_ok=True)

    rows = load_manifest_rows()
    for category, target in TARGETS.items():
        rows = collect_category(category, target, rows)
        save_manifest(rows)

    print("\nCollection summary")
    print("------------------")
    for category in ["documents", "screenshots", "charts", "natural_images", "objects"]:
        print(f"{category:15s}: {len(image_files(BASE / category))}")
    print(f"\nSource/license manifest: {MANIFEST}")
    print("Review the images manually and remove any that do not match their folder.")


if __name__ == "__main__":
    main()
