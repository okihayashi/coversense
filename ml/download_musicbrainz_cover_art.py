#!/usr/bin/env python3
"""Build a MusicBrainz + Cover Art Archive album-cover dataset.

The exported layout matches the existing CoverSense metadata contract:

data/musicbrainz_cover_art/
  images/<label>/<label>-00000.jpg
  metadata.csv
  labels.json

Use this as a separate dataset source, then compare it against the current
Hugging Face dataset without mixing their files.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm


DEFAULT_OUTPUT_DIR = Path("data/musicbrainz_cover_art")
DEFAULT_USER_AGENT = "CoverSense/0.1 (https://github.com/okihayashi/coversense)"
MUSICBRAINZ_BASE_URL = "https://musicbrainz.org/ws/2"
COVER_ART_BASE_URL = "https://coverartarchive.org"
METADATA_FIELDNAMES = [
    "image_path",
    "label",
    "label_name",
    "source",
    "source_id",
    "source_title",
    "artist",
    "first_release_date",
    "musicbrainz_score",
    "query_term",
    "cover_art_url",
]

GENRE_QUERIES = {
    "blues": ["blues"],
    "classical": ["classical"],
    "country": ["country"],
    "deathmetal": ["death metal"],
    "doommetal": ["doom metal"],
    "drumnbass": ["drum and bass", "drum'n'bass", "drum & bass"],
    "electronic": ["electronic"],
    "folk": ["folk"],
    "grime": ["grime"],
    "heavymetal": ["heavy metal"],
    "hiphop": ["hip hop", "hip-hop"],
    "jazz": ["jazz"],
    "lofi": ["lo-fi", "lofi"],
    "pop": ["pop"],
    "psychedelicrock": ["psychedelic rock"],
    "punk": ["punk"],
    "reggae": ["reggae"],
    "rock": ["rock"],
    "soul": ["soul"],
    "techno": ["techno"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--labels",
        default=",".join(GENRE_QUERIES),
        help="Comma-separated CoverSense labels to collect.",
    )
    parser.add_argument("--max-per-label", type=int, default=100)
    parser.add_argument("--search-limit", type=int, default=100)
    parser.add_argument("--max-pages-per-query", type=int, default=5)
    parser.add_argument("--thumbnail-size", type=int, choices=[250, 500, 1200], default=500)
    parser.add_argument("--image-size", type=int, default=300)
    parser.add_argument("--delay", type=float, default=1.1, help="Delay between MusicBrainz requests.")
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    parser.add_argument("--resume", action="store_true", help="Reuse existing images and fill missing labels.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def request_json(url: str, params: dict[str, str | int], user_agent: str) -> dict[str, Any]:
    full_url = f"{url}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(
        full_url,
        headers={
            "Accept": "application/json",
            "User-Agent": user_agent,
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def download_image(url: str, destination: Path, image_size: int, user_agent: str) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(request, timeout=30) as response:
        with Image.open(response) as image:
            image = image.convert("RGB")
            image.thumbnail((image_size, image_size))
            image.save(destination, quality=92)


def artist_credit_name(release_group: dict[str, Any]) -> str:
    credits = release_group.get("artist-credit") or []
    names = []
    for credit in credits:
        if isinstance(credit, dict):
            if credit.get("name"):
                names.append(str(credit["name"]))
            elif isinstance(credit.get("artist"), dict) and credit["artist"].get("name"):
                names.append(str(credit["artist"]["name"]))
    return " / ".join(names)


def search_release_groups(
    label: str,
    query_term: str,
    args: argparse.Namespace,
    seen_release_groups: set[str],
    start_index: int,
) -> list[dict[str, str]]:
    records = []
    query = f'tag:"{query_term}" AND primarytype:album'
    label_dir = args.output_dir / "images" / label
    label_dir.mkdir(parents=True, exist_ok=True)

    for page in range(args.max_pages_per_query):
        if len(records) >= args.max_per_label:
            break

        offset = page * args.search_limit
        try:
            payload = request_json(
                f"{MUSICBRAINZ_BASE_URL}/release-group/",
                {
                    "query": query,
                    "fmt": "json",
                    "limit": args.search_limit,
                    "offset": offset,
                },
                args.user_agent,
            )
        except urllib.error.HTTPError as error:
            print(f"MusicBrainz search failed for {label}/{query_term}: HTTP {error.code}")
            break
        except (TimeoutError, urllib.error.URLError) as error:
            print(f"MusicBrainz search failed for {label}/{query_term}: {error}")
            break

        time.sleep(args.delay)
        release_groups = payload.get("release-groups", [])
        if not release_groups:
            break

        for release_group in tqdm(release_groups, desc=f"{label}:{query_term}", leave=False):
            if len(records) >= args.max_per_label:
                break

            release_group_id = release_group.get("id")
            if not release_group_id or release_group_id in seen_release_groups:
                continue

            primary_type = str(release_group.get("primary-type") or "").lower()
            if primary_type and primary_type != "album":
                continue

            count = start_index + len(records)
            filename = f"{label}-{count:05d}.jpg"
            image_path = label_dir / filename
            cover_url = f"{COVER_ART_BASE_URL}/release-group/{release_group_id}/front-{args.thumbnail_size}.jpg"

            if not args.dry_run:
                try:
                    download_image(cover_url, image_path, args.image_size, args.user_agent)
                except urllib.error.HTTPError as error:
                    if error.code not in {404, 503}:
                        print(f"Cover download failed for {release_group_id}: HTTP {error.code}")
                    continue
                except (OSError, TimeoutError, urllib.error.URLError) as error:
                    print(f"Cover download failed for {release_group_id}: {error}")
                    continue

            seen_release_groups.add(release_group_id)
            records.append(
                {
                    "image_path": image_path.as_posix(),
                    "label": label,
                    "label_name": label,
                    "source": "musicbrainz-cover-art-archive",
                    "source_id": release_group_id,
                    "source_title": str(release_group.get("title") or ""),
                    "artist": artist_credit_name(release_group),
                    "first_release_date": str(release_group.get("first-release-date") or ""),
                    "musicbrainz_score": str(release_group.get("score") or ""),
                    "query_term": query_term,
                    "cover_art_url": cover_url,
                }
            )

    return records


def selected_labels(raw_labels: str) -> list[str]:
    labels = [label.strip() for label in raw_labels.split(",") if label.strip()]
    unknown = sorted(set(labels) - set(GENRE_QUERIES))
    if unknown:
        raise ValueError(f"Unknown labels: {', '.join(unknown)}")
    return labels


def read_existing_metadata(output_dir: Path) -> dict[str, dict[str, str]]:
    metadata_path = output_dir / "metadata.csv"
    if not metadata_path.exists():
        return {}
    with metadata_path.open(newline="", encoding="utf-8") as file:
        return {row["image_path"]: row for row in csv.DictReader(file)}


def existing_image_records(output_dir: Path, labels: list[str]) -> tuple[list[dict[str, str]], defaultdict[str, int], set[str]]:
    prior_rows = read_existing_metadata(output_dir)
    records = []
    counts: defaultdict[str, int] = defaultdict(int)
    seen_release_groups = set()

    for label in labels:
        label_dir = output_dir / "images" / label
        for image_path in sorted(label_dir.glob("*.jpg")):
            row = prior_rows.get(image_path.as_posix())
            if row is None:
                row = {
                    "image_path": image_path.as_posix(),
                    "label": label,
                    "label_name": label,
                    "source": "musicbrainz-cover-art-archive",
                    "source_id": "",
                    "source_title": "",
                    "artist": "",
                    "first_release_date": "",
                    "musicbrainz_score": "",
                    "query_term": "",
                    "cover_art_url": "",
                }
            records.append(row)
            counts[label] += 1
            if row.get("source_id"):
                seen_release_groups.add(row["source_id"])

    return records, counts, seen_release_groups


def write_dataset_files(
    output_dir: Path,
    records: list[dict[str, str]],
    label_counts: dict[str, int] | defaultdict[str, int],
    thumbnail_size: int,
) -> None:
    with (output_dir / "metadata.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=METADATA_FIELDNAMES)
        writer.writeheader()
        writer.writerows(records)

    label_payload = {
        "dataset": "musicbrainz-cover-art-archive",
        "labels": sorted(label_counts.keys()),
        "label_counts": dict(sorted(label_counts.items())),
        "source": {
            "musicbrainz": MUSICBRAINZ_BASE_URL,
            "cover_art_archive": COVER_ART_BASE_URL,
            "thumbnail_size": thumbnail_size,
        },
    }
    (output_dir / "labels.json").write_text(json.dumps(label_payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    labels = selected_labels(args.labels)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "images").mkdir(parents=True, exist_ok=True)

    if args.resume:
        records, label_counts, seen_release_groups = existing_image_records(args.output_dir, labels)
    else:
        seen_release_groups: set[str] = set()
        records = []
        label_counts: defaultdict[str, int] = defaultdict(int)

    for label in labels:
        for query_term in GENRE_QUERIES[label]:
            if label_counts[label] >= args.max_per_label:
                break
            remaining = args.max_per_label - label_counts[label]
            query_args = argparse.Namespace(**{**vars(args), "max_per_label": remaining})
            new_records = search_release_groups(label, query_term, query_args, seen_release_groups, label_counts[label])
            records.extend(new_records)
            label_counts[label] += len(new_records)
            write_dataset_files(args.output_dir, records, label_counts, args.thumbnail_size)

    write_dataset_files(args.output_dir, records, label_counts, args.thumbnail_size)
    print(f"Exported {len(records)} covers to {args.output_dir}")


if __name__ == "__main__":
    main()
