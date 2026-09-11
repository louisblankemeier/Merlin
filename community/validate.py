#!/usr/bin/env python3
"""Validate community entries and render the community index table.

Usage:
    python community/validate.py              # validate schema + check index is in sync
    python community/validate.py --write      # regenerate the index table in README.md
    python community/validate.py --check-links  # also verify every URL resolves
    python community/validate.py --links-only   # only verify URLs (used by the weekly CI job)

Only PyYAML is required beyond the standard library.
"""

from __future__ import annotations

import argparse
import datetime
import difflib
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

import yaml

COMMUNITY_DIR = Path(__file__).resolve().parent
ENTRIES_DIR = COMMUNITY_DIR / "entries"
INDEX_FILE = COMMUNITY_DIR / "README.md"

START_MARKER = "<!-- COMMUNITY_TABLE:START -->"
END_MARKER = "<!-- COMMUNITY_TABLE:END -->"

CATEGORIES = ("model", "dataset", "tool", "benchmark", "tutorial", "other")
LINK_KEYS = ("homepage", "code", "data", "paper", "demo", "docs")
LINK_LABELS = {
    "homepage": "Project",
    "code": "Code",
    "data": "Data",
    "paper": "Paper",
    "demo": "Demo",
    "docs": "Docs",
}
REQUIRED_KEYS = (
    "name",
    "description",
    "category",
    "authors",
    "license",
    "links",
    "added",
)
OPTIONAL_KEYS = ("citation", "contact", "tags", "merlin_components")
SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
URL_RE = re.compile(r"^https?://\S+$")
USER_AGENT = "Merlin-community-link-check/1.0 (+https://github.com/StanfordMIMI/Merlin)"

MAX_NAME = 80
MAX_DESCRIPTION = 300


class EntryError(Exception):
    """A single validation failure, reported with the offending file."""


def load_entries() -> list[tuple[Path, dict]]:
    """Parse every YAML file under entries/, sorted by display name."""
    if not ENTRIES_DIR.is_dir():
        raise EntryError(
            f"{ENTRIES_DIR} is missing; it holds one YAML file per project"
        )
    paths = sorted(p for p in ENTRIES_DIR.iterdir() if p.suffix in (".yaml", ".yml"))
    entries = []
    errors = []
    for path in paths:
        if path.name.startswith("."):
            continue
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            errors.append(
                f"{path.name}: not valid YAML ({exc.__class__.__name__}): {exc}"
            )
            continue
        if not isinstance(data, dict):
            errors.append(f"{path.name}: top level must be a mapping of fields")
            continue
        entries.append((path, data))
    if errors:
        raise EntryError("\n".join(errors))
    entries.sort(key=lambda item: str(item[1].get("name", item[0].stem)).lower())
    return entries


def validate_entry(path: Path, entry: dict) -> list[str]:
    """Return a list of human-readable problems with one entry."""
    problems = []

    def bad(msg: str) -> None:
        problems.append(f"{path.name}: {msg}")

    if path.suffix != ".yaml":
        bad("file must use the .yaml extension")
    if not SLUG_RE.match(path.stem):
        bad(
            f"filename must be a lowercase slug like 'my-project.yaml' (got '{path.stem}')"
        )

    missing = [key for key in REQUIRED_KEYS if entry.get(key) in (None, "", [], {})]
    if missing:
        bad(f"missing required field(s): {', '.join(missing)}")
    unknown = sorted(set(entry) - set(REQUIRED_KEYS) - set(OPTIONAL_KEYS))
    if unknown:
        bad(f"unknown field(s): {', '.join(unknown)}")

    name = entry.get("name")
    if name is not None:
        if not isinstance(name, str) or not name.strip():
            bad("'name' must be a non-empty string")
        elif len(name) > MAX_NAME:
            bad(f"'name' must be at most {MAX_NAME} characters (got {len(name)})")

    description = entry.get("description")
    if description is not None:
        if not isinstance(description, str) or not description.strip():
            bad("'description' must be a non-empty string")
        elif len(description) > MAX_DESCRIPTION:
            bad(
                f"'description' must be at most {MAX_DESCRIPTION} characters "
                f"(got {len(description)}); keep it to a sentence or two"
            )
        elif "\n" in description.strip():
            bad(
                "'description' must be a single line; use a YAML '>' folded block if long"
            )

    category = entry.get("category")
    if category is not None and category not in CATEGORIES:
        bad(f"'category' must be one of {', '.join(CATEGORIES)} (got '{category}')")

    authors = entry.get("authors")
    if authors is not None:
        if not isinstance(authors, list):
            bad("'authors' must be a list")
        else:
            for i, author in enumerate(authors):
                if not isinstance(author, dict):
                    bad(f"authors[{i}] must be a mapping with at least a 'name'")
                    continue
                if not str(author.get("name", "")).strip():
                    bad(f"authors[{i}] is missing 'name'")
                extra = sorted(set(author) - {"name", "affiliation", "email"})
                if extra:
                    bad(f"authors[{i}] has unknown field(s): {', '.join(extra)}")

    license_ = entry.get("license")
    if license_ is not None and (not isinstance(license_, str) or not license_.strip()):
        bad("'license' must be a non-empty string, e.g. 'MIT' or 'CC BY-NC 4.0'")

    links = entry.get("links")
    if links is not None:
        if not isinstance(links, dict):
            bad("'links' must be a mapping")
        else:
            unknown_links = sorted(set(links) - set(LINK_KEYS))
            if unknown_links:
                bad(
                    f"'links' has unknown key(s): {', '.join(unknown_links)}; "
                    f"allowed keys are {', '.join(LINK_KEYS)}"
                )
            if not str(links.get("homepage", "")).strip():
                bad("'links.homepage' is required so the index has somewhere to point")
            for key, url in links.items():
                if not isinstance(url, str) or not URL_RE.match(url.strip()):
                    bad(f"links.{key} must be an http(s) URL (got '{url}')")

    added = entry.get("added")
    if added is not None:
        if isinstance(added, datetime.date):
            pass  # PyYAML already parsed an ISO date for us
        elif isinstance(added, str):
            try:
                datetime.date.fromisoformat(added.strip())
            except ValueError:
                bad(f"'added' must be an ISO date like 2026-09-11 (got '{added}')")
        else:
            bad(f"'added' must be an ISO date like 2026-09-11 (got '{added}')")

    tags = entry.get("tags")
    if tags is not None and (
        not isinstance(tags, list)
        or not all(isinstance(t, str) and t.strip() for t in tags)
    ):
        bad("'tags' must be a list of strings")

    components = entry.get("merlin_components")
    if components is not None and (
        not isinstance(components, list)
        or not all(isinstance(c, str) and c.strip() for c in components)
    ):
        bad("'merlin_components' must be a list of strings")

    citation = entry.get("citation")
    if citation is not None and not isinstance(citation, str):
        bad("'citation' must be a string; use a YAML '|' block for BibTeX")

    contact = entry.get("contact")
    if contact is not None and not isinstance(contact, str):
        bad("'contact' must be a string (email address or URL)")

    return problems


def escape_cell(text: str) -> str:
    """Make a string safe to drop inside a Markdown table cell."""
    return " ".join(str(text).split()).replace("|", "\\|")


def render_links(links: dict) -> str:
    """One label per distinct URL.

    Projects commonly point 'homepage' and 'code' at the same repository; show
    that URL once, labelled with the more specific of the two.
    """
    chosen: dict[str, str] = {}
    for key in LINK_KEYS:
        url = str(links.get(key, "")).strip()
        if not url:
            continue
        if url not in chosen:
            chosen[url] = key
        elif chosen[url] == "homepage":
            chosen[url] = key
    ordered = sorted(chosen.items(), key=lambda item: LINK_KEYS.index(item[1]))
    return " · ".join(f"[{LINK_LABELS[key]}]({url})" for url, key in ordered)


def render_table(entries: list[tuple[Path, dict]]) -> str:
    """Render the summary table plus one collapsible block per entry."""
    if not entries:
        return (
            "_No community entries yet. Yours could be the first — see "
            "[CONTRIBUTING.md](CONTRIBUTING.md)._"
        )

    lines = [
        "| Project | Type | Description | License |",
        "| --- | --- | --- | --- |",
    ]
    for _, entry in entries:
        name = escape_cell(entry["name"])
        homepage = str(entry["links"]["homepage"]).strip()
        lines.append(
            f"| [{name}]({homepage}) "
            f"| {escape_cell(entry['category'])} "
            f"| {escape_cell(entry['description'])} "
            f"| {escape_cell(entry['license'])} |"
        )

    blocks = []
    for path, entry in entries:
        block = [
            "<details>",
            f"<summary><b>{entry['name']}</b> — {entry['category']}</summary>",
            "",
            f"- **Description:** {escape_cell(entry['description'])}",
        ]

        authors = []
        for author in entry["authors"]:
            who = str(author["name"]).strip()
            affiliation = str(author.get("affiliation", "")).strip()
            authors.append(f"{who} ({affiliation})" if affiliation else who)
        block.append(f"- **Authors:** {', '.join(authors)}")

        block.append(f"- **Links:** {render_links(entry['links'])}")
        block.append(f"- **License:** {escape_cell(entry['license'])}")

        if entry.get("merlin_components"):
            components = ", ".join(escape_cell(c) for c in entry["merlin_components"])
            block.append(f"- **Builds on:** {components}")
        if entry.get("tags"):
            tags = ", ".join(f"`{escape_cell(t)}`" for t in entry["tags"])
            block.append(f"- **Tags:** {tags}")
        if entry.get("contact"):
            block.append(f"- **Contact:** {escape_cell(entry['contact'])}")

        added = entry["added"]
        added = (
            added.isoformat()
            if isinstance(added, datetime.date)
            else str(added).strip()
        )
        block.append(
            f"- **Added:** {added} · **Entry:** [`{path.name}`](entries/{path.name})"
        )

        if entry.get("citation"):
            block += ["", "```bibtex", entry["citation"].strip(), "```"]

        block += ["", "</details>"]
        blocks.append("\n".join(block))

    return "\n".join(lines) + "\n\n" + "\n\n".join(blocks)


def splice_index(table: str) -> str:
    """Return the index file contents with the generated region replaced."""
    if not INDEX_FILE.exists():
        raise EntryError(f"{INDEX_FILE} is missing; it holds the generated table")
    text = INDEX_FILE.read_text(encoding="utf-8")
    start = text.find(START_MARKER)
    end = text.find(END_MARKER)
    if start == -1 or end == -1 or end < start:
        raise EntryError(
            f"{INDEX_FILE.name} must contain the {START_MARKER} and {END_MARKER} markers"
        )
    head = text[: start + len(START_MARKER)]
    tail = text[end:]
    return f"{head}\n\n{table}\n\n{tail}"


def check_links(entries: list[tuple[Path, dict]]) -> list[str]:
    """Request every URL once and report the ones that do not resolve."""
    seen: dict[str, list[str]] = {}
    for path, entry in entries:
        for url in (entry.get("links") or {}).values():
            if isinstance(url, str) and URL_RE.match(url.strip()):
                seen.setdefault(url.strip(), []).append(path.name)

    problems = []
    for url, sources in sorted(seen.items()):
        error = probe_url(url)
        if error:
            problems.append(f"{', '.join(sorted(set(sources)))}: {url} -> {error}")
        else:
            print(f"  ok   {url}")
    return problems


def probe_url(url: str, timeout: int = 20) -> str | None:
    """Return None if the URL resolves, else a short description of the failure."""
    last_error = None
    for method in ("HEAD", "GET"):
        request = urllib.request.Request(
            url, method=method, headers={"User-Agent": USER_AGENT}
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                if response.status < 400:
                    return None
                last_error = f"HTTP {response.status}"
        except urllib.error.HTTPError as exc:
            # Some hosts reject HEAD outright; retry with GET before believing them.
            if exc.code in (403, 405, 501) and method == "HEAD":
                last_error = f"HTTP {exc.code}"
                continue
            return f"HTTP {exc.code}"
        except urllib.error.URLError as exc:
            last_error = f"unreachable ({exc.reason})"
        except (TimeoutError, OSError) as exc:
            last_error = f"unreachable ({exc})"
    return last_error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--write",
        action="store_true",
        help="regenerate the table in community/README.md",
    )
    parser.add_argument(
        "--check-links", action="store_true", help="verify that every URL resolves"
    )
    parser.add_argument(
        "--links-only",
        action="store_true",
        help="only verify URLs, skip the index sync check",
    )
    args = parser.parse_args()

    try:
        entries = load_entries()
    except EntryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Found {len(entries)} community entr{'y' if len(entries) == 1 else 'ies'}.")

    problems = []
    for path, entry in entries:
        problems += validate_entry(path, entry)

    names = {}
    for path, entry in entries:
        key = str(entry.get("name", "")).strip().lower()
        if key and key in names:
            problems.append(
                f"{path.name}: duplicate 'name' — also used by {names[key]}"
            )
        elif key:
            names[key] = path.name

    if problems:
        print("\nSchema validation failed:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        print(
            "\nSee community/CONTRIBUTING.md for the field reference.",
            file=sys.stderr,
        )
        return 1
    print("Schema validation passed.")

    if not args.links_only:
        try:
            rendered = splice_index(render_table(entries))
        except EntryError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        current = INDEX_FILE.read_text(encoding="utf-8")
        if args.write:
            if rendered != current:
                INDEX_FILE.write_text(rendered, encoding="utf-8")
                print(f"Updated the table in community/{INDEX_FILE.name}.")
            else:
                print(f"community/{INDEX_FILE.name} is already up to date.")
        elif rendered != current:
            diff = difflib.unified_diff(
                current.splitlines(keepends=True),
                rendered.splitlines(keepends=True),
                fromfile=f"community/{INDEX_FILE.name} (committed)",
                tofile=f"community/{INDEX_FILE.name} (generated from entries/)",
            )
            print(
                f"\nerror: community/{INDEX_FILE.name} is out of date:\n",
                file=sys.stderr,
            )
            sys.stderr.writelines(diff)
            print(
                "\nRun 'python community/validate.py --write' and commit the result.\n"
                "If a line above looks hand-edited, change the YAML in "
                "community/entries/ instead — the table is generated.",
                file=sys.stderr,
            )
            return 1
        else:
            print(f"community/{INDEX_FILE.name} is up to date.")

    if args.check_links or args.links_only:
        print("\nChecking links...")
        link_problems = check_links(entries)
        if link_problems:
            print("\nUnreachable link(s):", file=sys.stderr)
            for problem in link_problems:
                print(f"  - {problem}", file=sys.stderr)
            return 1
        print("All links resolved.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
