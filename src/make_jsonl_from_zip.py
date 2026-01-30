#!/usr/bin/env python3
"""
Parse a ZIP of concept-description files into a JSONL.

Each file inside the ZIP has content like:

CONCEPT: 100146
TITLE: 16 Economics
MODEL: gpt-4o-mini
==================================================
<LLM response text here ... possibly multi-line>

We output JSONL lines with:
{"concept_id": "...", "label": "...", "LLM_Response_text": "..."}

Usage:
  python make_jsonl_from_zip.py /path/to/input.zip -o /path/to/out.jsonl

"""
import argparse
import io
import json
import re
import sys
import zipfile
from pathlib import Path


SEPARATOR_REGEX = re.compile(r'\r?\n={3,}\r?\n')  # line with 3+ '=' surrounded by newlines
CONCEPT_REGEX = re.compile(r'^\s*CONCEPT:\s*(.+?)\s*$', re.MULTILINE)
TITLE_REGEX   = re.compile(r'^\s*TITLE:\s*(.+?)\s*$', re.MULTILINE)


def decode_bytes(b: bytes) -> str:
    """Decode bytes with utf-8, fallback to latin-1."""
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        return b.decode("latin-1", errors="replace")


def parse_one_document(text: str):
    """
    Return (concept_id, title, llm_text) or (None, None, None) if not parseable.
    """
    # Normalize newlines
    t = text.replace('\r\n', '\n').replace('\r', '\n')

    # Split header/body on a separator line of ===
    parts = SEPARATOR_REGEX.split(t, maxsplit=1)
    if len(parts) < 2:
        return None, None, None

    header, body = parts[0], parts[1].strip()

    # Extract CONCEPT and TITLE from header
    m_concept = CONCEPT_REGEX.search(header)
    m_title = TITLE_REGEX.search(header)

    concept_id = m_concept.group(1).strip() if m_concept else None
    title = m_title.group(1).strip() if m_title else None

    if not concept_id or not title or not body:
        return None, None, None

    return concept_id, title, body


def process_zip(zip_path: Path):
    """
    Yield dicts with keys: concept_id, label, LLM_Response_text
    for each parseable file inside the zip.
    """
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for member in zf.namelist():
            # Skip directories
            if member.endswith('/'):
                continue

            # Only try plausible text-y files
            low = member.lower()
            if not low.endswith(('.json', '.txt', '.md', '.log')):
                continue

            try:
                with zf.open(member, 'r') as f:
                    raw = f.read()
                text = decode_bytes(raw)
                concept_id, title, llm = parse_one_document(text)
                if concept_id and title and llm:
                    yield {
                        "concept_id": concept_id,
                        "label": title,
                        "LLM_Response_text": llm,
                    }
            except Exception as e:
                # Skip unreadable members but print a warning
                print(f"[WARN] Failed to parse '{member}': {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Build JSONL from concept-description ZIP.")
    parser.add_argument(
        "zip_path",
        nargs="?",
        default="gpt_eurolex_4_to_5.zip",
        help="Path to the input ZIP file."
    )
    parser.add_argument(
        "-o", "--output",
        default="gpt_4_5_labels_descriptions.jsonl",
        help="Path to the output JSONL file."
    )
    args = parser.parse_args()

    zip_path = Path(args.zip_path)
    out_path = Path(args.output)

    if not zip_path.exists():
        print(f"[ERROR] ZIP not found: {zip_path}", file=sys.stderr)
        sys.exit(1)

    count = 0
    with out_path.open("w", encoding="utf-8") as out:
        for rec in process_zip(zip_path):
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1

    print(f"[OK] Wrote {count} records to {out_path}")


if __name__ == "__main__":
    main()
