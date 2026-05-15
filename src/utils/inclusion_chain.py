import re
import json
import os
import unicodedata
from collections import OrderedDict

from docling.document_converter import DocumentConverter

converter = DocumentConverter()

def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.lower()
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'[^\w\s./:-]', '', s)
    return s.strip()

def extract_doi(text: str):
    m = re.search(r'\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b', text, flags=re.I)
    return m.group(0).lower() if m else None

def find_references_block(text: str):
    # Match markdown headings (##, #, **bold**) and plain lines
    pattern = re.compile(
        r'(?im)^\s*(?:#{1,6}\s*|\*{1,2})?'
        r'(references|bibliography|works cited|literature cited)'
        r'(?:\*{1,2})?\s*$'
    )
    m = pattern.search(text)
    if not m:
        return None

    refs = text[m.end():].strip()

    tail_pattern = re.compile(
        r'(?im)^\s*(?:#{1,6}\s*|\*{1,2})?'
        r'(appendix|supplementary materials?|acknowledg(?:e)?ments?|notes|about the authors?)'
        r'(?:\*{1,2})?\s*$'
    )
    t = tail_pattern.search(refs)
    if t:
        refs = refs[:t.start()].strip()

    return refs if refs else None

def split_references(ref_block: str):
    lines = ref_block.splitlines()

    chunks = []
    current = []

    # Matches [1], 1. 1) or author-year style like "Smith, J." at line start
    numbered_pat = re.compile(r'^\s*(\[\d+\]|\d+[.)]\s)')
    author_year_pat = re.compile(r'^\s*[A-Z][a-z]+,?\s+[A-Z]')

    found_numbered = any(numbered_pat.match(line) for line in lines)
    # Treat author-year only if majority of non-empty lines look like entries
    non_empty = [l for l in lines if l.strip()]
    found_author_year = (
        not found_numbered
        and len(non_empty) > 1
        and sum(1 for l in non_empty if author_year_pat.match(l)) / len(non_empty) > 0.4
    )

    if found_numbered:
        for line in lines:
            if numbered_pat.match(line):
                if current:
                    chunks.append(" ".join(current).strip())
                current = [line.strip()]
            else:
                if line.strip():
                    current.append(line.strip())
        if current:
            chunks.append(" ".join(current).strip())
    elif found_author_year:
        for line in lines:
            if author_year_pat.match(line):
                if current:
                    chunks.append(" ".join(current).strip())
                current = [line.strip()]
            else:
                if line.strip():
                    current.append(line.strip())
        if current:
            chunks.append(" ".join(current).strip())
    else:
        raw = re.split(r'\n\s*\n+', ref_block)
        chunks = [" ".join(x.splitlines()).strip() for x in raw if x.strip()]

    cleaned = []
    for c in chunks:
        c = re.sub(r'\s+', ' ', c).strip()
        if len(c) > 20:
            cleaned.append(c)

    return cleaned

def make_key(citation: str):
    doi = extract_doi(citation)
    if doi:
        return f"doi::{doi}"

    norm = normalize_text(citation)
    norm = re.sub(r'^(\[\d+\]|\d+[.)])\s*', '', norm)
    norm = re.sub(r'\s+', ' ', norm).strip()

    return f"text::{norm}"


if __name__ == "__main__":
    data_root = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "reference"
    )
    pdf_files = [os.path.join(root, file) for root, _, files in os.walk(data_root) 
                 for file in files if file.lower().endswith('.pdf')]

    unique = OrderedDict()
    for pdf_path in pdf_files:
        try:
            result = converter.convert(pdf_path)
            doc = result.document
            text = doc.export_to_markdown()

            ref_block = find_references_block(text)
            if not ref_block:
                print(f"[WARN] No references section found: {pdf_path}")
                continue

            refs = split_references(ref_block)

            for ref in refs:
                key = make_key(ref)
                if key not in unique:
                    unique[key] = {
                        "citation": ref,
                        "doi": extract_doi(ref),
                        "source_files": [pdf_path],
                    }
                else:
                    unique[key]["source_files"].append(pdf_path)

            print(f"[OK] {pdf_path}: {len(refs)} refs")
        except Exception as e:
            print(f"[ERROR] {pdf_path}: {e}")
            
    with open(os.path.join(data_root, "unique_citations.json"), "w", encoding="utf-8") as f:
        json.dump(list(unique.values()), f, ensure_ascii=False, indent=2)
    with open(os.path.join(data_root, "unique_citations.txt"), "w", encoding="utf-8") as f:
        for i, item in enumerate(unique.values(), 1):
            f.write(f"[{i}] {item['citation']}\n")
            
    print(f"Saved to: {data_root}")
    
    