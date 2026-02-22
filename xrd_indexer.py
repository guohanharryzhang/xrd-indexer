#!/usr/bin/env python3
"""XRD Data Indexer — parse, characterize, and search XRDML files."""

import hashlib
import json
import math
import os
import re
import logging
import time

import xml.etree.ElementTree as ET

import numpy as np
import requests
import yaml
from scipy.signal import find_peaks

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def load_manifest(config):
    path = config["storage"]["manifest_file"]
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"samples": {}}


def save_manifest(manifest, config):
    path = config["storage"]["manifest_file"]
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)


def sample_id(name):
    norm = re.sub(r'[^\w\s-]', '', name.lower())
    norm = " ".join(norm.split())
    return hashlib.md5(norm.encode()).hexdigest()[:12]


def safe_filename(name):
    name = re.sub(r'[^\w\s-]', '', name.lower())
    name = re.sub(r'[\s]+', '_', name.strip())
    return name[:80]


# ── XRDML Parsing ──────────────────────────────────────────


def parse_xrdml(path):
    """Parse a PANalytical XRDML file directly from XML.

    Returns (two_theta, intensities, metadata) where:
    - two_theta: 1D numpy array of 2-theta angles
    - intensities: 1D numpy array of intensity values (counts/sec)
    - metadata: dict with scan info
    """
    tree = ET.parse(path)
    root = tree.getroot()

    # Detect namespace
    ns_uri = root.tag.split("}")[0].strip("{") if "}" in root.tag else ""
    ns = {"ns": ns_uri} if ns_uri else {}

    def find(el, tag):
        return el.find(f"ns:{tag}", ns) if ns else el.find(tag)

    def findtext(el, tag):
        return el.findtext(f"ns:{tag}", namespaces=ns) if ns else el.findtext(tag)

    def findall(el, tag):
        return el.findall(f"ns:{tag}", ns) if ns else el.findall(tag)

    # Sample ID
    sample_id = ""
    sample_el = find(root, "sample")
    if sample_el is not None:
        sid = findtext(sample_el, "id")
        if sid:
            sample_id = sid

    # First measurement
    meas = find(root, "xrdMeasurement")
    meas_type = meas.get("measurementType", "") if meas is not None else ""

    # First scan
    scan = find(meas, "scan") if meas is not None else None
    scan_axis = scan.get("scanAxis", "") if scan is not None else ""

    # Data points
    dp = find(scan, "dataPoints") if scan is not None else None

    two_theta = np.array([])
    intensities = np.array([])

    if dp is not None:
        # Parse positions — find 2Theta axis
        for pos in findall(dp, "positions"):
            if pos.get("axis") == "2Theta":
                start_el = findtext(pos, "startPosition")
                end_el = findtext(pos, "endPosition")
                if start_el and end_el:
                    start = float(start_el)
                    end = float(end_el)
                    # Number of points = number of intensity values
                    int_text = findtext(dp, "intensities")
                    if int_text:
                        n = len(int_text.strip().split())
                        two_theta = np.linspace(start, end, n)
                break

        # Parse intensities
        int_text = findtext(dp, "intensities")
        if int_text:
            intensities = np.array([float(x) for x in int_text.strip().split()])

            # Check units — convert counts to cps if needed
            int_el = find(dp, "intensities")
            units = int_el.get("unit", "") if int_el is not None else ""
            if units == "counts":
                time_text = findtext(dp, "commonCountingTime")
                if time_text:
                    counting_time = float(time_text)
                    if counting_time > 0:
                        intensities = intensities / counting_time

    metadata = {
        "sample": sample_id,
        "scan_axis": scan_axis,
        "measurement_type": meas_type,
    }

    if two_theta.size > 0:
        metadata["two_theta_start"] = float(round(two_theta[0], 4))
        metadata["two_theta_end"] = float(round(two_theta[-1], 4))
        metadata["step_size"] = float(round(np.mean(np.diff(two_theta)), 6)) if two_theta.size > 1 else 0.0
        metadata["num_points"] = int(two_theta.size)

    return two_theta, intensities, metadata


# ── Peak Detection ─────────────────────────────────────────


def detect_peaks(two_theta, intensities, config):
    """Find peaks in XRD data.

    Returns list of dicts with peak info: position (2θ), intensity, relative_intensity, d_spacing.
    """
    if two_theta.size == 0 or intensities.size == 0:
        return []

    peak_cfg = config.get("peaks", {})
    prominence = peak_cfg.get("prominence", 0.05)
    min_distance = peak_cfg.get("min_distance", 5)

    max_intensity = float(np.max(intensities))
    if max_intensity == 0:
        return []

    indices, properties = find_peaks(
        intensities,
        prominence=prominence * max_intensity,
        distance=min_distance,
    )

    peaks = []
    for idx in indices:
        theta_rad = np.radians(two_theta[idx] / 2.0)
        d_spacing = 1.5406 / (2.0 * np.sin(theta_rad)) if theta_rad > 0 else 0.0  # Cu Kα

        peaks.append({
            "two_theta": float(round(two_theta[idx], 4)),
            "intensity": float(round(intensities[idx], 2)),
            "relative_intensity": float(round(intensities[idx] / max_intensity * 100, 1)),
            "d_spacing": float(round(d_spacing, 4)),
        })

    # Sort by intensity descending
    peaks.sort(key=lambda p: p["intensity"], reverse=True)
    return peaks


# ── LLM Characterization ──────────────────────────────────


def build_characterization_prompt(name, metadata, peaks, stats):
    """Build a structured prompt for the LLM to describe the XRD pattern."""
    peak_lines = ""
    for i, p in enumerate(peaks[:20], 1):
        peak_lines += f"  {i}. 2θ = {p['two_theta']}°, d = {p['d_spacing']} Å, relative intensity = {p['relative_intensity']}%\n"

    if not peak_lines:
        peak_lines = "  No significant peaks detected.\n"

    return f"""You are an X-ray diffraction analyst. Describe the following XRD pattern. Do NOT attempt to identify specific materials or phases — only describe what the pattern shows.

Sample: {name}
Scan type: {metadata.get('scan_axis', 'unknown')}
2θ range: {metadata.get('two_theta_start', '?')}° to {metadata.get('two_theta_end', '?')}°
Step size: {metadata.get('step_size', '?')}°
Number of data points: {metadata.get('num_points', '?')}

Intensity statistics:
  Max intensity: {stats.get('max_intensity', '?')}
  Mean intensity: {stats.get('mean_intensity', '?')}
  Signal-to-noise ratio (max/mean): {stats.get('snr', '?')}

Detected peaks ({len(peaks)} total):
{peak_lines}
Describe:
1. Overall pattern characteristics (crystallinity, peak sharpness, background level)
2. Peak distribution across the 2θ range
3. Whether the pattern suggests single-phase or multi-phase material
4. Any notable features (preferred orientation, broad amorphous humps, peak splitting)
"""


def call_llm(prompt, config, max_retries=3):
    """Send a prompt to Ollama and return the response text."""
    url = f"{config['llm']['base_url']}/api/chat"
    payload = {
        "model": config["llm"]["model"],
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {
            "num_predict": config["llm"]["max_tokens"],
            "temperature": config["llm"]["temperature"],
        },
    }
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, json=payload, timeout=(10, 300))
            resp.raise_for_status()
            data = resp.json()
            return data["message"]["content"]
        except requests.RequestException as e:
            wait = 10 * (attempt + 1)
            log.warning(f"  LLM call failed (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                log.info(f"  Retrying in {wait}s...")
                time.sleep(wait)
        except (KeyError, IndexError) as e:
            log.error(f"  LLM response parse error: {e}")
            return None
    log.error(f"  LLM call failed after {max_retries} attempts")
    return None


def characterize_xrd(xrdml_path, config):
    """Full pipeline: parse → peaks → prompt → LLM description.

    Returns (sample_name, metadata, peaks, description, summary_md).
    """
    name = os.path.splitext(os.path.basename(xrdml_path))[0]

    log.info(f"Parsing {name}...")
    two_theta, intensities, metadata = parse_xrdml(xrdml_path)

    log.info(f"Detecting peaks...")
    peaks = detect_peaks(two_theta, intensities, config)

    stats = {}
    if intensities.size > 0:
        stats["max_intensity"] = float(round(np.max(intensities), 2))
        stats["mean_intensity"] = float(round(np.mean(intensities), 2))
        stats["snr"] = float(round(np.max(intensities) / np.mean(intensities), 2)) if np.mean(intensities) > 0 else 0.0

    log.info(f"Characterizing with LLM ({len(peaks)} peaks)...")
    prompt = build_characterization_prompt(name, metadata, peaks, stats)
    description = call_llm(prompt, config)

    summary_md = format_summary_markdown(name, metadata, peaks, stats, description)

    return name, metadata, peaks, description, summary_md


# ── Summary Formatting ─────────────────────────────────────


def format_summary_markdown(name, metadata, peaks, stats, description):
    """Format an XRD characterization as markdown."""
    lines = [f"# {name}", ""]

    lines.append("## Scan Metadata")
    lines.append("")
    lines.append(f"- **Scan axis:** {metadata.get('scan_axis', 'N/A')}")
    lines.append(f"- **Measurement type:** {metadata.get('measurement_type', 'N/A')}")
    lines.append(f"- **2θ range:** {metadata.get('two_theta_start', '?')}° – {metadata.get('two_theta_end', '?')}°")
    lines.append(f"- **Step size:** {metadata.get('step_size', '?')}°")
    lines.append(f"- **Data points:** {metadata.get('num_points', '?')}")
    lines.append("")

    lines.append("## Intensity Statistics")
    lines.append("")
    lines.append(f"- **Max intensity:** {stats.get('max_intensity', 'N/A')}")
    lines.append(f"- **Mean intensity:** {stats.get('mean_intensity', 'N/A')}")
    lines.append(f"- **Signal-to-noise (max/mean):** {stats.get('snr', 'N/A')}")
    lines.append("")

    lines.append(f"## Detected Peaks ({len(peaks)})")
    lines.append("")
    if peaks:
        lines.append("| # | 2θ (°) | d-spacing (Å) | Relative Intensity (%) |")
        lines.append("|---|--------|----------------|------------------------|")
        for i, p in enumerate(peaks[:30], 1):
            lines.append(f"| {i} | {p['two_theta']} | {p['d_spacing']} | {p['relative_intensity']} |")
    else:
        lines.append("No significant peaks detected.")
    lines.append("")

    lines.append("## LLM Characterization")
    lines.append("")
    lines.append(description if description else "*Characterization failed.*")
    lines.append("")

    return "\n".join(lines)


# ── Embeddings & Search ────────────────────────────────────


def embed_text(text, config):
    """Generate embedding via nomic-embed-text on Ollama."""
    model = config["llm"].get("embed_model", "nomic-embed-text")
    url = f"{config['llm']['base_url']}/api/embed"
    payload = {"model": model, "input": text}
    try:
        resp = requests.post(url, json=payload, timeout=(10, 120))
        resp.raise_for_status()
        data = resp.json()
        return data["embeddings"][0]
    except (requests.RequestException, KeyError, IndexError) as e:
        log.error(f"  Embedding failed: {e}")
        return None


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def build_embedding_index(config, manifest):
    """Build vector embeddings for all characterized samples."""
    index_path = config["storage"]["index_file"]
    summaries_dir = config["storage"]["summaries_dir"]
    entries = []

    samples_with_summaries = [
        (sid, s) for sid, s in manifest["samples"].items()
        if s.get("status") == "characterized" and s.get("summary_file")
    ]

    log.info(f"Building embedding index for {len(samples_with_summaries)} samples...")

    for i, (sid, sample) in enumerate(samples_with_summaries):
        summary_file = sample["summary_file"]
        if not os.path.exists(summary_file):
            log.warning(f"  Summary file missing: {summary_file}")
            continue

        with open(summary_file) as f:
            content = f.read()

        vec = embed_text(content, config)
        if vec is None:
            log.warning(f"  Failed to embed: {sample['name']}")
            continue

        entries.append({
            "sid": sid,
            "name": sample["name"],
            "summary_file": summary_file,
            "peak_count": sample.get("peak_count", 0),
            "scan_range": sample.get("scan_range", ""),
            "embedding": vec,
        })

        log.info(f"  [{i+1}/{len(samples_with_summaries)}] Embedded: {sample['name']}")

    with open(index_path, "w") as f:
        json.dump(entries, f)

    log.info(f"Embedding index saved to {index_path} ({len(entries)} entries)")
    return entries


def load_embedding_index(config):
    index_path = config["storage"]["index_file"]
    if not os.path.exists(index_path):
        return None
    with open(index_path) as f:
        return json.load(f)


def query_samples(question, config, top_k=5):
    """Semantic search + LLM-generated answer (RAG)."""
    entries = load_embedding_index(config)
    if not entries:
        return None, "No embedding index found. Upload and characterize samples first."

    log.info(f"Searching {len(entries)} samples for: {question}")

    q_vec = embed_text(question, config)
    if q_vec is None:
        return None, "Failed to embed question."

    scored = []
    for entry in entries:
        sim = cosine_similarity(q_vec, entry["embedding"])
        scored.append((sim, entry))
    scored.sort(key=lambda x: x[0], reverse=True)

    top = scored[:top_k]

    context_parts = []
    for i, (sim, entry) in enumerate(top):
        if os.path.exists(entry["summary_file"]):
            with open(entry["summary_file"]) as f:
                content = f.read()
            context_parts.append(f"--- Sample {i+1}: {entry['name']} (similarity: {sim:.3f}) ---\n{content}")

    context = "\n\n".join(context_parts)
    prompt = f"""Based on the following XRD sample characterizations, answer this question: {question}

{context}

Answer the question citing specific sample names. If none of the samples match, say so."""

    answer = call_llm(prompt, config)
    return top, answer
