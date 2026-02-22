#!/usr/bin/env python3
"""Gradio web UI for the XRD Data Indexer."""

import json
import logging
import os
import shutil
import sys
import time

import gradio as gr
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xrd_indexer as xi

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")


def load_config():
    return xi.load_config(CONFIG_PATH)


def progress_bar(current, total, width=30):
    if total == 0:
        return ""
    pct = current / total
    filled = int(width * pct)
    bar = "\u2588" * filled + "\u2591" * (width - filled)
    return f"`{bar}` **{current}/{total}** ({pct:.0%})"


# ── Upload & Analyze tab ───────────────────────────────────


def upload_and_analyze(files):
    """Process uploaded XRDML files: parse, detect peaks, characterize, embed."""
    if not files:
        yield "Please upload at least one XRDML file."
        return

    config = load_config()
    manifest = xi.load_manifest(config)
    summaries_dir = config["storage"]["summaries_dir"]
    upload_dir = config["storage"]["upload_dir"]
    os.makedirs(summaries_dir, exist_ok=True)
    os.makedirs(upload_dir, exist_ok=True)

    total = len(files)
    processed = 0

    for file_path in files:
        basename = os.path.basename(file_path)
        if not basename.lower().endswith(".xrdml"):
            continue

        processed += 1
        name = os.path.splitext(basename)[0]
        sid = xi.sample_id(name)

        # Skip already characterized
        if sid in manifest["samples"] and manifest["samples"][sid].get("status") == "characterized":
            yield (
                f"## Processing uploads\n\n"
                f"{progress_bar(processed, total)}\n\n"
                f"*{name}* already indexed, skipping."
            )
            continue

        yield (
            f"## Processing uploads\n\n"
            f"{progress_bar(processed, total)}\n\n"
            f"Parsing *{name}*..."
        )

        # Copy to uploads dir
        dest = os.path.join(upload_dir, basename)
        if not os.path.exists(dest):
            shutil.copy2(file_path, dest)

        try:
            sample_name, metadata, peaks, description, summary_md = xi.characterize_xrd(dest, config)
        except Exception as e:
            yield (
                f"## Processing uploads\n\n"
                f"{progress_bar(processed, total)}\n\n"
                f"**Error** processing *{name}*: {e}"
            )
            continue

        yield (
            f"## Processing uploads\n\n"
            f"{progress_bar(processed, total)}\n\n"
            f"Saving summary for *{name}*..."
        )

        # Save summary markdown
        filename = f"{xi.safe_filename(name)}_{sid}.md"
        filepath = os.path.join(summaries_dir, filename)
        with open(filepath, "w") as f:
            f.write(summary_md)

        scan_range = f"{metadata.get('two_theta_start', '?')}–{metadata.get('two_theta_end', '?')}"

        manifest["samples"][sid] = {
            "name": sample_name,
            "xrdml_file": dest,
            "summary_file": filepath,
            "status": "characterized",
            "peak_count": len(peaks),
            "scan_range": scan_range,
            "metadata": metadata,
            "date": time.strftime("%Y-%m-%d %H:%M"),
        }
        xi.save_manifest(manifest, config)

    # Build embedding index
    yield (
        f"## Building search index\n\n"
        f"Processed **{processed}** files. Updating embeddings..."
    )

    try:
        xi.build_embedding_index(config, manifest)
    except Exception as e:
        yield f"## Embedding failed\n\n{e}"
        return

    index_path = config["storage"]["index_file"]
    index_count = 0
    if os.path.exists(index_path):
        with open(index_path) as f:
            index_count = len(json.load(f))

    yield (
        f"## Upload complete!\n\n"
        f"- **{processed}** XRDML files processed\n"
        f"- **{index_count}** total samples in search index\n\n"
        f"Switch to the **Search** tab to query your samples."
    )


# ── Browse Library tab ─────────────────────────────────────


def browse_library():
    """Return a table of all indexed samples."""
    config = load_config()
    manifest = xi.load_manifest(config)

    if not manifest["samples"]:
        return "No samples indexed yet. Upload XRDML files in the Upload tab."

    rows = []
    for sid, sample in manifest["samples"].items():
        rows.append([
            sample.get("name", ""),
            sample.get("scan_range", ""),
            sample.get("peak_count", 0),
            sample.get("date", ""),
            sid,
        ])

    # Sort by date descending
    rows.sort(key=lambda r: r[3], reverse=True)
    return rows


def view_sample_detail(evt: gr.SelectData, table_data):
    """When a row is clicked, show the full summary."""
    if table_data is None or evt.index[0] >= len(table_data):
        return "Select a sample from the table."

    row = table_data[evt.index[0]]
    sid = row[4]

    config = load_config()
    manifest = xi.load_manifest(config)
    sample = manifest["samples"].get(sid)
    if not sample:
        return f"Sample {sid} not found in manifest."

    summary_file = sample.get("summary_file", "")
    if summary_file and os.path.exists(summary_file):
        with open(summary_file) as f:
            return f.read()

    return "Summary file not found."


# ── Search tab ─────────────────────────────────────────────


def search_samples(query):
    if not query.strip():
        return "Please enter a search query."

    config = load_config()
    t_start = time.time()

    top, answer = xi.query_samples(query, config)

    if top is None:
        return answer  # error message

    result = f"**Top {len(top)} matches:**\n\n"
    for i, (sim, entry) in enumerate(top):
        name = entry["name"]
        summary_part = ""
        if os.path.exists(entry["summary_file"]):
            with open(entry["summary_file"]) as f:
                content = f.read()
            if "## LLM Characterization" in content:
                summary_part = content.split("## LLM Characterization")[1].strip()
            else:
                summary_part = content[:500]

        result += f"<details><summary><strong>{i+1}. {name}</strong> ({sim:.0%} match)</summary>\n\n"
        if summary_part:
            result += f"{summary_part}\n\n"
        result += "</details>\n\n"

    if answer:
        result += f"---\n\n**Answer:**\n\n{answer}"

    elapsed = time.time() - t_start
    result += f"\n\n*Search took {elapsed:.1f}s*"
    return result


# ── Build UI ───────────────────────────────────────────────


css = """
.gradio-container { max-width: 100% !important; padding: 0 2rem !important; }
.contain { max-width: 100% !important; }
"""

with gr.Blocks(title="XRD Data Indexer") as demo:
    gr.Markdown("# XRD Data Indexer\nUpload XRDML files, characterize diffraction patterns, and search your library.")

    with gr.Tabs():
        # ── Upload Tab ──
        with gr.TabItem("Upload & Analyze"):
            gr.Markdown("Upload `.xrdml` files. The filename is used as the sample name.")
            upload_box = gr.File(
                label="Upload XRDML files",
                file_types=[".xrdml"],
                file_count="multiple",
            )
            upload_btn = gr.Button("Analyze", variant="primary")
            upload_status = gr.Markdown("Ready. Upload XRDML files and press Analyze.")

            upload_btn.click(
                fn=upload_and_analyze,
                inputs=upload_box,
                outputs=upload_status,
            )

        # ── Browse Tab ──
        with gr.TabItem("Browse Library"):
            refresh_btn = gr.Button("Refresh", variant="secondary")
            sample_table = gr.Dataframe(
                headers=["Name", "Scan Range", "Peaks", "Date", "ID"],
                datatype=["str", "str", "number", "str", "str"],
                column_count=(5, "fixed"),
                interactive=False,
            )
            sample_detail = gr.Markdown("Click a row to view the full characterization.")

            refresh_btn.click(fn=browse_library, outputs=sample_table)
            sample_table.select(fn=view_sample_detail, inputs=sample_table, outputs=sample_detail)

            # Load on tab open
            demo.load(fn=browse_library, outputs=sample_table)

        # ── Search Tab ──
        with gr.TabItem("Search"):
            with gr.Row():
                query_box = gr.Textbox(
                    placeholder='e.g. "peak around 34 degrees" or "high crystallinity" or "narrow FWHM"',
                    label="Search",
                    scale=4,
                )
                search_btn = gr.Button("Search", variant="primary", scale=1)
            results = gr.Markdown()

            search_btn.click(fn=search_samples, inputs=query_box, outputs=results)
            query_box.submit(fn=search_samples, inputs=query_box, outputs=results)

            gr.Examples(
                examples=[
                    "Which samples show high crystallinity?",
                    "Samples with a peak around 34 degrees",
                    "Broad amorphous features",
                    "Narrow sharp peaks with high signal-to-noise",
                ],
                inputs=query_box,
            )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861, css=css)
