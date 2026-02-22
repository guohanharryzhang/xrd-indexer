# XRD Data Indexer

Upload XRDML files, characterize X-ray diffraction patterns with local LLMs, and search your library via semantic search.

## How It Works

1. **Parse** PANalytical XRDML files — extract 2θ/intensity data and scan metadata
2. **Detect peaks** using `scipy.signal.find_peaks` — positions, d-spacings, relative intensities
3. **Characterize** via local LLM (`llama3.2:1b` on Ollama) — describes pattern features without identifying materials
4. **Embed** summaries with `nomic-embed-text` for semantic search
5. **Search** your library with natural language queries (RAG)

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com/) running locally with:
  - `ollama pull llama3.2:1b`
  - `ollama pull nomic-embed-text`

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Launch the Gradio UI:

```bash
python query_ui.py
```

Then open **http://localhost:7861** in your browser.

### Tabs

| Tab | Description |
|-----|-------------|
| **Upload & Analyze** | Upload `.xrdml` files. Filename becomes the sample name. Shows metadata, peaks, and LLM characterization. |
| **Browse Library** | Table of all indexed samples. Click a row to view the full characterization. |
| **Search** | Natural language queries like "peak around 34 degrees" or "high crystallinity". Returns matching samples with similarity scores and an LLM-generated answer. |

## Configuration

Edit `config.yaml` to adjust:

```yaml
llm:
  base_url: http://localhost:11434   # Ollama URL
  model: llama3.2:1b                 # LLM for characterization
  embed_model: nomic-embed-text      # Embedding model
  max_tokens: 512
  temperature: 0.3

peaks:
  prominence: 0.05      # min peak prominence (fraction of max)
  min_distance: 5        # min distance between peaks (data points)
```

## Project Structure

```
xrd_indexer.py      # Core backend — parsing, peaks, LLM, embeddings, search
query_ui.py         # Gradio web UI
config.yaml         # Configuration
requirements.txt    # Python dependencies
```
