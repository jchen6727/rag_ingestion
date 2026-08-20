# rag_ingestion

Turn a folder of clinical PDFs into a searchable, **tagged** corpus — and review
those tags. This tool reads each PDF, splits it into short passages ("chunks"),
labels each passage with clinical metadata (therapy type, presentation, cautions,
…) using Google's Gemini model, loads the result into Google Vertex AI Search, and
gives you simple tools to **inspect and visualize the tags**.

> **New here? Two starting points:**
> - To adjust *how passages get tagged* (no coding), read
>   **[INGESTION_FOR_CLINICIANS.md](INGESTION_FOR_CLINICIANS.md)**.
> - To *see the tags* on an already-processed corpus, jump to
>   [§5 Inspect & visualize the tags](#5-inspect--visualize-the-tags).

---

## What you need

- A computer with **Python 3.12** (or newer) and a terminal.
- For the tagging and cloud steps: a **Google Cloud project** with billing, and
  the `gcloud` command-line tool installed and logged in.
- You do **not** need a Google Cloud account just to *view* tags on an exported
  file (§5) — that runs on your own machine.

## 1. Install (once)

Open a terminal in this folder and run:

```bash
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only, smaller download
pip install plotly                                                    # only needed for the chart tools (§5)
```

## 2. Configure (once)

```bash
cp .env.example .env
```

Open `.env` in a text editor and fill in the lines for your Google Cloud project
(`GCP_PROJECT_ID`, `GCS_BUCKET_NAME`, `GCP_LOCATION`, `VERTEX_SEARCH_DATASTORE_ID`,
`VERTEX_SEARCH_ENGINE_ID`). The tool reads `.env` automatically — you don't need to
"source" it. Log in to Google:

```bash
gcloud auth login
gcloud auth application-default login
```

**To change how passages are tagged** (which is often the most useful knob for a
clinician), see **[INGESTION_FOR_CLINICIANS.md](INGESTION_FOR_CLINICIANS.md)** —
it walks through editing `config/ingestion_prompt.yaml` in plain language.

## 3. Preview one PDF's chunks and tags (safe, cheap)

Put a PDF in the `corpus/` folder, then:

```bash
# Chunks only, no cloud, no cost:
PYTHONPATH=. python scripts/inspect_chunks.py corpus/yourfile.pdf --no-metadata

# Chunks + AI tags for the first 15 passages (needs Google login):
PYTHONPATH=. python scripts/inspect_chunks.py corpus/yourfile.pdf --limit 15
```

This writes a readable report to `ingestion_review/…​.review.md` that you can open
and skim. Good for checking the tags look right before a full run.

Before a big run, confirm the AI model is reachable:

```bash
PYTHONPATH=. python scripts/check_llm.py
```

## 4. Full ingestion into the cloud

```bash
# a) Check your setup (prints exact fixes for anything missing):
scripts/preflight_check.sh

# b) Create the search index (once per project):
PYTHONPATH=. python scripts/setup_vertex_search.py

# c) Ingest (dry-run first — it tags but doesn't upload):
PYTHONPATH=. python scripts/batch_ingest.py --file corpus/yourfile.pdf --dry-run
PYTHONPATH=. python scripts/batch_ingest.py --file corpus/yourfile.pdf
```

Tips:
- A long book can take a while. Progress is printed as `…metadata X/N (Y%)`.
- If a run stops partway, **run the same command again** — it resumes from a
  checkpoint and won't re-tag (or re-pay for) chunks it already did.
- Add `--verbose` to any command to see full detail when debugging.

## 5. Inspect & visualize the tags

These three tools read an **exported chunks file** (`.jsonl`) — the file ingestion
uploads to Google Cloud Storage. Download it first (the `doc_id` is shown during
ingestion; its first 8 characters are enough to recognize):

```bash
gcloud storage cp "gs://$GCS_BUCKET_NAME/chunks/<doc_id>.jsonl" ./chunks.jsonl
```

Then:

```bash
# Read the passages that carry a given tag:
python scripts/view_chunks.py chunks.jsonl --filter therapeutic_modality=CBT --limit 5
python scripts/view_chunks.py chunks.jsonl --filter directionality=contraindicated --full

# Charts of how often each tag value appears (opens an interactive HTML file):
python scripts/metadata_viz.py chunks.jsonl --output tag_frequencies.html

# How often tag values co-occur, within a field (Jaccard similarity heatmap):
python scripts/metadata_jaccard.py chunks.jsonl --output tag_overlap.html
```

Open the generated `.html` files in a web browser. Add `--help` to any script to
see all options (e.g. `--fields` to pick which tags to chart).

> The chart tools (`metadata_viz.py`, `metadata_jaccard.py`) need `plotly`
> (`pip install plotly`). `view_chunks.py` needs nothing extra.

## 6. What's where

| Folder / file | What it holds |
|---|---|
| `corpus/` | put your source PDFs here |
| `config/rta_v1.json` | the list of allowed tags (the "schema") |
| `config/ingestion_prompt.yaml` | the plain-English tagging instructions (see the clinician guide) |
| `config/chunk_config.yaml` | how documents are split into passages |
| `ingestion/` | the pipeline (extract → chunk → tag → upload → index) |
| `scripts/` | the commands you run (ingest, inspect, visualize, provision) |
| `.env` | your project settings (you create this; never share it) |

## 7. Good to know

- **Keep `.env` private.** It points at your cloud project; never commit or share it.
- **Costs.** Tagging calls Gemini and indexing uses Vertex AI Search — both incur
  Google Cloud usage. The preview in §3 (`--limit`) keeps test costs small.
- **Reset the index** (careful — deletes indexed data): `PYTHONPATH=. python
  scripts/purge_datastore.py --confirm --dry-run` first, then without `--dry-run`.
- Run the test suite with `PYTHONPATH=. pytest tests/`.
