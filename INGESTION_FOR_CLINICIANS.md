# INGESTION_FOR_CLINICIANS.md — Configuring How Passages Get Tagged

**Who this is for:** clinicians (no coding needed). You'll learn how to adjust the
plain-English instructions the AI follows when it labels ("tags") each passage of
a document during ingestion.
**What you'll edit:** one text file — `config/ingestion_prompt.yaml`.
**You cannot break anything permanently:** if an edit is malformed, the system
quietly falls back to the built-in instructions and keeps working.

---

## 1. What "tagging" is, in one paragraph

When a document is ingested, it's split into short passages. The AI reads each
passage and attaches clinical **tags** — for example: which therapy it's about
(CBT, DBT, …), which presentation (depression, trauma, …), whether it describes
something to *do*, *avoid*, or *use caution* with, and any stated cautions. Those
tags are what let the system later find the right guidance. The **ingestion
prompt** is the set of instructions that tells the AI *how* to decide those tags.

## 2. The two halves of the prompt (only one is yours to edit)

The instructions the AI receives have two parts:

1. **The vocabulary** — the exact list of allowed tag values (e.g. the therapy
   names, the presentation names). **This is generated automatically** from the
   official schema. You do **not** edit it here, and you should **not** paste tag
   names into your file — doing so risks them drifting out of sync.
2. **The guidance** — plain-English coaching on *how to choose* among those tags:
   definitions, how to tell two similar cases apart, what never to miss. **This is
   the part you edit**, in `config/ingestion_prompt.yaml`.

Think of it as: the schema decides *what labels exist*; your guidance decides
*how carefully and consistently they get applied*.

## 3. The file you edit

Open `config/ingestion_prompt.yaml` in any text editor. It has five parts:

| Key | What it is | Edit it? |
|---|---|---|
| `system_preamble` | One line telling the AI its role | Rarely |
| `output_instruction` | Tells the AI to answer in a strict format | Leave as-is |
| `allowed_values_header` | A heading printed before the auto-generated tag list | Leave as-is |
| `guidance` | **The clinical coaching bullets — your main edit point** | **Yes** |
| `closing_instruction` | Final "answer in the required format" line | Leave as-is |

## 4. Editing the `guidance` bullets (the useful part)

The `guidance:` section is a list. Each line starting with `- "..."` is one piece
of advice. You can reword existing bullets, add new ones, or remove ones you
disagree with.

**Example — adding a disambiguation rule.** Say clinicians keep seeing "rupture"
and "crisis" tagged inconsistently. You'd add a bullet:

```yaml
guidance:
  - "- When a passage describes a strain in the therapeutic relationship, tag it as a rupture; reserve crisis language for acute safety escalation (self-harm, danger). If both are present, note both."
```

**Example — emphasizing a safety point:**

```yaml
  - "- If a passage says a technique should be avoided under certain conditions, always capture that as a caution — never describe the technique as simply recommended."
```

**Rules of thumb for good bullets:**
- Write one clear idea per bullet, the way you'd coach a new trainee.
- Describe *how to decide*, not *what the labels are* (the labels come from the
  schema automatically).
- Keep each bullet to a sentence or two.
- Start each bullet's text with `- ` inside the quotes (that's just how it's
  formatted in the file — match the existing lines).

**What to avoid:**
- Don't paste lists of tag names/values — that's the schema's job.
- Don't remove the quotation marks or the leading `-` list markers.
- Don't rename the five keys in section 3.

## 5. Formatting safety (so an edit doesn't silently do nothing)

This file is YAML, which is picky about quotes and indentation. Two safeguards:

1. Keep each bullet wrapped in straight double quotes: `- "your text here"`.
   Avoid "smart quotes" from word processors — use a plain text editor.
2. If the file ever can't be read, the system logs a warning and uses the
   built-in default instructions, so ingestion never stops. That also means: if
   your change *seems* to have no effect, the file may have a formatting error —
   check the run's log for a "Could not load ingestion prompt config" warning.

## 6. Seeing the effect of your change (before spending on a full run)

You don't have to run a whole ingestion to check your wording. Ask a developer (or
follow the README) to run the **preview** on one document:

```
PYTHONPATH=. python scripts/inspect_chunks.py corpus/<some_file>.pdf --limit 15
```

That tags the first 15 passages and writes a readable report you can open
(`ingestion_review/…​.review.md`). Read a few passages and check the tags reflect
your guidance. Adjust the bullets and re-run until it reads right.

## 7. Where the tags themselves are defined (a heavier, separate change)

If you find you need a tag value that doesn't exist at all — a new presentation,
a new in-session event, a new patient-state — that's a change to the **schema**
(`config/rta_v1.json`), not the prompt. That's a bigger, reviewed change (it can
require re-processing the corpus) and should go through a developer with clinical
sign-off, because those tag lists are safety-relevant. Use the `guidance` file for
*how tags are applied*; use the schema for *which tags exist*.

---

*Questions about a specific tag's meaning, or think a value is missing? Raise it
with the developer team — the tag definitions live in `config/rta_v1.json` and
changes to modalities and safety flags get extra review.*
