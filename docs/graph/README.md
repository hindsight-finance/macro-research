# Concept Graph (Obsidian vault)

An [[Concept Map|interlinked map]] of the concepts and experiments covered in this
research workspace, designed to be browsed as an **Obsidian graph**.

## How to open

Open the repo's **`docs/`** directory as an Obsidian vault (Open folder as vault →
select `docs/`). Obsidian builds the graph automatically from the `[[wikilinks]]` in
these notes (graph view: Ctrl/Cmd-G). Three things wire the graph together:

- **Concept nodes** (`docs/graph/concepts/`) — durable, reusable ideas (a feature, a
  window, an indicator). Some are **variants** of a broader concept (e.g.
  [[macro-open-1550]] is a variant of [[macro-window]]).
- **Experiment nodes** (`docs/graph/experiments/`) — one per study/thread. Each links
  to the concepts it uses and to its source artifacts (spec / plan / report / formal log).
- **Tags & properties** (YAML frontmatter) — the category→variant→params hierarchy that
  would clutter the graph if it were all nodes (see "Tag taxonomy & params" below).

Because the vault root is `docs/`, an experiment node's links resolve to the existing
`docs/experiments/`, `docs/reports/`, `docs/plans/`, and `docs/superpowers/{specs,plans}/`
files by basename — so those existing documents appear in the graph without being edited.

Start at [[Concept Map]].

## Relationship to the formal experiment log

This graph is a **navigational/visual layer**, not a replacement for the formal
experiment-log protocol in `AGENTS.md`. Fully-written logs still live in
`docs/experiments/NNNN-short-name.md` and are indexed in `docs/research_log.md`.
Experiment nodes here are concise summaries that **link out** to those canonical
artifacts rather than duplicating their detail or numbers.

## Tag taxonomy & params (the granularity model)

Three conceptual levels, mapped to three mechanisms so the graph stays readable:

| Level | Example | Represented as |
| --- | --- | --- |
| **Category** | VWAP, volume delta, ADX | a **concept node** |
| **Variant** | 3:50pm VWAP, `eth_rth_pre59` imbalance | a **nested tag** (`#feature/vwap/anchor-1550`); promoted to a node only when ≥2 experiments share it |
| **Param** | `side_threshold_pts: 0.25`, `bucket_size: 5s` | a **flat frontmatter property** — never a node |

**Why params are not nodes:** they are degree-1 leaves (each links only to its one
experiment) and multiply fastest where a study has the least detail — they'd bury the
concept↔experiment structure under noise. As frontmatter they stay Dataview-queryable and
graph-filterable without adding nodes.

### Tag registry (use these slugs; nesting via `/`)

```
asset/nq
window/{macro, macro/open-1550, macro/close-1559, h3pm, post, pm}
feature/vwap/{anchor-0930, anchor-1300, anchor-1500, anchor-1550, anchor-1555, retouch-1550}
feature/volume-delta/{imbalance-pre350, imbalance-pre59, bucket-5s, bucket-1m}
feature/fvg/{alignment, excursion, success-context, minute-volume, delta-dominance}
feature/{tick-density, barrier, mae-mfe, outcome}
feature/barrier/{break-direction}
feature/trend/{adx, atr, dra, irr, lag-hurst, mss, spd, efficiency-ratio,
               variance-ratio, containment, trendability, state-detector, regime}
infra/{time, pipeline, ticks}
```

### Flat param keys (Obsidian-valid; only include those that apply)

`anchors`, `checkpoints`, `predictors`, `targets`, `support_windows`, `candles`,
`residual_windows`, `bins`, `conviction`, `prior_windows`, `bucket_size`,
`side_threshold_pts`, `sample_n`, `lookback`, `era_filters`, `model`.

> **Obsidian schema rule:** frontmatter properties must be **flat** (Text / List / Number /
> Checkbox / Date). A nested map (`params:` with sub-keys) is *not* a valid Obsidian property
> and won't render — keep every param a top-level key.

### Querying params with Dataview

With the Dataview community plugin, a params table across a feature family is one query:

````markdown
```dataview
TABLE anchors, checkpoints, targets, sample_n
FROM #feature/vwap AND "graph/experiments"
SORT file.name
```
````

## Conventions

- **Filenames are node labels** — kebab-case, no dates, globally unique basenames.
- **Concepts** in `concepts/`, **experiments** in `experiments/`.
- **Frontmatter**: `type: concept|experiment`; experiments also carry `status:`
  (`Exploratory | Validated | Superseded | Deprecated`), nested `tags:`, and flat param keys.
- **Link by basename**: `[[anchored-vwap]]`, `[[2026-05-15-macro-vwap-features-design]]`.
- **Don't invent findings.** Experiment summaries state scope and link to the source artifact
  for numbers; only headline metrics committed in a report/log get quoted.

## Maintenance protocol

When a new concept or study is covered, **before considering the work done**:

1. **New study** → add `experiments/<slug>.md` from the template; set `tags:` (category +
   variants the study touches) and flat param keys; link the concepts + source artifacts; add a
   row to [[Concept Map]].
2. **New reusable idea** → add `concepts/<slug>.md`; add its slug to the registry, a link from
   [[Concept Map]], and its tag to the registry above; backlink it from the experiments using it.
3. **New variant** → start as a nested tag; promote to a `concepts/<slug>.md` node only once a
   second experiment uses it, linking it as a `**Variant of.**` the parent concept.
4. **Status change / supersede** → update `status:` and add `Supersedes:` / `Superseded by:`
   wikilinks so the graph shows lineage.
5. Keep it additive and reversible; never rewrite an existing canonical artifact to fit a node.

### Experiment node template

```markdown
---
type: experiment
status: Exploratory
tags: [asset/nq, window/macro, window/macro/open-1550, feature/volume-delta]
predictors: [eth_rth_pre59]
targets: [k359_50_59]
bucket_size: 5s
sample_n: 841
---
# <Study name>

**Question.** <one sentence>

**Scope.** <asset / window / sample size if known>

**Headline.** <one-line finding, only if a committed report/log states it; else "see source artifacts">

**Concepts.** [[macro-window]] · [[macro-close-1559]] · [[volume-delta]]

**Artifacts.** [[<spec basename>]] · [[<plan basename>]] · [[<report basename>]]
```
