# MBSE Copilot

A lightweight prototype for **MBSE retrieval and question answering** on **real model data**.

The current focus is not a generic chatbot pipeline, but a retrieval system that can better answer structure-aware questions over MBSE artifacts, such as:

- which package an element belongs to
- which diagram contains an element
- which higher-level system a block belongs to
- what elements a package contains
- what properties or relations an element has

The project is evolving from simple embedding search toward a more structured pipeline:

**question understanding -> retrieval planning -> evidence retrieval -> answer generation**

---

## Current focus

The main work in this repo is now centered on **real-data testing**.

Instead of only testing synthetic XML examples, the project is moving toward:

- retrieval over real exported model chunks
- query understanding before retrieval
- type-aware and structure-aware ranking
- better handling of MBSE-specific questions

A key idea is:

> do not directly embed the raw user question and search everything;
> first understand what the question is asking, then transform it into a retrieval-oriented query, then retrieve evidence, and finally generate the answer.

---

## Repository structure

```text
api/        FastAPI service
data/       raw and processed data
docs/       notes and documentation
eval/       evaluation scripts and query sets
ingest/     parsing, indexing, and data preparation
tools/      local testing scripts for retrieval / QA