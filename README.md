# A Lightweight Lexical-Metadata Hybrid for Duplicate Bug Report Retrieval

This repository contains the working code, experiment outputs, and paper materials for a research study on duplicate bug report retrieval in large software repositories.

The project investigates whether a lightweight hybrid model, built from lexical similarity and structured metadata, can improve candidate retrieval coverage without the computational overhead of heavier semantic architectures.

## Research Focus

Duplicate bug reports increase manual triage effort, fragment issue discussion, and slow software maintenance. This project explores a practical retrieval-oriented solution based on:

- **TF-IDF lexical similarity**
- **LSA as a comparative baseline**
- **Component-level metadata similarity**
- **A lightweight lexical-metadata hybrid model**

The current experiment uses a processed **10,000-report subset** from the MSR 2013 Eclipse and Mozilla defect tracking dataset. The study is framed as a controlled subset experiment rather than a full-corpus benchmark.

## Repository Structure

- `prepare_real_data.py` — prepares the working dataset from the source files
- `run_standard_duplicate_bug_benchmark.py` — main experiment script for TF-IDF, LSA, metadata-only, and hybrid retrieval
- `requirements.txt` — Python package requirements
- `results/` — experiment outputs and result files
- `paper/` — LaTeX source and compiled paper files

## Current Development Stages

This repository is being updated in stages to reflect continuous research progress:

1. project initialization and dependency setup
2. dataset preparation
3. baseline retrieval experiments
4. hybrid model development
5. result analysis
6. paper writing and revision

## Method Summary

The retrieval pipeline compares four configurations:

1. **TF-IDF baseline**  
   Sparse lexical similarity using TF-IDF vectors and cosine similarity.

2. **LSA baseline**  
   Latent semantic analysis using truncated SVD over the TF-IDF representation.

3. **Metadata-only model**  
   Component-level metadata matching used as a simple structured retrieval signal.

4. **Hybrid model**  
   A weighted fusion of normalized TF-IDF similarity and component-level metadata similarity.

The final hybrid configuration uses:

```text
0.85 × TF-IDF similarity + 0.15 × metadata similarity
