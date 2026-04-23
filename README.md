# A Lightweight Lexical-Metadata Hybrid for Duplicate Bug Report Retrieval

This repository contains the working code, experiment outputs, and paper materials for a research study on duplicate bug report retrieval in large software repositories.

The project investigates whether a lightweight hybrid model, built from lexical similarity and structured metadata, can provide competitive retrieval quality without the computational overhead of heavier semantic architectures.

## Research Focus

Duplicate bug reports increase manual triage effort, fragment issue discussion, and slow software maintenance. This project explores a practical retrieval-oriented solution based on:

- **TF-IDF lexical similarity**
- **LSA as a comparative baseline**
- **Component-level metadata similarity**
- **A lightweight lexical-metadata hybrid model**

The work is being developed incrementally, and the repository is organized to reflect that progression.

## Repository Structure

- `prepare_real_data.py` — prepares the working dataset from the source files
- `run_standard_duplicate_bug_benchmark.py` — main experiment script for baseline and hybrid retrieval
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

Future work will extend the baseline retrieval pipeline with a lightweight lexical-metadata hybrid model.

The model is evaluated using:

- MRR
- Hit@1
- Recall@5
- Recall@10

## Notes

This repository is intended to document the ongoing development of the project, including intermediate steps, revised experiment settings, and paper updates.

## Author

**Shah Nadim Kamran Rian**  
Department of Electrical and Computer Engineering  
North South University


