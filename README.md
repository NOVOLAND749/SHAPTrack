# SHAPTrack

SHAPTrack is a two-stage explainable machine-learning framework (XGBoost + SHAP) for multi-omics integration. A
genomics-only model is trained on pre-selected SNPs, then a joint model on the same SNPs plus all metabolites. The share
of SHAP importance each SNP retains when metabolites are added separates SNPs whose effects are carried by metabolites
(losing importance, followed up with mediation analysis) from SNPs whose effects are masked by metabolites (gaining
importance, followed up with suppression-type mediation and SNP × metabolite interaction analysis).

This repository contains the code for the analysis of LDL cholesterol (LDL-C) in the Canadian Longitudinal Study on
Aging (CLSA): genotype dosages and 1,257 Metabolon plasma metabolites in 8,953 participants. The citation will be added on
publication.

## Data

No data are included. The analysis needs:

| Input | Source |
|---|---|
| Genotypes (imputed bgen, sample QC, relatedness), Metabolon metabolomics, baseline lipids and lipid-lowering medication, age, sex | CLSA, through the CLSA data access process (<https://www.clsa-elcv.ca/data-access>) |
| GLGC 2021 LDL-C summary statistics (trans-ancestry, INV) | <https://csg.sph.umich.edu/willer/public/glgc-lipids2021/> |
| rsID and gene annotation | SNPnexus (<https://www.snp-nexus.org>, GRCh38), UCSC dbSNP 141 on hg38, PLINK hg38 gene list |

Every input path is set in `config.yaml` (`paths` and `gwas`). Inputs are expected under `data/`, intermediate files
are written to `derived/` and results to `results/`; none of these folders is tracked.

## Installation

```bash
conda env create -f environment.yml
conda activate shaptrack
```

PLINK 2 is needed for the GWAS step (`gwas.plink2` in the config). Analyses were run on Linux (CPU only).

## Configuration

All parameters (splits, seeds, pre-selection, XGBoost settings, label thresholds, multiple-testing settings) live in
`config.yaml`; scripts read no hard-coded values. To use other file locations, copy the file, edit `paths` and `gwas`,
and set `SHAPTRACK_CONFIG=<your copy>` (copies named `config.local.yaml` are ignored by git).

## Running

`run_all.sh` runs the analysis in the order used for the manuscript. Each stage can be run on its own:

```bash
bash run_all.sh outcome      # corrected LDL-C
bash run_all.sh gwas         # candidate SNPs from a GWAS of corrected LDL-C
bash run_all.sh data         # analysis file, rsID map, GLGC labels, MAF
bash run_all.sh cv           # leakage tests, smoke test, nested CV for k
bash run_all.sh splits       # 20 splits: genomics-only and joint models, SHAP
bash run_all.sh downstream   # labels, mediation, interaction
bash run_all.sh figures      # Figures 2-3, Supplementary Table S1
```

The rsID map needs a SNPnexus result (`paths.snpnexus_previous`): `scripts/make_snpnexus_upload.py` writes the upload
file; run the batch query on the SNPnexus website (assembly GRCh38). SNPs without a SNPnexus rsID are annotated from dbSNP
141 by `scripts/make_rsid_map.py`.

## Analysis steps

| Step | Script | Main output (`results/`) |
|---|---|---|
| Corrected LDL-C (treated participants: TC / 0.8 − HDL − TG / 5) | `scripts/make_corrected_ldl.py` | `derived/ldl_corrected.csv` |
| Candidate SNPs: GWAS (MAF ≥ 0.01, p ≤ 0.05), clumping (r² 0.1, 250 kb) | `scripts/make_gwas_pheno.py`, `scripts/run_gwas.sh` | `derived/gwas/` |
| Analysis file, rsID / gene map | `scripts/build_analysis_data.py`, `scripts/make_snpnexus_upload.py`, `scripts/make_rsid_map.py` | `derived/analysis_data.pkl` |
| Sample checks, covariates, PCs, family clusters | `python -m src.data` | log |
| GLGC labels: anchor (p < 5 × 10⁻⁸), candidate (5 × 10⁻⁸ ≤ p < 0.05), unsupported | `python -m src.glgc` | `glgc/annotated_all_snps.csv` |
| MAF and imputation quality | `scripts/snp_maf_info.py` | `glgc/snp_maf_info.csv` |
| Number of pre-selected SNPs (k) by nested CV | `scripts/run_nested_cv.py` | `nested_cv.csv`, `nested_cv_decision.json` |
| 20 splits (60/40): pre-selection, residualization and early stopping on training data only; genomics-only and joint XGBoost models; SHAP on test data | `scripts/run_splits.py` (`--version v1`, `--version v2`) | `splits/`, `v2/splits/`, `v2/aggregate/` |
| Single-metabolite mediation for every GLGC anchor (Table 1, suppressor counts in Table 2) | `scripts/anchor_mediation.py` | `v2/anchor_mediation/` |
| Mediation through a cross-fitted metabolome LDL-C score (proportion mediated) | `scripts/metabolome_score_mediation.py` | `v3/` |
| Losing / gaining labels from the retained share of importance | `scripts/classify_snps.py` | `v6/v2/{snp_table,losing,gaining}.csv` |
| Mediation for suggestive (GLGC candidate) SNPs | `scripts/candidate_mediation.py` | `v6/candidate_mediation/` |
| Interaction scan and sensitivity analyses for gaining SNPs (Table 2) | `scripts/gaining_interaction.py` | `v6/gaining_interaction/` |
| Table 2 extended to gaining SNPs without GLGC support | `scripts/gaining_table_all_groups.py` | `v6/gaining_all_groups/` |
| Interaction scan of every pre-selected SNP (global FDR) | `scripts/global_interaction_scan.py` | `v2/split_interaction/` |
| Figures 2–3, Supplementary Table S1 | `scripts/make_figures.py`, `scripts/make_supp_table_s1.py` | `figures/`, `tables/` |

Mediation (Baron–Kenny with Sobel test) and interaction models are ordinary least squares on all participants with
age, sex and PC1–PC10 in every model and log-transformed, standardized metabolites (`src/mediation.py`,
`src/interaction.py`, vectorized in `src/ols.py`). Multiple testing: Bonferroni within SNP (0.05 / 1,257) and
Benjamini–Hochberg FDR across the tests of an analysis.

Version labels in file and key names follow the analysis record: `v1` = pre-specified models and rank-based labels;
`v2` = models with cross-validated early stopping (the models reported); `v3` = metabolome-score mediation; `v6` = the
share-based labels reported in the manuscript.

## Tests

```bash
python tests/test_ols.py          # vectorized mediation / interaction scans equal statsmodels OLS
python tests/test_no_leakage.py   # planted-feature and index-disjointness tests (needs the data)
python scripts/smoke_test.py      # leakage tests + small end-to-end run (needs the data)
```

## License

MIT (see `LICENSE`).
