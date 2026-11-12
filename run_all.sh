#!/bin/bash
# Full SHAPTrack analysis in the order used for the manuscript. Run from the repository root:
#   bash run_all.sh <outcome|gwas|data|cv|splits|downstream|figures|all>
# Configuration: config.yaml, or the file named by $SHAPTRACK_CONFIG. Python: $PYTHON (default: python).
# Long stages (gwas, cv, splits) take hours to days on a multi-core node; logs go to <results_dir>/logs/.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}
export SHAPTRACK_CONFIG=${SHAPTRACK_CONFIG:-config.yaml}
RES=$($PY -c "import yaml,os; print(yaml.safe_load(open(os.environ['SHAPTRACK_CONFIG']))['paths']['results_dir'])")
NSEED=$($PY -c "import yaml,os; s=yaml.safe_load(open(os.environ['SHAPTRACK_CONFIG']))['splits']; print(s['seed_base'], s['seed_base'] + s['n_seeds'] - 1)")
L=$RES/logs; mkdir -p "$L"
ts=$(date +%Y%m%d_%H%M%S)
run() { local name=$1; shift; echo "$(date '+%F %T') start $name"
    "$@" > "$L/${name}_$ts.log" 2>&1 || { echo "FAILED $name (see $L/${name}_$ts.log)"; exit 1; }
    echo "$(date '+%F %T') done  $name"; }

outcome() {     # corrected medication-adjusted LDL-C (paths.outcome_file)
    run corrected_ldl $PY scripts/make_corrected_ldl.py
}
gwas() {        # candidate SNPs: GWAS of corrected LDL-C, clumping, dosage export of the index SNPs
    run gwas_pheno $PY scripts/make_gwas_pheno.py
    seq 22 -1 1 | xargs -P "${GWAS_PARALLEL:-6}" -I{} bash scripts/run_gwas.sh {} "${GWAS_THREADS:-12}" "${GWAS_MEM_MB:-70000}"
}
data() {        # analysis file, rsID / gene map, sample checks, GLGC labels, MAF / Rsq
    run build_data $PY scripts/build_analysis_data.py
    run rsid_map $PY scripts/make_rsid_map.py   # needs paths.snpnexus_previous (SNPnexus result, see README)
    run step0 $PY -m src.data
    run glgc $PY -m src.glgc
    run snp_maf_info $PY scripts/snp_maf_info.py
}
cv() {          # leakage tests + small end-to-end run, then nested CV for k
    run smoke_test $PY -u scripts/smoke_test.py
    run nested_cv $PY scripts/run_nested_cv.py
}
splits() {      # 20 train/test splits: genomics-only and joint models, SHAP; pre-specified and main models
    set -- $NSEED
    run splits_prespecified $PY scripts/run_splits.py --models prespecified --seeds $(seq "$1" "$2")
    run splits_main $PY scripts/run_splits.py --models main --seeds $(seq "$1" "$2")
}
downstream() {  # labels and OLS follow-up analyses reported in the manuscript
    run anchor_mediation $PY scripts/anchor_mediation.py
    run metabolome_score $PY scripts/metabolome_score_mediation.py
    run classify_snps $PY scripts/classify_snps.py
    run candidate_mediation $PY scripts/candidate_mediation.py
    run gaining_interaction $PY scripts/gaining_interaction.py
    run table2_extended $PY scripts/gaining_table_all_groups.py
    run global_interaction_scan $PY scripts/global_interaction_scan.py --jobs "${JOBS:-48}"
}
figures() {     # Figures 2-3 and Supplementary Table S1
    mkdir -p figures tables
    run figures $PY scripts/make_figures.py "$RES" figures "$RES/main/anchor_mediation/full.csv"
    run supp_table_s1 $PY scripts/make_supp_table_s1.py "$RES" tables/Supplementary_Table_S1.csv
}

case "${1:-}" in
    outcome|gwas|data|cv|splits|downstream|figures) "$1" ;;
    all) outcome; gwas; data; cv; splits; downstream; figures ;;
    *) echo "usage: bash run_all.sh <outcome|gwas|data|cv|splits|downstream|figures|all>"; exit 2 ;;
esac
