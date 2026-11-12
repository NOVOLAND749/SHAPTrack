#!/bin/bash
# SNP-list GWAS with the corrected LDL-C, per chromosome: bgen -> pgen (MAF >= 0.01), --glm (same covariates as the
# original GWAS_LDL_adjusted), p <= 0.05, --clump (p1 0.05, r2 0.1, 250 kb, LD from all samples in the bgen), export of the
# index SNPs' dosages (REF counted, --export A) for the analysis sample. Temporary pgen removed after each chromosome.
# Usage (from the repository root): bash scripts/run_gwas.sh <chr> [threads] [memory_MB]
#   all chromosomes, 6 at a time: setsid nohup bash -c 'seq 22 -1 1 | xargs -P 6 -I{} bash scripts/run_gwas.sh {} 12 70000' > derived/gwas/logs/driver.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
CHR=$1; THREADS=${2:-12}; MEM=${3:-60000}
CFG=${SHAPTRACK_CONFIG:-config.yaml}
cfgval() { ${PYTHON:-python} -c "import yaml,sys; g=yaml.safe_load(open('$CFG'))['gwas']; print(g.get('$1', ''))"; }
PLINK=$(cfgval plink2); BGEN=$(cfgval bgen); BGEN=${BGEN/\{chr\}/$CHR}; SAMPLE=$(cfgval sample); COVAR=$(cfgval covar)
MAF=$(cfgval maf); PKEEP=$(cfgval p_keep); P1=$(cfgval clump_p1); R2=$(cfgval clump_r2); KB=$(cfgval clump_kb)
OUT=$(cfgval out_dir); PHENO=$(cfgval pheno_file); PHENO=${PHENO:-pheno_ldl_corrected.txt}
mkdir -p "$OUT"/{tmp,glm,clump,export,logs}
say() { echo "$(date '+%F %T') chr$CHR $*"; }
P="$PLINK --threads $THREADS --memory $MEM"
T="$OUT/tmp/chr$CHR"

say "make-pgen"
$P --bgen "$BGEN" ref-first --sample "$SAMPLE" --maf "$MAF" --make-pgen --out "$T" > "$OUT/logs/chr${CHR}_pgen.out" 2>&1
say "glm"
$P --pfile "$T" --pheno "$OUT/$PHENO" --covar "$COVAR" --glm hide-covar --covar-variance-standardize \
   --out "$OUT/glm/chr$CHR" > "$OUT/logs/chr${CHR}_glm.out" 2>&1
G=$(ls "$OUT"/glm/chr$CHR.*.glm.linear)
awk -v p="$PKEEP" 'NR==1{for(i=1;i<=NF;i++) if($i=="P") c=i; print; next} ($c!="NA" && $c<=p)' "$G" > "$OUT/glm/chr${CHR}_sig.txt"
say "clump ($(($(wc -l < "$OUT/glm/chr${CHR}_sig.txt") - 1)) SNPs with p <= $PKEEP)"
$P --pfile "$T" --clump "$OUT/glm/chr${CHR}_sig.txt" --clump-p1 "$P1" --clump-r2 "$R2" --clump-kb "$KB" \
   --out "$OUT/clump/chr$CHR" > "$OUT/logs/chr${CHR}_clump.out" 2>&1
awk 'NR==1{for(i=1;i<=NF;i++) if($i=="ID") c=i; next} {print $c}' "$OUT/clump/chr$CHR.clumps" > "$OUT/clump/chr${CHR}_index.txt"
say "export ($(wc -l < "$OUT/clump/chr${CHR}_index.txt") index SNPs)"
$P --pfile "$T" --extract "$OUT/clump/chr${CHR}_index.txt" --keep "$OUT/keep_analysis.txt" --export A \
   --out "$OUT/export/chr$CHR" > "$OUT/logs/chr${CHR}_export.out" 2>&1
rm -f "$T".pgen "$T".pvar "$T".psam
say "done"
