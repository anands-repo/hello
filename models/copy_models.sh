#!/bin/bash

set -e
set -x
set -o pipefail

SCRIPTPATH=$(dirname $(realpath -s $0))

### Copy HELLO Illumina models
# (From /root/storage/GIAB/GIAB_HG003_GRCh38/variant_calls/illumina_2020_10_30_1821/call.sh)
cp /root/storage/GIAB/train/HG002/illumina_data_dump_mapq_threshold_2020_10_05_1055/illumina_multi_coverage_mapq_threshold_hg002_continue_run16.wrapper.dnn $SCRIPTPATH/.

### Copy HELLO Hybrid
# (From /root/storage/GIAB/GIAB_HG003_GRCh38/variant_calls/hybrid_2020_10_19_1527/no_ensemble_bugfix_2020_12_20_0114/call30x15x.sh for example)
cp /root/storage/GIAB/GIAB_HG003_GRCh38/variant_calls/hybrid_2020_10_19_1527/no_ensemble_bugfix_2020_12_20_0114/no_ensemble_multi_coverage_mapq_threshold_hg002_continued17.wrapper.dnn $SCRIPTPATH/.

### Copy HELLO PacBio models
# (From /root/storage/GIAB/GIAB_HG003_GRCh38/variant_calls/pacbio_2020_10_30_1838/call.sh)
cp /root/storage/GIAB/train/HG002/pacbio_data_dump_mapq_threshold_2020_10_06_1953/pacbio_multi_coverage_mapq_threshold_hg00216.wrapper.dnn $SCRIPTPATH/.

### DeepVariant Illumina models
# (From /root/storage/GIAB/GIAB_HG003_GRCh38/variant_calls/dv_illumina_2020_11_05_1222/cmd.sh)
mkdir -p $SCRIPTPATH/dv_illumina
cp /root/storage/GIAB/train/deepvariant/HG002/illumina_2020_10_10_1146/evaluation_directory/model.ckpt-1524144* $SCRIPTPATH/dv_illumina/.

### DeepVariant PacBio models
# (best checkpoint file from /root/storage/GIAB/GIAB_HG003_GRCh38/variant_calls/dv_pacbio_2020_12_01_2107/cmd.sh)
# (From /root/storage/GIAB/train/deepvariant/HG002/pacbio_2020_11_29_1300/gcloud-training-results/training_results_2020_11_30_1308/evaluation_directory/best_checkpoint.txt)
mkdir -p $SCRIPTPATH/dv_pacbio
MODEL=$(cat /root/storage/GIAB/train/deepvariant/HG002/pacbio_2020_11_29_1300/gcloud-training-results/training_results_2020_11_30_1308/evaluation_directory/best_checkpoint.txt)
cp ${MODEL}* $SCRIPTPATH/dv_pacbio/.

### DeepVariant hybrid models
# (From /root/storage/GIAB/GIAB_HG003_GRCh38/variant_calls/dv_hybrid_2020_2020_11_17_1127/run_ax_by.sh)
# (Same as /root/storage/GIAB/train/deepvariant/HG002/hybrid_2020_11_02_1210/gcloud_training_2020_11_13_2221/evaluations/best_checkpoint.txt)
mkdir -p $SCRIPTPATH/dv_hybrid
cp /root/storage/GIAB/train/deepvariant/HG002/hybrid_2020_11_02_1210/gcloud_training_2020_11_13_2221/evaluations/model.ckpt-333900* ./dv_hybrid/.

### Compress deepvariant files
compress() {
    src=$1
    target=${src}.tar.gz
    tar cvfz $target $src
    rm -r $src
}

compress $SCRIPTPATH/dv_illumina
compress $SCRIPTPATH/dv_pacbio
compress $SCRIPTPATH/dv_hybrid

echo "DONE"
