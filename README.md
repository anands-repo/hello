# HELLO: A DNN-based small variant caller

(NEW) PacBio haplotagged model is available in models directory. The source code for running the model is in hello_dev.tar.gz, pending merge.

The repository contains files for HELLO - a small variant caller that is designed for running standalone and hybrid small variant calling.

The following Docker image may be used with the tool. These images may not be final, and we will update here when the images are updated.

`docker pull oddjobs/hello_image.x86_64`

To build the tool, please run

```
cmake .
make -j 12
```

NOTE: To properly download models, git-lfs needs to be installed. Once installed, please do `git lfs pull` inside the git repo.

To run Illumina variant calling, please use the following command

```
python python/call.py \
    --ibam $bam \
    --ref $REF \
    --network models/illumina_multi_coverage_mapq_threshold_hg002_continue_run16.wrapper.dnn \
    --num_threads $NUM_THREADS \
    --workdir $workdir --mapq_threshold 5
```


To run PacBio variant calling, please use the following command

```
python python/call.py \
    --pbam $bam \
    --ref $REF \
    --network models/pacbio_multi_coverage_mapq_threshold_hg00216.wrapper.dnn \
    --num_threads $NUM_THREADS \
    --workdir $workdir --mapq_threshold 5
```

To run hybrid Illumina-PacBio variant calling, please use the following command

```
python python/call.py \
    --ibam $ibam \
    --pbam $pbam \
    --ref $ref \
    --network models/no_ensemble_multi_coverage_mapq_threshold_hg002_continued17.wrapper.dnn \
    --num_threads $NUM_THREADS \
    --workdir $workdir \
    --mapq_threshold 5 \
    --reconcilement_size 0
```

The output VCF file may be found in `$workdir/results.mean.vcf`.

A preliminary version of our draft is currently available at https://www.biorxiv.org/content/10.1101/2020.03.23.004473v1, however we plan to update this shortly with substantial new information.

Errata: Currently, the VCF output by the tool has a syntax error with the INFO field. This may be fixed with the following code
```
fix_vcf() {
    correct_string="##INFO=<ID=MixtureOfExpertPrediction,Type=String,Number=1,Description=\"Mean predictions from experts\">"
    cat $1 | sed "s?##INFO=<ID=MixtureOfExpertPrediction,Description=\"Mean predictions from experts\"?$correct_string?g" > $2
}

fix_vcf $workdir/results.mean.vcf $workdir/results.mean.corrected.vcf
```

