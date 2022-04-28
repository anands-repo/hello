# HELLO: A DNN-based small variant caller

HELLO is a Deep Neural Network-based small variant caller that can call variants for Illumina, PacBio, and hybrid Illumina-PacBio settings. HELLO uses customized Deep Neural Networks which provide accurate variant calls with relatively small model size.

# Information regarding HELLO's methodology

The methodologies used in the current version of HELLO are described in our publication at BMC Bioinformatics: https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-021-04311-4

Older methodologies used in a prior version of HELLO, with only support for hybrid variant calling, is described at https://www.biorxiv.org/content/10.1101/2020.03.23.004473v1.

The models released in this package (including models for hybrid variant calling) do not follow methodologies described in the bioRxiv article, but follow the methodologies described in the paper published at BMC Bioinformatics, which are significantly different from the bioRxiv version.

# Information regarding HELLO's code and usage

(NEW) PacBio haplotagged model is available in models directory.

The repository contains files for HELLO - a small variant caller that is designed for running standalone and hybrid small variant calling.

The following Docker image may be used with the tool. These images may not be final, and we will update here when the images are updated.

`docker pull oddjobs/hello_deps`

To build the tool, please run inside docker container

```
git clone https://github.com/anands-repo/hello.git
cd hello
cmake .
make -j 12
```


To run Illumina variant calling, please use the following command

```
python python/call.py \
    --ibam $bam \
    --ref $REF \
    --network models/illumina_multi_coverage_mapq_threshold_hg002_continue_run16.wrapper.dnn \
    --num_threads $NUM_THREADS \
    --workdir $workdir --mapq_threshold 5
```


To run PacBio variant calling, please use the following command (`--include_hp` to be added for haplotagged PacBio reads).

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
