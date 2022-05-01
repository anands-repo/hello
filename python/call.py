# © 2019 University of Illinois Board of Trustees.  All rights reserved
import os
import argparse
import subprocess
import random
import warnings
import logging
import glob
import HotspotDetectorDVFiltered
import shardHotspots
import caller_calling
import multiprocessing
from PySamFastaWrapper import PySamFastaWrapper
from functools import partial
import tqdm
import libCallability
import pysam
import prepareVcf
import torch
import numpy as np
import random
import os
import re

libCallability.initLogging(False)
torch.set_num_threads(1)
random.seed(13)
np.random.seed(13)
# torch.set_num_threads isn't working in multiprocessing
os.environ["OMP_NUM_THREADS"] = "1"


def get_bam_string(bam):
    bam = os.path.abspath(bam)
    bamname0 = os.path.split(bam)[-1]
    bamname1 = os.path.split(os.path.split(bam)[0])[-1]
    bamname = bamname1.replace("/", "__") + "___" + bamname0.replace("/", "__")
    return bamname.replace(".", "__")


def get_workdir(ibam, pbam, chrom=None, string="features"):
    prefix_dir = string
    if chrom:
        prefix_dir = "%s_%s" % (prefix_dir, chrom)
    if ibam:
        prefix_dir += "_" + get_bam_string(ibam)
    if pbam:
        prefix_dir += "_" + get_bam_string(pbam)
    return prefix_dir


def get_chunks(length, nJobs):
    split_size = length // nJobs
    final_job = nJobs * split_size < length
    ranges = []
    for i in range(nJobs):
        start = i * split_size
        stop = min((i + 1) * split_size, length)
        ranges.append((start, stop))
    if final_job:
        ranges.append((nJobs * split_size, length))
    return ranges


def launcher(args_, functor, stage):
    try:
        return functor(args_)
    except Exception as e:
        logger.error("Failure when executing %s for args %s" % (stage, str(args_)))
        raise e


def get_reference_chromosomes(ref):
    with pysam.FastaFile(ref) as fhandle:
        references = set(fhandle.references)
        base_chromosomes = [str(x) for x in range(1, 23)] + ["X", "Y"]
        no_prefix_chromosomes = references.intersection(base_chromosomes)
        prefixed_chromosomes = references.intersection(["chr%s" % i for i in base_chromosomes])

    return max(no_prefix_chromosomes, prefixed_chromosomes, key=lambda x: len(x))


def sort_hotspots(hotspots):
    pattern = re.compile(r"job_chromosome(?:.*?)_job([0-9]+).txt")
    return sorted(hotspots, key=lambda x: int(pattern.match(os.path.split(x)[-1]).group(1)))


def main(args):
    # Convenience assignments
    ibam, pbam = args.ibam, args.pbam
    pacbio = pbam and not ibam

    features_dir = get_workdir(ibam, pbam)
    features_dir = os.path.join(args.workdir, features_dir)
    if not os.path.exists(features_dir):
        os.makedirs(features_dir)
    feature_file_number = 0
    caller_args = []
    ref = PySamFastaWrapper(args.ref)

    bam_arg = None
    if ibam and pbam:
        bam_arg = "%s,%s" % (ibam, pbam)
    elif ibam:
        bam_arg = ibam
    elif pbam:
        bam_arg = pbam
    else:
        raise ValueError("At least one of ibam or pbam must be provided")

    workers = multiprocessing.Pool(args.num_threads)

    for chrom in args.CHROMOSOMES:
        # Prepare output directory
        output_dir = os.path.join(
            args.workdir, get_workdir(ibam, pbam, chrom=chrom, string="hotspots"))

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Create hotspot command args
        ref.chrom = chrom
        hotspot_args = []

        for i, (start, stop) in enumerate(get_chunks(len(ref), 500)):
            output_name = os.path.join(
                output_dir, "job_chromosome%s_job%d.txt" % (chrom, i))
            hotspot_args.append(argparse.Namespace(
                bam=bam_arg,
                ref=args.ref,
                region="%s,%d,%d" % (chrom, start, stop),
                pacbio=pacbio,
                output=output_name,
                hybrid_hotspot=args.hybrid_hotspot,
                q_threshold=args.q_threshold,
                mapq_threshold=args.mapq_threshold,
            ))

        # Run hotspot detection in parallel
        logger.info("Launching hotspot detection for chromosome %s" % chrom)
        hotspot_results = []
        runner = partial(
            launcher,
            functor=HotspotDetectorDVFiltered.main,
            stage="Hotspot detection for chromosome %s" % chrom
        )
        for result in tqdm.tqdm(
            workers.imap_unordered(runner, hotspot_args),
            desc="Hotspots, chromosome %s" % chrom,
            total=len(hotspot_args)):
            hotspot_results.append(result)

        logger.info("Combining all hotspots and sharding")
        hotspot_results = sort_hotspots(hotspot_results)
        hotspot_name = os.path.join(output_dir, "hotspots.txt")
        with open(hotspot_name, "w") as fhandle:
            for r in hotspot_results:
                if os.path.exists(r):
                    subprocess.run(["cat", r], stdout=fhandle, check=True)

        # Shard hotspots
        shard_name = os.path.join(output_dir, "shard")
        shard_args = argparse.Namespace(
            hotspots=hotspot_name,
            minSeparation=25,
            maxShards=500,
            outputPrefix=shard_name,
        )
        shardHotspots.main(shard_args)
        logger.info("Completed sharding, creating caller commands for running NN")
        shards = glob.glob("%s*.txt" % shard_name)

        # Create caller commands
        for shard in shards:
            output_prefix = os.path.join(features_dir, "features%d" % feature_file_number)
            logfilename = os.path.join(features_dir, "features%d.log" % feature_file_number)
            args_ = argparse.Namespace(
                bam=bam_arg,
                activity=shard,
                ref=args.ref,
                network=args.network,
                outputPrefix=output_prefix,
                debug=False,
                provideFeatures=True,
                hotspotMode="BOTH",
                chrPrefixes=None,
                featureLength=150,
                truth=None,
                highconf=None,
                intersectRegions=False,
                simple=True,
                reuseSearchers=False,
                noAlleleLevelFilter=False,
                clr=False,
                hybrid_hotspot=args.hybrid_hotspot,
                pacbio=pacbio,
                test_labeling=False,
                only_contained=False,
                hybrid_eval=False,
                q_threshold=args.q_threshold,
                mapq_threshold=args.mapq_threshold,
                keep_hdf5=False,
                reconcilement_size=args.reconcilement_size,
                include_hp=args.include_hp,
                log=logfilename,
            )
            feature_file_number += 1
            caller_args.append(args_)

        logger.info("Created call commands for chromosome %s" % chrom)

    logger.info("Finished hotspot generation for all chromosomes. Launching all caller commands.")

    logfiles = []
    runner = partial(launcher, functor=caller_calling.main, stage="Caller")
    for feature_filename, feature_logs in tqdm.tqdm(
        workers.imap_unordered(runner, caller_args),
        desc="Caller progress",
        total=len(caller_args),
    ):
        logfiles.append(feature_logs)

    logger.info("Completed runs, checking log files")

    for logfilename in logfiles:
        with open(logfilename, 'r') as lhandle:
            if "Completed running the script" not in lhandle.read():
                logger.error("Log file %s doesn't have termination string" % logfilename)
                raise ValueError("Did not run")

    logger.info("Completed running all caller commands correctly, preparing vcf")

    prepare_vcf_args = argparse.Namespace(
        prefix=os.path.join(features_dir, "features"),
        ref=args.ref,
        tmpdir="/tmp/vcftemp",
        numThreads=args.num_threads,
        outputPrefix=os.path.join(args.workdir, "results"),
        checkRuns=False,
    )
    result_vcf_name = prepareVcf.main(prepare_vcf_args)
    logger.info("Completed runs. Results in %s" % result_vcf_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create data dump pipeline")

    parser.add_argument(
        "--ibam",
        help="Illumina BAM file (comma-separated if multiple files)",
        required=False,
    )

    parser.add_argument(
        "--pbam",
        help="Pacbio BAM file (comma-separated if multiple files)",
        required=False,
    )

    parser.add_argument(
        "--ref",
        help="Reference sequence",
        required=True
    )

    parser.add_argument(
        "--workdir",
        help="Working directory",
        required=True,
    )

    parser.add_argument(
        "--chromosomes",
        help="Chromosomes to use (comma-separated)",
        required=False,
    )

    parser.add_argument(
        "--network",
        help="Network path",
        required=True,
    )

    parser.add_argument(
        "--hybrid_hotspot",
        help="Enable hybrid hotspot mode",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "--q_threshold",
        default=10,
        type=int,
        help="Quality threshold",
    )

    parser.add_argument(
        "--mapq_threshold",
        default=10,
        type=int,
        help="Mapping quality threshold",
    )

    parser.add_argument(
        "--num_threads",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--reconcilement_size",
        help="Size of a hotspot region to enable reconcilement of pacbio/illumina representations",
        default=10,
        type=int,
    )

    parser.add_argument(
        "--include_hp",
        help="Include HP tags in tensors",
        default=False,
        action="store_true",
    )

    args = parser.parse_args()

    # Logging
    logger = logging.getLogger(name="call")
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s %(levelname)s:%(message)s")
    lhandler = logging.StreamHandler()
    lhandler.setFormatter(fmt)
    logger.setLevel(logging.INFO)
    logger.addHandler(lhandler)

    if args.chromosomes:
        args.CHROMOSOMES = args.chromosomes.split(",")
    else:
        args.CHROMOSOMES = get_reference_chromosomes(args.ref)

    logger.info("Will run variant calling for chromosomes %s" % str(args.CHROMOSOMES))

    main(args)
