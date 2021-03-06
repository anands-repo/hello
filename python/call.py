# © 2019 University of Illinois Board of Trustees.  All rights reserved
import os
import argparse
import subprocess
import random
import warnings
import logging
import glob
from find_chr_prefixes import get_reference_prefixes

CHROMOSOMES = [str(i) for i in range(1, 23)]


def parallel_execute(cmd, nt):
    subprocess.call(
        "cat %s | shuf | parallel --eta -j %d" % (cmd, nt), shell=True, executable="/bin/bash"
    )


def get_bam_string(bam):
    bam = os.path.abspath(bam)
    bamname0 = os.path.split(bam)[-1]
    bamname1 = os.path.split(os.path.split(bam)[0])[-1]
    bamname = bamname1.replace("/", "__") + "___" + bamname0.replace("/", "__")
    return bamname.replace(".", "__")


def main(ibam, pbam):
    if (not ibam) or (not pbam):
        raise NotImplementedError("Only hybrid mode has been implemented")

    logging.info("Creating hotspot detection jobs")

    caller_commands = []
    feature_prefixes = []

    ib = ibam
    pbam_selected = pbam
    ib_string = get_bam_string(ib)  # ib.replace(".", "__").replace("/", "___")
    pb_string = get_bam_string(pbam_selected)  # pbam_selected.replace(".", "__").replace("/", "___")

    features_dir = "features_%s__%s" % (
        ib_string,
        pb_string
    )

    features_dir = os.path.join(args.workdir, features_dir)

    if not os.path.exists(features_dir):
        os.makedirs(features_dir)

    caller_command_filename = os.path.join(features_dir, "caller_commands.sh")
    chandle = open(caller_command_filename, "w")
    logfiles = []

    feature_file_number = 0

    for chrom in CHROMOSOMES:
        output_dir = os.path.join(
            args.workdir,
            "hotspots_%s_%s_%s" % (ib_string, chrom, pb_string)
        )
        output_dir.replace(".", "__")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Create hotspot jobs
        create_cmd = [
            "python",
            create_hotspot_jobs_scripts,
            "--nJobs", "%d" % 500,
            "--chromosome", "%s" % chrom,
            "--bam", "%s" % ib,
            "--bam2", "%s" % pbam_selected,
            "--ref", "%s" % args.ref,
            "--log",
            "--outputDir", "%s" % output_dir,
            "--q_threshold", "%d" % args.q_threshold,
            "--mapq_threshold", "%d" % args.mapq_threshold,
        ]

        if args.hybrid_hotspot:
            create_cmd += ["--hybrid_hotspot"]

        subprocess.call(create_cmd)

        command = os.path.join(output_dir, "jobs_chromosome%s.sh" % chrom)
        results = [os.path.join(output_dir, "jobs_chromosome%s_job%d.txt" % (chrom, i)) for i in range(501)]

        logging.info("Created jobs to create hotspots, running to generate hotspots")
        # subprocess.call(
        #     "cat %s | shuf | parallel --eta -j %d" % (command, args.num_threads), shell=True, executable="/bin/bash"
        # )
        parallel_execute(command, args.num_threads)

        logging.info("Combining all hotspots and sharding")
        hotspot_name = os.path.join(output_dir, "hotspots.txt")
        fhandle = open(hotspot_name, "w")
        for r in results:
            if os.path.exists(r):
                subprocess.call(["cat", r], stdout=fhandle)
                pass
        fhandle.close()

        # Shard hotspots
        shard_name = os.path.join(output_dir, "shard")
        shard_command = [
            "python",
            shard_script,
            "--hotspots", hotspot_name,
            "--outputPrefix", shard_name
        ]
        subprocess.call(shard_command)

        logging.info("Completed sharding, creating caller commands for dumping training data")

        # python /root/storage/subsampled/Illumina/30x/training/hello/python/caller.py --activity /root/storage/subsampled/Illumina/30x/training/shards0/shard0.txt
        # --bam /root/storage/subsampled/Illumina/30x/HG001.hs37d5.30x.RG.realigned.bam,/root/storage/subsampled/PacBio/30x/HG001.SequelII.pbmm2.hs37d5.whatshap.haplotag.RTG.trio.bam
        # --ref /root/storage/data/hs37d5.fa --truth /root/storage/subsampled/3.3.2/HG001_GRCh37_GIAB_highconf_CG-IllFB-IllGATKHC-Ion-10X-SOLID_CHROM1-X_v.3.3.2_highconf_PGandRTGphasetransfer.vcf
        # --highconf /root/storage/subsampled/3.3.2/HG001_GRCh37_GIAB_highconf_CG-IllFB-IllGATKHC-Ion-10X-SOLID_CHROM1-X_v.3.3.2_highconf_nosomaticdel.bed --featureLength 250 --intersect --reuse --simple
        # --outputPrefix /root/storage/subsampled/Illumina/30x/training/shards0/shard0.txt_data >& /root/storage/subsampled/Illumina/30x/training/shards0/shard0.txt_log.log
        shards = glob.glob("%s*.txt" % shard_name)

        for shard in shards:
            output_prefix = os.path.join(features_dir, "features%d" % feature_file_number)
            logfilename = os.path.join(features_dir, "features%d.log" % feature_file_number)
            command_string = "python %s" % caller_command
            command_string += " --activity %s" % shard
            command_string += " --bam %s,%s" % (ib, pbam_selected)
            command_string += " --ref %s" % args.ref
            command_string += " --network %s" % args.network
            command_string += " --featureLength %d" % 150
            command_string += " --simple"
            command_string += " --provideFeatures"
            command_string += " --outputPrefix %s" % output_prefix
            command_string += " --hybrid_hotspot" if args.hybrid_hotspot else ""
            command_string += " --q_threshold %d" % args.q_threshold
            command_string += " --mapq_threshold %d" % args.mapq_threshold
            command_string += " --reconcilement_size %d" % args.reconcilement_size
            chandle.write(
                command_string + " >& " + logfilename + "\n"
            )
            feature_file_number += 1
            logfiles.append(logfilename)

        logging.info("Created call command for chromosome %s" % chrom)

    chandle.close()

    logging.info("Launching all caller commands")

    # subprocess.call(
    #     "cat %s | parallel -j %d --eta" % (caller_command_filename, args.num_threads), shell=True, executable="/bin/bash"
    # )
    parallel_execute(caller_command_filename, args.num_threads)

    logging.info("Completed runs, checking log files")

    for logfilename in logfiles:
        with open(logfilename, 'r') as lhandle:
            if "Completed running the script" not in lhandle.read():
                logging.error("File %s doesn't have termination string" % logfilename)
                return

    logging.info("All commands completed correctly, preparing vcf")

    prepare_vcf_command = [
        "python",
        prepare_vcf_script,
        "--prefix", os.path.join(features_dir, "features"),
        "--ref", args.ref,
        "--outputPrefix", os.path.join(args.workdir, "results"),
    ]

    subprocess.call(prepare_vcf_command)


def main_single(bam, pacbio):
    logging.info("Creating hotspot detection jobs")

    caller_commands = []
    feature_prefixes = []

    ib = bam
    pbam_selected = None
    ib_string = get_bam_string(ib)  # ib.replace(".", "__").replace("/", "___")
    pb_string = ""

    features_dir = "features_%s__%s" % (
        ib_string,
        pb_string
    )

    features_dir = os.path.join(args.workdir, features_dir)

    if not os.path.exists(features_dir):
        os.makedirs(features_dir)

    caller_command_filename = os.path.join(features_dir, "caller_commands.sh")
    chandle = open(caller_command_filename, "w")
    logfiles = []

    feature_file_number = 0

    for chrom in CHROMOSOMES:
        output_dir = os.path.join(
            args.workdir,
            "hotspots_%s_%s_%s" % (ib_string, chrom, pb_string)
        )
        output_dir.replace(".", "__")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Create hotspot jobs
        create_cmd = [
            "python",
            create_hotspot_jobs_scripts,
            "--nJobs", "%d" % 500,
            "--chromosome", "%s" % chrom,
            "--bam", "%s" % ib,
            "--ref", "%s" % args.ref,
            "--log",
            "--outputDir", "%s" % output_dir,
            "--q_threshold", "%d" % args.q_threshold,
            "--mapq_threshold", "%d" % args.mapq_threshold,
        ]
        if pacbio:
            create_cmd.append("--pacbio")
        subprocess.call(create_cmd)

        command = os.path.join(output_dir, "jobs_chromosome%s.sh" % chrom)
        results = [os.path.join(output_dir, "jobs_chromosome%s_job%d.txt" % (chrom, i)) for i in range(501)]

        logging.info("Created jobs to create hotspots, running to generate hotspots")
        # subprocess.call(
        #     "cat %s | shuf | parallel --eta -j %d" % (command, args.num_threads), shell=True, executable="/bin/bash"
        # )
        parallel_execute(command, args.num_threads)

        logging.info("Combining all hotspots and sharding")
        hotspot_name = os.path.join(output_dir, "hotspots.txt")
        fhandle = open(hotspot_name, "w")
        for r in results:
            if os.path.exists(r):
                subprocess.call(["cat", r], stdout=fhandle)
                pass
        fhandle.close()

        # Shard hotspots
        shard_name = os.path.join(output_dir, "shard")
        shard_command = [
            "python",
            shard_script,
            "--hotspots", hotspot_name,
            "--outputPrefix", shard_name
        ]
        subprocess.call(shard_command)

        logging.info("Completed sharding, creating caller commands for dumping training data")

        # python /root/storage/subsampled/Illumina/30x/training/hello/python/caller.py --activity /root/storage/subsampled/Illumina/30x/training/shards0/shard0.txt
        # --bam /root/storage/subsampled/Illumina/30x/HG001.hs37d5.30x.RG.realigned.bam,/root/storage/subsampled/PacBio/30x/HG001.SequelII.pbmm2.hs37d5.whatshap.haplotag.RTG.trio.bam
        # --ref /root/storage/data/hs37d5.fa --truth /root/storage/subsampled/3.3.2/HG001_GRCh37_GIAB_highconf_CG-IllFB-IllGATKHC-Ion-10X-SOLID_CHROM1-X_v.3.3.2_highconf_PGandRTGphasetransfer.vcf
        # --highconf /root/storage/subsampled/3.3.2/HG001_GRCh37_GIAB_highconf_CG-IllFB-IllGATKHC-Ion-10X-SOLID_CHROM1-X_v.3.3.2_highconf_nosomaticdel.bed --featureLength 250 --intersect --reuse --simple
        # --outputPrefix /root/storage/subsampled/Illumina/30x/training/shards0/shard0.txt_data >& /root/storage/subsampled/Illumina/30x/training/shards0/shard0.txt_log.log
        shards = glob.glob("%s*.txt" % shard_name)

        for shard in shards:
            output_prefix = os.path.join(features_dir, "features%d" % feature_file_number)
            logfilename = os.path.join(features_dir, "features%d.log" % feature_file_number)
            command_string = "python %s" % caller_command
            command_string += " --activity %s" % shard
            command_string += " --bam %s" % ib
            command_string += " --ref %s" % args.ref
            command_string += " --network %s" % args.network
            command_string += " --featureLength %d" % 150
            command_string += " --simple"
            command_string += " --provideFeatures"
            command_string += " --outputPrefix %s" % output_prefix
            command_string += " --pacbio" if pacbio else ""
            command_string += " --q_threshold %d" % args.q_threshold
            command_string += " --mapq_threshold %d" % args.mapq_threshold
            chandle.write(
                command_string + " >& " + logfilename + "\n"
            )
            feature_file_number += 1
            logfiles.append(logfilename)

        logging.info("Created call command for chromosome %s" % chrom)

    chandle.close()

    logging.info("Launching all caller commands")

    # subprocess.call(
    #     "cat %s | parallel -j %d --eta" % (caller_command_filename, args.num_threads), shell=True, executable="/bin/bash"
    # )
    parallel_execute(caller_command_filename, args.num_threads)

    logging.info("Completed runs, checking log files")

    for logfilename in logfiles:
        with open(logfilename, 'r') as lhandle:
            if "Completed running the script" not in lhandle.read():
                logging.error("File %s doesn't have termination string" % logfilename)
                return

    logging.info("All commands completed correctly, preparing vcf")

    prepare_vcf_command = [
        "python",
        prepare_vcf_script,
        "--prefix", os.path.join(features_dir, "features"),
        "--ref", args.ref,
        "--outputPrefix", os.path.join(args.workdir, "results"),
    ]

    subprocess.call(prepare_vcf_command)


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

    args = parser.parse_args()

    if args.chromosomes:
        CHROMOSOMES = args.chromosomes.split(",")
    else:
        ref_prefix = get_reference_prefixes(args.ref)
        if ref_prefix:
            CHROMOSOMES = [ref_prefix + i for i in CHROMOSOMES]

    logging.basicConfig(level=logging.INFO)

    path = os.path.split(os.path.abspath(__file__))[0]

    ibams = []
    pbams = []

    if args.ibam:
        ibams = args.ibam.split(",")

    if args.pbam:
        pbams = args.pbam.split(",")

    assert(len(ibams) > 0 or len(pbams) > 0), "Provide at least one BAM file"

    create_hotspot_jobs_scripts = os.path.join(
        path, "createJobsDVFiltered.py"
    )

    shard_script = os.path.join(
        path, "shardHotspots.py"
    )

    caller_command = os.path.join(
        path, "caller_calling.py"
    )

    prepare_vcf_script = os.path.join(
        path, "prepareVcf.py"
    )

    # Create hotspot jobs
    if len(ibams) > 0 and len(pbams) > 0:
        main(ibams[0], pbams[0])
    else:
        if len(ibams) > 0:
            main_single(ibams[0], pacbio=False)
        else:
            main_single(pbams[0], pacbio=True)
