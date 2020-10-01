import os
import argparse
import subprocess
import random
import warnings
import logging
import glob
import pybedtools
import PySamFastaWrapper
from find_chr_prefixes import get_reference_prefixes
from call import parallel_execute
from call import get_bam_string

CHROMOSOMES = [str(i) for i in range(1, 21)]



def intersect(bed, truth, chromosome, prefix):
    """
    Perform bed intersection at chromosome level

    :param bed: str
        Bed file path

    :param truth: str
        Truth vcf path

    :param chromosome: str
        Chromosome

    :param prefix: str
        Prefix of the output file

    :return: tuple
        (chromosomal bed, chromosomal vcf)
    """
    logging.info("Preparing chromosomal bed and vcf files for chromosome %s" % chromosome)
    reference = PySamFastaWrapper.PySamFastaWrapper(args.ref, chrom=chromosome)
    bed_name = prefix + ".bed"
    vcf_name = prefix + ".vcf"

    vcf_bed = pybedtools.BedTool(truth)
    hc_bed = pybedtools.BedTool(bed)

    subset_op = pybedtools.BedTool("\t".join([chromosome, "0", "%d" % len(reference)]), from_string=True)
    vcf_bed.intersect(subset_op, header=True).saveas(vcf_name)
    hc_bed.intersect(subset_op).saveas(bed_name)
    logging.info("Completed preparing necessary ground-truth data")

    return (bed_name, vcf_name)


def main_single(bams, pacbio):
    train_files = []
    caller_commands = []

    for bam in bams:
        for chrom in CHROMOSOMES:
            bam_string = get_bam_string(bam)  # bam.replace(".", "__").replace("/", "___")

            output_dir = os.path.join(
                args.workdir,
                "hotspots_%s_%s" % (bam_string, chrom)
            )

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Create hotspot jobs
            create_cmd = [
                "python",
                create_hotspot_jobs_scripts,
                "--nJobs", "%d" % 500,
                "--chromosome", "%s" % chrom,
                "--bam", "%s" % bam,
                "--ref", "%s" % args.ref,
                "--log",
                "--outputDir", "%s" % output_dir,
                "--q_threshold", "%d" % args.q_threshold,
                "--mapq_threshold", "%d" % args.mapq_threshold,
            ]

            create_cmd += ["--pacbio"] if pacbio else []
            subprocess.call(create_cmd)

            command = os.path.join(output_dir, "jobs_chromosome%s.sh" % chrom)
            results = [os.path.join(output_dir, "jobs_chromosome%s_job%d.txt" % (chrom, i)) for i in range(501)]

            logging.info("Created jobs to create hotspots, running to generate hotspots")
            # subprocess.call(
            #     "cat %s | shuf | parallel --eta -j 30" % command, shell=True, executable="/bin/bash"
            # )
            parallel_execute(command, args.num_threads)

            logging.info("Combining all hotspots and sharding")
            hotspot_name = os.path.join(output_dir, "hotspots.txt")
            fhandle = open(hotspot_name, "w")
            for r in results:
                if os.path.exists(r):
                    subprocess.call(["cat", r], stdout=fhandle)
            fhandle.close()

            # Shard hotspots
            shard_name = os.path.join(output_dir, "shard")
            shard_command = [
                "python",
                shard_script,
                "--hotspots", hotspot_name,
                "--outputPrefix", shard_name,
            ]
            subprocess.call(shard_command)

            logging.info("Completed sharding, creating caller commands for dumping training data")

            shards = glob.glob("%s*.txt" % shard_name)
            caller_command_filename = os.path.join(output_dir, "caller_commands.sh")

            chrom_bed, chrom_vcf = intersect(
                args.bed,
                args.truth,
                chrom,
                prefix=os.path.join(output_dir, "chromosome%s_ground_truth" % chrom)
            )

            with open(caller_command_filename, "w") as fhandle:
                for shard in shards:
                    command_string = "python %s" % caller_command
                    command_string += " --activity %s" % shard
                    command_string += " --bam %s" % bam
                    command_string += " --ref %s" % args.ref
                    command_string += " --truth %s" % chrom_vcf
                    command_string += " --highconf %s" % chrom_bed
                    command_string += " --featureLength %d" % 150
                    command_string += " --intersect"
                    command_string += " --simple"
                    command_string += " --outputPrefix %s" % os.path.join(output_dir, "%s_data" % shard)
                    command_string += " --pacbio" if pacbio else ""
                    command_string += " --test_labeling" if args.test_labeling else ""
                    command_string += " --q_threshold %d" % args.q_threshold
                    command_string += " --mapq_threshold %d" % args.mapq_threshold
                    command_string += " --keep_hdf5" if args.keep_hdf5 else ""
                    fhandle.write(command_string + " >& " + os.path.join(output_dir, "%s_log" % shard) + "\n")

            logging.info("Created data dump commands")

            if not args.norun_caller:
                logging.info("Launching data dump")
                # subprocess.call(
                #     "cat %s | parallel -j 30 --eta" % caller_command_filename, shell=True, executable="/bin/bash"
                # )
                parallel_execute(caller_command_filename, args.num_threads)

                logging.info("Completed data dump")

                # Collect list of results
                train_files.extend(glob.glob(os.path.join(output_dir, "*.index")))
            else:
                logging.info("Saving data dump command")
                caller_commands.append(caller_command_filename)

    if not args.norun_caller:
        if not args.no_data_lst:
            training_files = os.path.join(args.workdir, "data.lst")

            with open(training_files, 'w') as dhandle:
                for line in train_files:
                    dhandle.write(line + "\n")

            logging.info("Training data files in %s" % training_files)
    else:
        logging.info("Use the following scripts to dump training data")

        for cmd_file in caller_commands:
            logging.info("Run file: %s" % cmd_file)


def main(ibams, pbams, random_combine=False):
    if not (len(ibams) > 0 and len(pbams) > 0):
        raise NotImplementedError

    if not random_combine:
        if len(ibams) > 1 or len(pbams) > 1:
            wanrings.warn("Non-random combine mode not suited for multiple IBAMs and PAMs")

    train_files = []
    caller_commands = []

    for ib in ibams:
        for chrom in CHROMOSOMES:
            logging.info("Creating hotspot detection jobs for chromosome %s" % chrom)
            pbam_selected = random.sample(pbams, 1)[0] if random_combine else pbams[0]
            ib_string = get_bam_string(ib)  # ib.replace(".", "__").replace("/", "___")
            pb_string = get_bam_string(pbam_selected)  # pbam_selected.replace(".", "__").replace("/", "___")

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
            #     "cat %s | shuf | parallel --eta -j 30" % command, shell=True, executable="/bin/bash"
            # )
            parallel_execute(command, args.num_threads)

            logging.info("Combining all hotspots and sharding")
            hotspot_name = os.path.join(output_dir, "hotspots.txt")
            fhandle = open(hotspot_name, "w")
            for r in results:
                if os.path.exists(r):
                    subprocess.call(["cat", r], stdout=fhandle)
            fhandle.close()

            # Shard hotspots
            shard_name = os.path.join(output_dir, "shard")
            shard_command = [
                "python",
                shard_script,
                "--hotspots", hotspot_name,
                "--outputPrefix", shard_name,
            ]
            subprocess.call(shard_command)

            logging.info("Completed sharding, creating caller commands for dumping training data")

            shards = glob.glob("%s*.txt" % shard_name)
            caller_command_filename = os.path.join(output_dir, "caller_commands.sh")

            chrom_bed, chrom_vcf = intersect(
                args.bed,
                args.truth,
                chrom,
                prefix=os.path.join(output_dir, "chromosome%s_ground_truth" % chrom)
            )

            with open(caller_command_filename, "w") as fhandle:
                for shard in shards:
                    command_string = "python %s" % caller_command
                    command_string += " --activity %s" % shard
                    command_string += " --bam %s,%s" % (ib, pbam_selected)
                    command_string += " --ref %s" % args.ref
                    command_string += " --truth %s" % chrom_vcf
                    command_string += " --highconf %s" % chrom_bed
                    command_string += " --featureLength %d" % 150
                    command_string += " --intersect"
                    command_string += " --simple"
                    command_string += " --outputPrefix %s" % os.path.join(output_dir, "%s_data" % shard)
                    command_string += " --test_labeling" if args.test_labeling else ""
                    command_string += " --hybrid_hotspot" if args.hybrid_hotspot else ""
                    command_string += " --hybrid_eval" if args.hybrid_eval else ""
                    command_string += " --q_threshold %d" % args.q_threshold
                    command_string += " --mapq_threshold %d" % args.mapq_threshold
                    command_string += " --reconcilement_size %d" % args.reconcilement_size
                    command_string += " --keep_hdf5" if args.keep_hdf5 else ""
                    fhandle.write(command_string + " >& " + os.path.join(output_dir, "%s_log" % shard) + "\n")

            logging.info("Created data dump commands")

            if not args.norun_caller:
                logging.info("Launching data dump")
                # subprocess.call(
                #     "cat %s | parallel -j 30 --eta" % caller_command_filename, shell=True, executable="/bin/bash"
                # )
                parallel_execute(caller_command_filename, args.num_threads)

                logging.info("Completed data dump")

                # Collect list of results
                train_files.extend(glob.glob(os.path.join(output_dir, "*.index")))
            else:
                logging.info("Saving data dump command")
                caller_commands.append(caller_command_filename)

    if not args.norun_caller:
        if not args.no_data_lst:
            training_files = os.path.join(args.workdir, "data.lst")

            with open(training_files, "w") as dhandle:
                for line in train_files:
                    dhandle.write(line + "\n")

            logging.info("Training data files in %s" % training_files)
    else:
        logging.info("Use the following scripts to dump training data")

        for cmd_file in caller_commands:
            logging.info("Run file: %s" % cmd_file)


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
        "--truth",
        help="Truth VCF",
        required=True
    )

    parser.add_argument(
        "--bed",
        help="High confidence bed file",
        required=True
    )

    parser.add_argument(
        "--workdir",
        help="Working directory",
        required=True,
    )

    parser.add_argument(
        "--random_combine",
        help="Randomly combine different pacbio files with each Illumina file",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "--norun_caller",
        help="Do not run caller, just provide run commands (for launching on different machines)",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "--chromosomes",
        help="Chromosomes to use (comma-separated)",
        required=False,
    )

    parser.add_argument(
        "--test_labeling",
        help="Use runs for testing labeling",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "--hybrid_hotspot",
        help="Use hybrid hotspot mode",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "--hybrid_eval",
        help="Evaluate ground-truth in hybrid manner",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "--no_data_lst",
        help="Do not dump data.lst",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--mapq_threshold",
        help="Threshold for mapping quality score",
        type=int,
        default=10
    )

    parser.add_argument(
        "--q_threshold",
        help="Threshold for base quality score",
        type=int,
        default=10
    )

    parser.add_argument(
        "--num_threads",
        type=int,
        default=30
    )

    parser.add_argument(
        "--reconcilement_size",
        help="Size of a hotspot region to enable reconcilement of pacbio/illumina representations",
        default=10,
        type=int,
    )

    parser.add_argument(
        "--keep_hdf5",
        help="Keep hdf5 files",
        default=False,
        action="store_true",
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

    ibams = None
    pbams = None

    if args.ibam:
        ibams = args.ibam.split(",")

    if args.pbam:
        pbams = args.pbam.split(",")

    create_hotspot_jobs_scripts = os.path.join(
        path, "createJobsDVFiltered.py"
    )

    shard_script = os.path.join(
        path, "shardHotspots.py"
    )

    caller_command = os.path.join(
        path, "caller.py"
    )

    # Create hotspot jobs
    if args.ibam and args.pbam:
        main(ibams, pbams, random_combine=args.random_combine)
    else:
        if args.ibam:
            main_single(ibams, pacbio=False)
        else:
            main_single(pbams, pacbio=True)
