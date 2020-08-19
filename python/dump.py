import os
import argparse
import subprocess
import random
import warnings
import logging
import glob

CHROMOSOMES = [str(i) for i in range(21)]


def main(ibams, pbams, random_combine=False):
    if not (len(ibams) > 0 and len(pbams) > 0):
        raise NotImplementedError

    if not random_combine:
        if len(ibams) > 1 or len(pbams) > 1:
            wanrings.warn("Non-random combine mode not suited for multiple IBAMs and PAMs")

    logging.info("Creating hotspot detection jobs")

    train_files = []

    for ib in ibams:
        for chrom in CHROMOSOMES:
            pbam_selected = random.sample(pbams, 1)[0] if random_combine else pbams[0]
            ib_string = ib.replace(".", "__").replace("/", "___")
            pb_string = pbam_selected.replace(".", "__").replace("/", "___")

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
                "--outputDir", "%s" % output_dir
            ]
            subprocess.call(create_cmd)

            command = os.path.join(output_dir, "jobs_chromosome%s.sh" % chrom)
            results = [os.path.join(output_dir, "jobs_chromosome%s_job%d.txt" % (chrom, i)) for i in range(501)]

            logging.info("Created jobs to create hotspots, running to generate hotspots")
            subprocess.call(
                "cat %s | shuf | parallel --eta -j 30" % command, shell=True, executable="/bin/bash"
            )

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
            caller_command_filename = os.path.join(output_dir, "caller_commands.sh")

            with open(caller_command_filename, "w") as fhandle:
                for shard in shards:
                    command_string = "python %s" % caller_command
                    command_string += " --activity %s" % shard
                    command_string += " --bam %s,%s" % (ib, pbam_selected)
                    command_string += " --ref %s" % args.ref
                    command_string += " --truth %s" % args.truth
                    command_string += " --highconf %s" % args.bed
                    command_string += " --featureLength %d" % 150
                    command_string += " --intersect"
                    command_string += " --reuse"
                    command_string += " --simple"
                    command_string += " --outputPrefix %s" % os.path.join(output_dir, "%s_data" % shard)
                    fhandle.write(command_string + " >& " + os.path.join(output_dir, "%s_log" % shard) + "\n")

            logging.info("Created data dump commands, launching data dump")

            subprocess.call(
                "cat %s | parallel -j 30 --eta" % caller_command_filename, shell=True, executable="/bin/bash"
            )

            logging.info("Completed data dump")

            # Collect list of results
            train_files.extend(glob.glob(os.path.join(output_dir, "*.index")))

    training_files = os.path.join(args.workdir, "data.lst")

    with open(training_files, "w") as dhandle:
        for line in train_files:
            dhandle.write(line + "\n")

    logging.info("Training data files in %s" % training_files)


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

    args = parser.parse_args()

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
    main(ibams, pbams, random_combine=args.random_combine)
