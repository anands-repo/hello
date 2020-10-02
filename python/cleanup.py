import shutil
import os
import subprocess
import sys
import logging
import random

SHARD_VCF_EXPR = "shard[0-9]*.txt_data.vcf"
SHARD_BED_EXPR = "shard[0-9]*.txt_data.bed"
SHARD_LOG_EXPR = "shard[0-9]*.txt_log"
HOTSP_TXT_EXPR = "jobs_chromosome*_job[0-9]*.txt"
HOTSP_LOG_EXPR = "jobs_chromosome*_job[0-9]*.log"


def find(expression, basedir):
    suffix = random.randint(1, 1000000)
    tmpname = "/tmp/finder%d.txt" % suffix
    with open(tmpname, 'w') as fhandle:
        subprocess.call(
            [
                "find",
                basedir,
                "-name",
                expression
            ],
            stdout=fhandle
        )
    return [x.strip() for x in open(tmpname, 'r').readlines()]


def move_file(source_name, target_base, source_base):
    # source_name = os.path.abspath(source_name)
    # target_base = os.path.abspath(target_base)
    # source_base = os.path.abspath(source_base)
    target_name = target_base + source_name[len(source_base):]
    surplus = os.path.split(target_name)[0]
    if not os.path.exists(surplus):
        logging.info("Making directory %s" % surplus)
        os.makedirs(surplus)
    logging.debug("Moving %s -> %s" % (source_name, target_name))
    shutil.move(source_name, target_name)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) != 3:
        logging.info("Usage: python %s <source directory> <target directory>" % (__file__))
        sys.exit()

    source = sys.argv[1]
    target = sys.argv[2]

    if os.path.exists(target):
        logging.error("Directory %s exists" % target)
        sys.exit()

    if not os.path.exists(source):
        logging.error("Source doesn't exist")
        sys.exit()

    os.makedirs(target)

    logging.info("Moving shard vcfs")
    
    for expression in [SHARD_VCF_EXPR, SHARD_BED_EXPR, SHARD_LOG_EXPR, HOTSP_TXT_EXPR, HOTSP_TXT_EXPR]:
        for source_name in find(expression, source):
            move_file(source_name, target, source)

    # Count number of files in the source
    num_files_orig = 0
    for r, d, f in os.walk(target):
        num_files_orig += len(f)

    # Tar the target and delete the directory
    logging.info("Tarring the archive")
    subprocess.call(
        [
            "tar", "cvf", "%s.tar" % target, target
        ],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

    shutil.rmtree(target)

    logging.info("Archived %d files in %s.tar" % (num_files_orig, target))
