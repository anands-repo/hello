# © 2019 University of Illinois Board of Trustees.  All rights reserved
"""
Script generates random combinations for the dump command for a set of reads
"""
import argparse
import dump
import os
import random
from find_chr_prefixes import get_reference_prefixes

CHROMOSOMES = dump.CHROMOSOMES

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate dump commands with random mixing of hybrid bam files"
    )

    parser.add_argument(
        "--ibams",
        help="File containing a list of bam files for one sequencing technology",
        required=True,
    )

    parser.add_argument(
        "--pbams",
        help="File containing a list of bam files for one sequencing technology",
        required=True,
    )

    parser.add_argument(
        "--ref",
        help="Reference fasta",
        required=True,
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
        "--iteration_base",
        choices=["i", "p"],
        default="i",
        help="Whether to keep ibams as base of iteration or pbams as base",
    )

    parser.add_argument(
        "--chr_prefix",
        help="Prefix of chromosomes",
        default="",
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

    args = parser.parse_args()

    CHROMOSOMES = [args.chr_prefix + c for c in CHROMOSOMES]

    ref_prefix = get_reference_prefixes(args.ref)

    if ref_prefix:
        CHROMOSOMES = [ref_prefix + i for i in CHROMOSOMES]

    if not os.path.exists(args.workdir):
        os.makedirs(args.workdir)

    ibams = [i.strip() for i in open(args.ibams).readlines()]
    pbams = [p.strip() for p in open(args.pbams).readlines()]

    if args.iteration_base == "i":
        bam_base = ibams
        bam_select = pbams
        bam_base_designation = "--ibam"
        bam_select_designation = "--pbam"
    else:
        bam_base = pbams
        bam_select = ibams
        bam_base_designation = "--pbam"
        bam_select_designation = "--ibam"

    dump_command = os.path.join(
        os.path.split(os.path.abspath(__file__))[0],
        "dump.py"
    )

    with open(os.path.join(args.workdir, "commands.sh"), 'w') as fhandle:
        for bam in bam_base:
            for chrom in CHROMOSOMES:
                selected_bam = random.sample(bam_select, 1)[0]

                command_string = "python %s" % dump_command
                command_string += " %s %s" % (bam_base_designation, bam)
                command_string += " %s %s" % (bam_select_designation, selected_bam)
                command_string += " --chromosomes %s" % chrom
                command_string += " --ref %s" % args.ref
                command_string += " --workdir %s" % args.workdir
                command_string += " --truth %s" % args.truth
                command_string += " --bed %s" % args.bed
                command_string += " --hybrid_eval"
                command_string += " --no_data_lst"
                command_string += " --q_threshold %d" % args.q_threshold
                command_string += " --mapq_threshold %d" % args.mapq_threshold
                command_string += " --num_threads %d" % args.num_threads
                command_string += " --reconcilement_size %d" % args.reconcilement_size

                fhandle.write(command_string + '\n')
