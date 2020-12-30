# © 2019 University of Illinois Board of Trustees.  All rights reserved
"""
Script to produce VCF from training data
"""
import argparse
import vcfFromContigs
from PySamFastaWrapper import PySamFastaWrapper
import multiprocessing
# import MemmapData
import MemmapDataLite
import _pickle as pickle
import logging
import subprocess
import os

chr_prefix = ""


def vcf_from_file_wrapper(args):
    return vcf_from_file(*args)


def gen_header(ref):
    """
    Create header from reference file

    :param ref: str
        Reference filename
    """
    reference = PySamFastaWrapper(ref)
    header = "##fileformat=VCFv4.1\n"

    for chromosome in [str(x) for x in range(1, 23)] + list("XY"):
        chromosome = chr_prefix + chromosome
        reference.chrom = chromosome
        header += "##contig=<ID=%s,length=%d>\n" % (chromosome, len(reference))

    header += '##INFO=<ID=VCFFromTrainingData,Type=String,Number=1,Description="Labeled From Ground Truth">\n'
    header += '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
    header += '##FILTER=<ID=FAIL,Description="Failed call">\n'
    header += '#' + '\t'.join(["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT", "SAMPLE1"]) + "\n"
    return header


def vcf_from_file(filename, ref):
    """
    Produce VCF entries from one file

    :param filename: str
        Filename

    :param ref: str
        Reference path
    """
    data = pickle.load(open(filename, 'rb'))
    data.setIndexingMode('string')
    reference = PySamFastaWrapper(ref)

    records = []
    null_locations = []

    for location in data.locations:
        site_data = data[location]
        chromosome, start, stop = location.split("_")
        start = int(start)
        stop = int(stop)
        reference.chrom = chromosome
        ref_allele = ''.join(reference[start: stop])
        all_alleles = [a for a in site_data.keys() if a != 'siteLabel']
        labels = [site_data[a]['label'][0] > 0 for a in all_alleles]
        allele_label_dict = dict(zip(all_alleles, labels))
        alleles_except_ref_alleles = [a for a in all_alleles if a != ref_allele]
        reordered_alleles = [ref_allele] + alleles_except_ref_alleles
        gt = [i for i, a in enumerate(reordered_alleles) if ((a in allele_label_dict) and (allele_label_dict[a]))]

        # If there is any discrepancy avoid using the data point
        true_alleles = set(reordered_alleles[i] for i in gt)

        # if sum(site_data[a]['supportingReadsStrict'][0] if a in site_data else 0 for a in true_alleles) == 0:
        #     logging.debug("Warning, site %s has no supporting reads" % location)
        #     null_locations.append((chromosome, start, stop))
        #     continue

        if len(gt) == 0:
            logging.debug("Warning, site %s has no valid genotypes" % location)
            null_locations.append((chromosome, start, stop))
            continue

        if len(gt) == 1:
            gt = 2 * gt

        records.extend(
            vcfFromContigs.createVcfRecord(
                chromosome,
                0,
                reference,
                [start],
                [ref_allele],
                [alleles_except_ref_alleles],
                [gt],
                "VCFFromTrainingData",
            )
        )

    return records, null_locations


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create VCF file from training data")

    parser.add_argument(
        "--data",
        help="List of data files",
        required=True,
    )

    parser.add_argument(
        "--ref",
        help="Reference path",
        required=True,
    )

    parser.add_argument(
        "--vcf",
        help="Output vcf file",
        required=True,
    )

    parser.add_argument(
        "--nt",
        help="Number of threads to use",
        default=20,
        type=int,
    )

    parser.add_argument(
        "--chr_prefix",
        help="Chromosomal prefixes",
        default="",
        required=False
    )

    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="Display debug messages",
    )

    args = parser.parse_args()

    chr_prefix = args.chr_prefix

    logging.basicConfig(level=logging.INFO if not args.debug else logging.DEBUG)

    if args.nt > 0:
        workers = multiprocessing.Pool(args.nt)
        mapper = workers.imap_unordered
    else:
        mapper = map

    arguments = []

    for filename in open(args.data, 'r').readlines():
        filename = filename.strip()
        arguments.append(
            (filename, args.ref)
        )

    header = gen_header(args.ref)

    tmp_vcf = "/tmp/%s" % os.path.split(args.vcf)[-1]

    with open(tmp_vcf, 'w') as fhandle, open(os.path.splitext(args.vcf)[0] + ".null.bed", 'w') as bhandle:
        fhandle.write(header)
        for i, (result, null) in enumerate(mapper(vcf_from_file_wrapper, arguments)):
            for record in result:
                fhandle.write(str(record) + '\n')

            for n in null:
                bhandle.write("%s\t%d\t%d\n" % (n[0], n[1], n[2]))

            if (i + 1) % 500 == 0:
                logging.info("Completed processing %d files" % (i + 1))

    logging.info("Sorting vcf")

    with open(args.vcf, 'w')  as fhandle:
        subprocess.call(["vcf-sort", tmp_vcf], stdout=fhandle)

    logging.info("Done")
