# © 2019 University of Illinois Board of Trustees.  All rights reserved
"""
Prepare a VCF file from a set of features in the current directory
"""
import glob
import pickle
from vcfFromContigs import createVcfRecord
from PySamFastaWrapper import PySamFastaWrapper as ReferenceCache
from multiprocessing import Pool
import argparse
import os
import subprocess
import math
import shutil
import numpy as np
import sys


def checkLogs(path):
    shardFiles = glob.glob("%s/shard[0-9]*.txt" % path);
    print("Found %d input shard files" % len(shardFiles));

    # For each shard file check whether the log file has the appropriate string
    for shard in shardFiles:
        logName = shard + ".log";
        with open(logName, 'r') as fhandle:
            if "Completed running the script" not in fhandle.read():
                print("File %s doesn't have termination string" % logName);
                return False;

    return True;


def callAlleles(likelihoodDict, chromosome, start, length, ref):
    """
    Given a likelihood dictionary, call alleles

    :param likelihoodDict: dict
        Dictionary of likelihoods

    :param chromosome: str
        Chromosome

    :param start: int
        Position in the reference sequence

    :param length: int
        Length of variant in reference sequence

    :param ref: ReferenceCache
        Reference cache object

    :return: str
        Variant record
    """
    refAllele = ''.join(ref[start: start + length]);
    topAlleleCombination = sorted([(v, k) for k, v in likelihoodDict.items()], reverse=True)[0];
    likelihood, topAlleles = topAlleleCombination;
    likelihood = min(float(likelihood), 1 - 1e-8);  # Quality score restricted to value 80
    quality = -10 * math.log10(1 - likelihood);
    altAlleles = list(set(topAlleles).difference({refAllele}));
    allelesAtSite = set();

    for key in likelihoodDict:
        for k in key:
            allelesAtSite.add(k);

    allelesAtSite = list(allelesAtSite);

    if len(altAlleles) == 0:
        genotypes = [0, 0];
        allAlleles = allelesAtSite;
        altAlleles = list(set(allAlleles).difference({refAllele}));
        if len(altAlleles) == 0:
            return None;
    else:
        genotypes = [];

        for allele in topAlleles:
            if allele == refAllele:
                genotypes.append(0);
            else:
                try:
                    altIndex = altAlleles.index(allele);
                except ValueError:
                    # Rethrow with message (should probably put in a finally block)
                    raise ValueError("Cannot find allele %s in altAllele list %s" % (allele, str(altAlleles)));

                genotypes.append(altIndex + 1);

    record = createVcfRecord(
        chromosome,
        start,
        ref,
        [0],
        [refAllele],
        [altAlleles],
        [genotypes],
        string="MixtureOfExpertPrediction",
        qual=quality
    )[0];

    return record;


def vcfWrapper(args):
    return vcfRecords(*args);


def vcfRecords(data, ref, tmpdir):
    """
    Creates vcf records for a site

    :param data: str
        Filename containing site predictions from each expert and meta-expert predictions

    :param ref: str
        Reference cache database location

    :param tmpdir: str
        Temporary directory path
    """
    items = pickle.load(open(data, 'rb'));
    prefix = os.path.join(tmpdir, os.path.split(data)[1]);
    ref = ReferenceCache(database=ref);

    e0handle = open(prefix + ".expert0.vcf", 'w');
    e1handle = open(prefix + ".expert1.vcf", 'w');
    e2handle = open(prefix + ".expert2.vcf", 'w');
    bhandle = open(prefix + ".best.vcf", 'w');
    mhandle = open(prefix + ".mean.vcf", 'w');
    chandle = open(prefix + ".choices.bed", 'w');

    chromosomes = set();

    for siteDict in items:
        if ref.chrom != siteDict['chromosome']:
            ref.chrom = siteDict['chromosome'];

        expertRecords = [
            callAlleles(likelihoodDict, siteDict['chromosome'], siteDict['position'], siteDict['length'], ref)
            for likelihoodDict in siteDict['expertPredictions']
        ];

        e0handle.write(expertRecords[0] + '\n');
        e1handle.write(expertRecords[1] + '\n');
        e2handle.write(expertRecords[2] + '\n');

        bestRecord = expertRecords[np.argmax(siteDict['meta'])];
        bhandle.write(bestRecord + '\n');

        def average(allelePairing):
            return sum(
                float(siteDict['expertPredictions'][i][allelePairing]) * float(siteDict['meta'][i])
                for i in range(3)
            );

        meanLikelihoodDict = dict({
            allelePairing: average(allelePairing) for allelePairing in siteDict['expertPredictions'][0]
        });

        meanRecord = callAlleles(
            meanLikelihoodDict, siteDict['chromosome'], siteDict['position'], siteDict['length'], ref
        );

        mhandle.write(meanRecord + '\n');

        chandle.write('\t'.join([
            siteDict['chromosome'],
            str(siteDict['position']),
            str(siteDict['position'] + siteDict['length']),
            str(np.argmax(siteDict['meta']))
        ]) + '\n');

        chromosomes.add(siteDict['chromosome']);

    for h in [e0handle, e1handle, e2handle, bhandle, mhandle, chandle]:
        h.close();

    return chromosomes;


def headerString(ref, chromosomes, info):
    ref = ReferenceCache(database=ref);
    string = "##fileformat=VCFv4.1\n";
    for chromosome in chromosomes:
        ref.chrom = chromosome;
        length = len(ref);
        string += "##contig=<ID=%s,length=%d>\n" %(chromosome, length);
    # string += '##INFO=<ID=MixtureOfExpertsPrediction,Description="Obtained from mixture-of-experts">\n';
    string += info + '\n';
    string += '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n';
    string += '##FILTER=<ID=FAIL,Description="Failed call">\n';
    string += "#" + '\t'.join("CHROM  POS     ID      REF     ALT     QUAL    FILTER  INFO    FORMAT  SAMPLE1".split()) + '\n';
    return string;


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create VCF files from a set of feature files in a directory"
    );

    parser.add_argument(
        "--prefix",
        help="Prefix of sharded features",
        required=True,
    );

    parser.add_argument(
        "--ref",
        help="Reference cache location",
        required=True,
    );

    parser.add_argument(
        "--tmpdir",
        help="Temporary directory",
        default="/tmp/vcftemp"
    );

    parser.add_argument(
        "--numThreads",
        help="Number of CPU threads to use",
        default=4,
        type=int,
    );

    parser.add_argument(
        "--outputPrefix",
        help="Prefix of output file",
        required=None,
    );

    parser.add_argument(
        "--checkRuns",
        help="Check runs before proceeding",
        default=False,
        action="store_true",
    );

    args = parser.parse_args();

    allResults = glob.glob("%s*.features" % args.prefix);

    if args.checkRuns:
        path = os.path.split(args.prefix)[0];
        print("Checking logs in path %s" % path);
        if checkLogs(path):
            print("All runs have completed");
        else:
            print("Some runs weren't completed. Stopping ... ");
            sys.exit(-1);

    if os.path.exists(args.tmpdir):
        shutil.rmtree(args.tmpdir);

    os.makedirs(args.tmpdir);

    # First create sub-vcf files
    mapper = Pool(args.numThreads).imap_unordered;

    callArgs = [
        (result, args.ref, args.tmpdir)
        for result in allResults
    ];

    chromosomes = set();

    for i, chr_ in enumerate(mapper(vcfWrapper, callArgs)):
        chromosomes = chromosomes.union(chr_);
        if (i + 1) % 100 == 0:
            print("Completed processing %d files" % (i + 1));

    def combineVcfs(suffix, info, label):
        searchString = os.path.join(args.tmpdir, suffix);
        print("Search string %s" % searchString);
        allFiles = glob.glob(searchString);
        print("Found %d files" % len(allFiles));
        header = headerString(
            args.ref,
            chromosomes,
            info='##INFO=<ID=%s,Description="%s"' % (info[0], info[1])
        );
        tempvcf = args.outputPrefix + ".%s.tmp.vcf" % label;
        finalvcf = args.outputPrefix + ".%s.vcf" % label;
        with open(tempvcf, 'w') as fhandle:
            fhandle.write(header);
            for f in allFiles:
                contents = open(f, 'r').read().rstrip();
                if len(contents) > 0:
                    fhandle.write(contents + '\n');
        with open(finalvcf, 'w') as fhandle:
            command = ["vcf-sort", tempvcf];
            print("Running command %s" % str(command));
            subprocess.call(
                command, stdout=fhandle
            );
        os.remove(tempvcf);

    # Create VCF headers and combine sub-vcf files
    combineVcfs(
        "*expert0.vcf",
        ("MixtureOfExpertPrediction", "Prediction from NGS expert"),
        label="ngs"
    );

    combineVcfs(
        "*expert1.vcf",
        ("MixtureOfExpertPrediction", "Prediction from TGS expert"),
        label="tgs"
    );

    combineVcfs(
        "*expert2.vcf",
        ("MixtureOfExpertPrediction", "Prediction from NGS_TGS expert"),
        label="ngs_tgs"
    );

    combineVcfs(
        "*best.vcf",
        ("MixtureOfExpertPrediction", "Prediction from best expert"),
        label="best"
    );

    combineVcfs(
        "*mean.vcf",
        ("MixtureOfExpertPrediction", "Mean predictions from experts"),
        label="mean"
    );

    # Combine all bed files and sort
    allBed = glob.glob(os.path.join(args.tmpdir, "*.choices.bed"));
    choiceCounts = dict({0: 0, 1: 0, 2: 0});
    with open(args.outputPrefix + ".choices.bed", 'w') as fhandle:
        for f in allBed:
            with open(f, 'r') as rhandle:
                for line in rhandle.readlines():
                    line = line.rstrip();
                    if (len(line) > 0):
                        fhandle.write(line + '\n');
                        items = line.split();
                        choice = int(items[3]);
                        choiceCounts[choice] += 1;

    print("Choice histogram = %s" % (str(choiceCounts)));
