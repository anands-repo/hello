# © 2019 University of Illinois Board of Trustees.  All rights reserved
"""
Analyze variants directly from Pileup data
"""
from PileupContainerLite import PileupContainerLite
# from ReferenceCache import ReferenceCache
from PySamFastaWrapper import PySamFastaWrapper as ReferenceCache
import pickle
from functools import reduce
import numpy as np
from vcfFromContigs import createVcfRecord
import argparse
import logging
import vcf
import sys
import ast
import torch
import math
import AlleleSearcherDNN
import os
import MixtureOfExpertsTools
import MixtureOfExpertsAdvanced
import PileupDataTools
import trainDataTools
import h5py
import libCallability
from MemmapDataLite import MemmapperCompound
import pybedtools
import random

torch.set_num_threads(1);  # Torch to be run using a single thread
random.seed(13)
np.random.seed(13)

try:
    profile
except NameError:
    def profile(x):
        return x;


FEATURE_LENGTH = 150


def one_hot_encode(bases):
    """
    One-hot encode a base sequence

    :param bases: str
        Base sequence

    :return: np.array
        A list with one-hot-encoded values
    """
    indices = dict(zip('ACGT', range(4)))
    zeros = torch.zeros(len(bases), 5)
    encoded_bases = [indices[i] if (i in indices) else 4 for i in bases]
    zeros[list(range(len(bases))), encoded_bases] = 1
    return zeros


def get_reference_segment(chromosome, start, stop, ref):
    """
    Get reference slices in a given region

    :param chromosome: str
        Chromosome

    :param start: int
        Start position of variant location

    :param stop: int
        Stop position of variant location

    :param ref: PySamFastaWrapper
        Reference object

    :return: np.array
        Numpy array with one-hot encoded bases
    """
    span = FEATURE_LENGTH
    ref.chrom = chromosome
    midpoint = (start + stop) // 2
    left = midpoint - span // 2
    right = left + span
    refbases = ''.join(ref[left: right])
    return one_hot_encode(refbases)


def chromosomesInCluster(hotspots):
    """
    Determine chromosomes in a hotspots file

    :param hotspots: str
        Hotspots filename

    :return: dict
        Chromosomes and first last positions within the chromosomes in the cluster
    """
    chromosomes = dict();

    with open(hotspots, 'r') as fhandle:
        for line in fhandle:
            point = ast.literal_eval(line);
            chromosome = point['chromosome'];
            position = point['position'];
            if chromosome in chromosomes:
                values = chromosomes[chromosome];
                chromosomes[chromosome] = (min(values[0], position), max(values[1], position + 1));
            else:
                chromosomes[chromosome] = (position, position + 1);

    return chromosomes;


def number_alleles(alleles, ref_allele):
    """
    Add VCF_type numbering to alleles

    :param alleles: list
        List of alleles

    :param ref_allele: str
        Reference allele

    :return: dict
        A dictionary mapping alleles to number
    """
    numbering = dict()
    counter = 1

    for a in alleles:
        if a != ref_allele:
            numbering[a] = counter
            counter += 1

    numbering[ref_allele] = 0

    return numbering


def dump_vcfs_from_labels(entry, ref, vhandle):
    """
    Given site data, dump out a vcf file. This is to test if the site data has been
    correctly labeled or not

    :param entry: dict
        Site data dictionary

    :param ref: ReferenceCache
        Accessor for indexed references

    :param vhandle: File
        File handle into which to write VCF
    """
    logging.debug("Received item from datagen")
    site_type = 'PASS'

    if 'type' in entry:
        site_type = entry['type']
        logging.debug("Discarding entry")
        return

    alleles = entry['alleles']
    chromosome = entry['chromosome']
    ref.chrom = chromosome
    start = entry['start']
    stop = entry['stop']
    ref_allele = ''.join(ref[start: stop])
    labels = entry['labels']
    allele_numbering = number_alleles(alleles, ref_allele)
    positive_alleles = set(a for a, l in zip(alleles, labels) if l > 0)
    genotypes = [allele_numbering[a] for a in positive_alleles]
    alt_alleles = [a for a in alleles if a != ref_allele]

    if len(genotypes) == 0:
        genotypes = [0, 0]
        site_type = 'UNLABELED'

    if len(genotypes) == 1:
        genotypes += genotypes

    if len(alt_alleles) == 0:
        alt_alleles = [ref_allele]
        site_type = 'NO_ALT'

    logging.debug("Creating vcf record")

    record = createVcfRecord(
        chromosome,
        start,
        ref,
        [0],
        [ref_allele],
        [alt_alleles],
        [genotypes],
        string="LabeledFromGroundTruth",
        qual=50,
        qualifier=site_type
    )

    logging.debug("Completed ceating vcf record")

    for r in record:
        vhandle.write(r + '\n')


def intersect(vcf, bed, hotspots, outputPrefix):
    """
    Given a hotspots file, determine minimally spanning bed/vcf files

    :param vcf: str
        Vcf file

    :param bed: str
        Bed file

    :param hotspots: str
        hotspots file

    :param outputPrefix: str
        Prefix of output file

    :return: tuple
        (new vcf, new bed)
    """
    bedName = outputPrefix + ".bed";
    vcfName = outputPrefix + ".vcf";
    chromosomes = chromosomesInCluster(hotspots);
    bedString = "";

    for i, (key, value) in enumerate(chromosomes.items()):
        chromosome = key;
        start = value[0] - 10;
        stop = value[1] + 10;
        bedString += "\n" if i > 0 else "";
        bedString += "\t".join([chromosome, str(start), str(stop)]);

    subset = pybedtools.BedTool(bedString, from_string=True);
    vcfBed = pybedtools.BedTool(vcf);
    hcBed = pybedtools.BedTool(bed);
    vcfBed.intersect(subset, header=True).saveas(vcfName);
    hcBed.intersect(subset).saveas(bedName);

    return (vcfName, bedName);


class NoReadsFoundError(Exception):
    """
    Custom exception thrown by PileupAnalyzer indicating that
    there are no reads at the site
    """
    def __init__(self, value):
        self.value = value;

    def __str__(self):
        return repr(self.value);


def copy(inhandle, outhandle, location, exclude_keys):
    outgroup = outhandle.create_group(location);
    ingroup = inhandle[location];

    for allele in ingroup.keys():
        if allele == 'siteLabel':
            dset = outgroup.create_dataset('siteLabel', dtype=np.int32, shape=(1, ));
            dset[:] = ingroup['siteLabel'][:];
            continue;

        alleleInGroup = ingroup[allele];
        alleleOutGroup = outgroup.create_group(allele);

        for attribute in alleleInGroup.keys():
            if attribute in exclude_keys:
                continue
            dset = alleleOutGroup.create_dataset(
                attribute, dtype=alleleInGroup[attribute].dtype, shape=alleleInGroup[attribute].shape
            );
            dset[:] = alleleInGroup[attribute][:];


def postProcessHdf5(
    hdf5name,
    outputPrefix,
    hybrid,
):
    """
    For generating training data, convert the pure hdf5 file into .index format

    :param hdf5name: str
        Name of the hdf5 file

    :param outputPrefix: str
        Prefix of the output file

    :param hybrid: bool
        Whether we are in hybrid mode or not
    """
    # Anand: Added reference segment on September 14 2020
    memmapperKeys = ['feature', 'feature2', 'label', 'segment'];
    memmapperConcatenations = [False, False, True, False];

    if not hybrid:
        # Anand: Added reference segment on September 14 2020
        memmapperKeys = ['feature', 'label', 'segment'];
        memmapperConcatenations = [False, True, False];

    # Anand: Added reference segment on September 14 2020
    dtypes = {'feature': 'uint8', 'feature2': 'uint8', 'segment': 'uint8', 'label': 'float32'};

    logging.info("Converting to integrated format");

    MemmapperCompound(
        hdf5name,
        prefix=outputPrefix,
        keys=memmapperKeys,
        concatenations=memmapperConcatenations,
        hybrid=hybrid,
        mode='integrated',
        dtypes=dtypes,
        memtype="memmap",
    );


def addToDict(
    fhandle,
    chromosome,
    start,
    stop,
    alleles,
    tensors,
    labels,
    siteLabel,
    scores,
    supportingReads,
    supportingReadsStrict,
    tensors2,
    supportingReads2,
    supportingReadsStrict2,
    segment,
):
    """
    Add contents to an hdf5 file

    :param fhandle: dict
        Dictionary data into which to add items

    :param chromosome: str
        Chromosome

    :param start: int
        Start position in chromosome reference

    :param stop: int
        Stop position in chromosome reference

    :param alleles: list
        List of alleles at site

    :param tensors: list
        Feature tensors (np.ndarray)

    :param labels: list
        List of labels in the same order as tensors

    :param siteLabel: int
        Whether site is heterozygous or homozygous

    :param scores: list
        Scoring results from network

    :param supportingReads: int
        Number of supporting reads for each allele

    :param supportingReadsStrict: int
        Number of reads containing the allele in its entirety

    :param tensors2: list
        Second set of feature tensors for hybrid variant calling

    :param supportingReads2: list
        Second set of number of supporting reads for hybrid calling

    :param supportingReadsStrict2:
        Second set of strict supporting reads for hybrid calling

    :param segment: np.array
        Reference segment
    """
    logging.debug("Writing chr, start, stop = %s, %d, %d to file" % (chromosome, start, stop));
    groupName = '_'.join([chromosome, str(start), str(stop)]);  # We will maintain a flat hierarchy

    if groupName in fhandle:
        del fhandle[groupName];
    else:
        fhandle[groupName] = dict()
        mainGroup = fhandle[groupName]

        for i, (allele, tensor, label) in enumerate(zip(alleles, tensors, labels)):
            mainGroup[allele] = dict()
            alleleGroup = mainGroup[allele];
            alleleGroup['label'] = np.zeros(shape=(1, ), dtype=np.float32)
            alleleGroup['feature'] = np.zeros(shape=tensor.shape, dtype=tensor.dtype)
            alleleGroup['label'][:] = label;
            alleleGroup['feature'][:] = tensor;
            alleleGroup['supportingReads'] = [supportingReads[i]];
            alleleGroup['supportingReadsStrict'] = [supportingReadsStrict[i]];

            # Adding reference segment: adding it to each allele (September 14 2020)
            alleleGroup['segment'] = np.zeros(shape=segment.shape, dtype=np.uint8)
            alleleGroup['segment'][:] = segment[:]

            if (len(tensors2) > 0):
                tensor2 = tensors2[i];
                alleleGroup['feature2'] = np.zeros(shape=tensor2.shape, dtype=tensor2.dtype)
                alleleGroup['feature2'][:] = tensor2;
                alleleGroup['supportingReads2'] = [supportingReads2[i]];
                alleleGroup['supportingReadsStrict2'] = [supportingReadsStrict2[i]];

        mainGroup['siteLabel'] = siteLabel


def addToHDF5File(
    fhandle,
    chromosome,
    start,
    stop,
    alleles,
    tensors,
    labels,
    siteLabel,
    scores,
    supportingReads,
    supportingReadsStrict,
    tensors2,
    supportingReads2,
    supportingReadsStrict2,
    segment,
):
    """
    Add contents to an hdf5 file

    :param fhandle: h5py.File
        File object into which a group should be added

    :param chromosome: str
        Chromosome

    :param start: int
        Start position in chromosome reference

    :param stop: int
        Stop position in chromosome reference

    :param alleles: list
        List of alleles at site

    :param tensors: list
        Feature tensors (np.ndarray)

    :param labels: list
        List of labels in the same order as tensors

    :param siteLabel: int
        Whether site is heterozygous or homozygous

    :param scores: list
        Scoring results from network

    :param supportingReads: int
        Number of supporting reads for each allele

    :param supportingReadsStrict: int
        Number of reads containing the allele in its entirety

    :param tensors2: list
        Second set of feature tensors for hybrid variant calling

    :param supportingReads2: list
        Second set of number of supporting reads for hybrid calling

    :param supportingReadsStrict2:
        Second set of strict supporting reads for hybrid calling

    :param segment: np.array
        Reference segment
    """
    logging.debug("Writing chr, start, stop = %s, %d, %d to file" % (chromosome, start, stop));
    groupName = '_'.join([chromosome, str(start), str(stop)]);  # We will maintain a flat hierarchy

    try:
        # If group already exists, delete it: it might be erroneous
        groupData = fhandle[groupName];
        del fhandle[groupName];
    except Exception:
        # If group doesn't exist, add data
        mainGroup = fhandle.create_group(groupName);

        for i, (allele, tensor, label) in enumerate(zip(alleles, tensors, labels)):
            alleleGroup = mainGroup.create_group(allele);
            alleleGroup.create_dataset('label', shape=(1,), dtype='float32');
            alleleGroup.create_dataset('feature', shape=tensor.shape, dtype=tensor.dtype);
            alleleGroup.create_dataset('supportingReads', shape=(1,), dtype='int32');
            alleleGroup.create_dataset('supportingReadsStrict', shape=(1,), dtype='int32');
            alleleGroup['label'][:] = label;
            alleleGroup['feature'][:] = tensor;
            alleleGroup['supportingReads'][:] = supportingReads[i];
            alleleGroup['supportingReadsStrict'][:] = supportingReadsStrict[i];

            # Adding reference segment: adding it to each allele (September 14 2020)
            alleleGroup.create_dataset('segment', shape=segment.shape, dtype='uint8')
            alleleGroup['segment'][:] = segment[:]

            if (len(tensors2) > 0):
                tensor2 = tensors2[i];
                alleleGroup.create_dataset('feature2', shape=tensor2.shape, dtype='float32');
                alleleGroup.create_dataset('supportingReads2', shape=(1,), dtype='int32');
                alleleGroup.create_dataset('supportingReadsStrict2', shape=(1,), dtype='int32');
                alleleGroup['feature2'][:] = tensor2;
                alleleGroup['supportingReads2'][:] = supportingReads2[i];
                alleleGroup['supportingReadsStrict2'][:] = supportingReadsStrict2[i];

        mainGroup.create_dataset('siteLabel', shape=(1,), dtype=int);
        mainGroup['siteLabel'][:] = siteLabel;


def dumpTrainingData(datagen, filename, hybrid=False, dump_vcf=False, ref=None, keep=False):
    """
    Dump training data to disk

    :param datagen: trainDataTools.data
        Iterator from a generator

    :param filename: str
        Prefix of the output file

    :param hybrid: bool
        Whether we are operating in hybrid mode or not

    :param dump_vcf: bool
        Dump vcf file from labeled ground-truth to test labeling accuracy

    :param ref: PySamFastaWrapper
        Reference reader

    :param keep: bool
        Keep HDF5 data in original format
    """
    missedCalls = [];
    tooLong = [];
    hdf5 = filename + ".hdf5";
    vcf = filename + ".labeled.vcf"

    if dump_vcf:
        vhandle = open(vcf, 'w')

    if keep:
        fhandle = h5py.File(filename + ".hdf5", 'w')
    else:
        fhandle = dict()

    for i, entry in enumerate(datagen):
        if 'type' in entry:
            if entry['type'] == "TOO_LONG":
                tooLong.append(entry);
            else:
                missedCalls.append(entry);
        else:
            # Update Sept 14 2020 - add reference segment here
            entry['segment'] = get_reference_segment(
                entry['chromosome'],
                entry['start'],
                entry['stop'],
                ref,
            )

            entry['fhandle'] = fhandle;

            if keep:
                addToHDF5File(**entry);
            else:
                addToDict(**entry)

        if (i + 1) % 100 == 0:
            logging.info("Completed %d locations" % (i + 1));

        dump_vcfs_from_labels(entry, ref, vhandle)

    if not keep:
        postProcessHdf5(fhandle, filename, hybrid=hybrid)
    else:
        fhandle.close()

    if dump_vcf:
        vhandle.close()


def scoreSite(siteDict, network, cache, hybrid=False):
    """
    Scores the alleles at a site based on data

    :param siteDict: dict
        Tensors and other information at a site

    :param network: torch.nn.Module
        Neural network to compute likelihoods

    :param cache: PySamFastaWrapper
        A Reference object

    :param hybrid: bool
        Whether we are running the tool in hybrid mode
    """
    featureDict = dict();

    for i, (allele, t1) in enumerate(zip(siteDict['alleles'], siteDict['tensors'])):
        if hybrid:
            tensorPacket = (torch.Tensor(t1), torch.Tensor(siteDict['tensors2'][i]));
        else:
            tensorPacket = (torch.Tensor(t1), None);

        featureDict[allele] = tensorPacket;

    # Find reference (September 14 2020 - update for using reference segment)
    ref_segment = get_reference_segment(
        siteDict['chromosome'],
        siteDict['start'],
        siteDict['stop'],
        cache,
    )
    ref_segment = torch.unsqueeze(ref_segment, dim=0)

    with torch.no_grad():
        logLikelihoods = network(featureDict, ref_segment);

    return logLikelihoods;


def vcfRecords(siteDict, network, ref, hybrid=False):
    """
    Creates vcf records for a site

    :param siteDict: dict
        Tensors and other information at a site

    :param network: torch.nn.Module
        Neural network to compute likelihoods

    :param ref: ReferenceCache
        Reference cache for obtaining reference allele etc

    :param hybrid: bool
        Whether we are running the tool in hybrid mode
    """
    if ref.chrom != siteDict['chromosome']:
        ref.chrom = siteDict['chromosome'];

    supportMsg = dict(
        zip(
            siteDict['alleles'],
            zip(siteDict['supportingReadsStrict'], siteDict['supportingReadsStrict2'])
        )
    );

    logging.debug(
        "Supports at site = %s" % str(supportMsg)
    );

    returns = scoreSite(siteDict, network, ref, hybrid);

    if type(returns) is dict:
        logLikelihoods = returns;
        features = None;
    else:
        logLikelihoods = returns[0];
        features = returns[1:];

    logging.debug("Likelihoods at site = %s" % str(logLikelihoods));

    refAllele = ''.join(ref[siteDict['start']: siteDict['stop']]);
    topAlleleCombination = sorted([(v, k) for k, v in logLikelihoods.items()], reverse=True)[0];
    logLikelihood, topAlleles = topAlleleCombination;
    logLikelihood = min(float(logLikelihood), 1 - 1e-8);  # Quality score restricted to value 80
    quality = -10 * math.log10(1 - logLikelihood);
    altAlleles = list(set(topAlleles).difference({refAllele}));
    allelesAtSite = siteDict['alleles'];

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
        siteDict['chromosome'],
        siteDict['start'],
        # ref.database,
        ref,
        [0],
        [refAllele],
        [altAlleles],
        [genotypes],
        string="MixtureOfExpertPrediction",
        qual=quality
    )[0];

    if features is None:
        return record;
    else:
        expert0, expert1, expert2, meta = features;
        features = dict({
            'chromosome': siteDict['chromosome'],
            'position': siteDict['start'],
            'length': len(refAllele),
            'meta': meta.cpu().data.numpy(),
            'expertPredictions': (expert0, expert1, expert2),
        });
        return record, features;


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Call variants for Mixture of Experts model");

    parser.add_argument(
        "--bam",
        help="Comma-separated list of BAM files",
        required=True,
    );

    parser.add_argument(
        "--activity",
        help="Active regions file",
        required=True,
    );

    parser.add_argument(
        "--ref",
        help="Reference cache location",
        required=True,
    );

    parser.add_argument(
        "--network",
        help="AlleleSearcherDNN trained model path",
        required=False,
    );

    parser.add_argument(
        "--outputPrefix",
        help="Prefix of output file",
        required=True,
    );

    parser.add_argument(
        "--debug",
        help="Enable debug mode",
        action="store_true",
        default=False,
    );

    parser.add_argument(
        "--provideFeatures",
        help="Provide features in addition to vcf records",
        default=False,
        action="store_true",
    );

    parser.add_argument(
        "--hotspotMode",
        help="Mode for hotspot creation",
        choices=["BAM1", "BOTH", "INCLUSIVE"],
        default="BOTH",
    );

    parser.add_argument(
        "--chrPrefixes",
        help="Comma-separated list of prefix for BAM files",
        default=None,
    );

    parser.add_argument(
        "--featureLength",
        help="Length of feature maps",
        default=100,
        type=int,
    );

    parser.add_argument(
        "--truth",
        help="Ground truth VCF if training data generation is desired",
        default=None,
    );

    parser.add_argument(
        "--highconf",
        help="High-confidence bed file if generating training data is desired",
        default=None,
    );

    parser.add_argument(
        "--intersectRegions",
        help="Intersect ground-truth/high-confidence regions with hotspot range before running",
        default=False,
        action="store_true",
    );

    parser.add_argument(
        "--simple",
        help="Use simple features",
        default=False,
        action="store_true",
    );

    parser.add_argument(
        "--reuseSearchers",
        help="Re-use searchers between hotspots and feature generation",
        default=False,
        action="store_true",
    );

    parser.add_argument(
        "--noAlleleLevelFilter",
        help="Do not use allele-level filters for reads",
        default=False,
        action="store_true",
    );

    parser.add_argument(
        "--clr",
        action="store_true",
        help="Second read type is CLR instead of CCS",
        default=False,
    );

    parser.add_argument(
        "--hybrid_hotspot",
        help="Use hybrid hotspot detection method",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "--pacbio",
        help="For single BAM mode indicates whether a read is pacbio or not",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "--test_labeling",
        help="This mode uses ground-truth variants etc, and dumps out a vcf file to test whether labeling is correct",
        action="store_true",
        default=False,        
    )

    parser.add_argument(
        "--only_contained",
        help="For training data dump, only use locations that are fully contained in confident regions",
        default=False,
        action="store_true"
    )

    parser.add_argument(
        "--hybrid_eval",
        help="Evaluate ground-truth in hybrid manner",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "--q_threshold",
        help="Quality score threshold",
        default=10,
        type=int,
    )

    parser.add_argument(
        "--mapq_threshold",
        help="Mapping quality threshold",
        default=10,
        type=int,
    )

    parser.add_argument(
        "--keep_hdf5",
        help="Retain data as hdf5, do not convert to memmap",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "--reconcilement_size",
        help="Size of a hotspot region to enable reconcilement of pacbio/illumina representations",
        default=10,
        type=int,
    )

    args = parser.parse_args();

    FEATURE_LENGTH = args.featureLength

    if args.only_contained:
        trainDataTools.STRICT_INTERSECTION = True

    if args.hybrid_eval:
        trainDataTools.HYBRID_TRUTH_EVAL = True

    libCallability.initLogging(args.debug);
    logging.basicConfig(
        level=logging.INFO if not args.debug else logging.DEBUG, format='%(asctime)-15s %(message)s'
    );

    if args.intersectRegions and (args.highconf is not None):
        logging.info("Shrinking ground-truth to fit the given active regions");
        newVcf, newBed = intersect(args.truth, args.highconf, args.activity, args.outputPrefix);
        args.truth = newVcf;
        args.highconf = newBed;
        logging.info("Obtained new vcf and bed files %s, %s" % (args.truth, args.highconf));

    # Create read factories and searcher factories
    bamfiles = args.bam.split(",");
    bamReaders = [];
    chrPrefixes = ["" for _ in bamfiles];

    if args.chrPrefixes is not None:
        chrPrefixes = args.chrPrefixes.split(",");
        assert(len(chrPrefixes) == len(bamfiles)), \
            "When provided, the length of chr prefixes should be the same as that of the bam files";

    readSamplers = [];

    for i, (bam, chrPrefix) in enumerate(zip(bamfiles, chrPrefixes)):
        reader = PileupDataTools.ReadSampler(
            bamfile=bam,
            readRate=(PileupDataTools.READ_RATE_ILLUMINA if i == 0 else PileupDataTools.READ_RATE_PACBIO),
            chrPrefix=chrPrefix,
            pacbio=(i > 0) or (len(bamfiles) == 1 and args.pacbio),
        );
        readSamplers.append(reader);

    searcherFactory = PileupDataTools.SearcherFactory(
        ref=args.ref,
        featureLength=args.featureLength,
        pacbio=args.pacbio,
        useInternalLeftAlignment=False,
        noAlleleLevelFilter=args.noAlleleLevelFilter,
        clr=False,      # TBD: Currently doesn't support CLR reads
        hybrid_hotspot=args.hybrid_hotspot,
        min_mapq=args.mapq_threshold,
        q_threshold=args.q_threshold,
        reassembly_size=args.reconcilement_size,
    );

    logging.info("Getting callable sites");

    # Obtain the list of hotspots to be called
    hotspots, searcherCollection = PileupDataTools.candidateReader(
        readSamplers=readSamplers,
        searcherFactory=searcherFactory,
        activity=args.activity,
        hotspotMode=args.hotspotMode,
        provideSearchers=args.reuseSearchers,
    );

    logging.info("Obtained %d hotspots" % len(hotspots));

    # Generate data, and either dump to disk, or call variants
    datagen = trainDataTools.data(
        hotspots,
        readSamplers,
        searcherFactory,
        args.ref,
        vcf=args.truth,
        bed=args.highconf,
        hotspotMethod=args.hotspotMode,
        searcherCollection=searcherCollection,
    );

    featureList = None;

    if args.highconf is not None:
        # if not args.test_labeling:
        #     dumpTrainingData(datagen, args.outputPrefix, hybrid=(len(bamfiles) > 1));
        # else:
        #     cache = ReferenceCache(database=args.ref);
        #     dump_vcfs_from_labels(datagen, args.outputPrefix + ".labeled", cache)
        ref = ReferenceCache(database=args.ref)
        dumpTrainingData(datagen, args.outputPrefix, hybrid=(len(bamfiles) > 1), dump_vcf=True, ref=ref, keep=args.keep_hdf5);
    else:
        assert(args.network is not None), "Provide DNN for variant calling";
        cache = ReferenceCache(database=args.ref);
        fhandle = open(args.outputPrefix + ".vcf", 'w');
        network = torch.load(args.network, map_location='cpu');
        network.eval();        

        if args.provideFeatures:
            network.providePredictions = True;
            featureList = [];
        else:
            featureList = None;

        for i, siteDict in enumerate(datagen):
            if siteDict is None:
                continue;

            if len(siteDict['alleles']) == 0:
                continue;

            record = vcfRecords(siteDict, network, cache, hybrid=(len(bamfiles) > 1));

            if record is not None:
                if type(record) is tuple:
                    record, featureDict = record;
                    fhandle.write(str(record) + '\n');
                    featureList.append(featureDict);
                else:
                    fhandle.write(str(record) + "\n");

            if (i + 1) % 100 == 0:
                logging.info("Completed %d sites" % (i + 1));

        fhandle.close();

    if featureList is not None:
        with open(args.outputPrefix + ".features", "wb") as whandle:
            pickle.dump(featureList, whandle);

    logging.info("Completed running the script");
