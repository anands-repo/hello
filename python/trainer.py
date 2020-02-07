"""
Classes for generating and manipulating training data for the new AlleleSearcher DNN
"""
from evaluator import Stepper, bedReader, genotype, modifyVcfRecord
from PileupContainer import adjustCigartuples
import ast
from AlleleSearcher import AlleleSearcher
from AlleleSearcherLite import AlleleSearcherLite
from ReferenceCache import ReferenceCache
from PileupContainerLite import PileupContainerLite
import vcf
from more_itertools import peekable
import sys
import argparse
import h5py
import logging
import libCallability
import pybedtools
import os
from multiprocessing import Pool
import torch
from AlleleSearcherDNN import Network, ConvolutionalNetwork, RecurrentNetwork, Flatten, sizeTensor, ScoringWrapper, GraphSearcherWrapper, GraphSearcher, ResidualBlock
import math
import importlib
import numpy as np
from ReadConvolver import ReadConvolverWrapper, ReadConvolverHybridWrapper
import tables
from MemmapData import MemmapperCompound

torch.set_num_threads(1);

SearcherClass = AlleleSearcher;

adjustCigartuplesFunction = adjustCigartuples;

SUBSAMPLE_RATES = None;


def samplefromMultiNomial(dict_):
    """
    Samples from a multinomial distribution defined by dict_

    :param dict_: dict
        A dictionary indicating, item: probability

    :return: object
        Key that has been sampled
    """
    keys, values = tuple(zip(*dict_.items()));
    sampledIndex = np.argmax(np.random.multinomial(1, values));
    return keys[sampledIndex];


def dummyFunction(*args):
    """
    Dummy function that does nothing
    """
    pass;


def modifyCandidate(candidate, newStart,  newStop, *args):
    """
    Modify a candidate to span new start and stop positions

    :param candidate: dict
        Candidate region

    :param newStart: int
        New start position

    :param newStop: int
        New stop position
    """
    newCandidate = dict(candidate);
    newCandidate['start'] = newStart;
    newCandidate['stop'] = newStop;
    return newCandidate;


def mergeRegions(regions):
    """
    Merge consecutive activity into a single one

    :param regions: list
        List of activity spots

    :return: list
        List of merged activity
    """
    cluster = [];
    merged = [];

    def boundary(cluster):
        return min([c[0] for c in cluster]), max([c[1] for c in cluster]);

    for r in regions:
        if len(cluster) == 0:
            cluster.append(r);
        else:
            start, stop = boundary(cluster);
            if (start <= r[0] <= stop) or (start <= r[1] <= stop):
                cluster.append(r);
            else:
                merged.append((start, stop));
                cluster = [r];

    if len(cluster) > 0:
        merged.append(boundar(cluster));

    return merged;


def evaluateFeatures(features, network, depthMultiplier=None):
    """
    Evaluate allele likelihoods given features

    :param features: dict
        Dictionary mapping alleles to features

    :param network: torch.nn.Module
        Network object using which to evaluate features

    :param depthMultiplier: torch.Tensor/tuple
        If depth normalization is desired, provide these here
    """
    if (type(network) is GraphSearcherWrapper) or (type(network) is ReadConvolverWrapper) or (type(network) is ReadConvolverHybridWrapper):
        logging.debug("Evaluating using GraphSearcherWrapper/ReadConvolverWrapper");
        if len(features) > 1:
            with torch.no_grad():
                if depthMultiplier is not None:
                    alleles, evals = network(features, depthMultiplier=depthMultiplier);
                else:
                    alleles, evals = network(features);
                evals = evals.cpu().data.numpy().flatten().tolist();
        else:
            alleles = list(features.keys());
            evals = [0];  # Currently not evaluating single allele sites
    else:
        alleles = [];
        evals = [];

        for allele in feature:
            alleles.append(allele);
            with torch.no_grad():
                evals.append(network(feature).cpu().data.numpy().flatten()[0]);

    return list(zip(evals, alleles));


def createJobSplitWrapper(args):
    """
    Wrapper for createJobSplit to be called through multiprocessing

    :param args: tuple
        Arguments for createJobSplit

    :return: str
        The command to be run
    """
    return createJobSplit(*args);


def createJobSplit(bedRegions, highconf, groundTruth, hotspots, prefix, args):
    """
    Creates a job split for parallel processing

    :param bedRegions: list
        List of bed regions which are part of the job split

    :param highconf: bed
        The high-confidence bed file

    :param groundTruth: str
        Ground-truth vcf name

    :param hotspots: list
        List of hotspots within the bedRegions

    :param prefix: str
        Prefix for temporary files (include temp directory path)

    :param args: ArgumentParser
        Original arguments passed to the script

    :return: str
        The command is returned
    """
    bedString = "";

    for i, (chromosome, start, stop) in enumerate(bedRegions):
        bedString += "\n" if i > 0 else "";
        bedString += "\t".join([chromosome, str(start), str(stop)]);

    subset = pybedtools.BedTool(bedString, from_string=True);
    vcfBed = pybedtools.BedTool(groundTruth);
    hcBed = pybedtools.BedTool(highconf);
    vcfName = os.path.abspath(prefix + ".vcf");
    hcName = os.path.abspath(prefix + ".bed");
    hotspotsName = os.path.abspath(prefix + ".txt");

    vcfBed.intersect(subset, header=True).saveas(vcfName);
    hcBed.intersect(subset).saveas(hcName);

    with open(hotspotsName, 'w') as fhandle:
        for item in hotspots:
            fhandle.write(str(item) + '\n');

    cmd = 'python %s' % (os.path.join(os.path.split(os.path.abspath(__file__))[0], "trainer.py"));
    cmd += " --bam %s" % args.bam;
    cmd += " --highconf %s" % hcName;
    cmd += " --ref %s" % args.ref;
    cmd += " --hotspots %s" % hotspotsName;
    cmd += " --groundTruth %s" % vcfName;
    cmd += " --readRate %s" % ",".join(map(str, args.readRate));
    cmd += " --pacbio" if args.pacbio else "";
    cmd += " --outputPrefix %s" % prefix;
    cmd += " --padlength %d" % args.padlength;
    cmd += " --featureLength %d" % args.featureLength;

    if args.subsampleRates is not None:
        cmd += " --subsampleRates %s" % args.subsampleRates;

    if args.network is not None:
        cmd += " --network %s" % args.network;
        cmd += " --config %s" % args.config;

    if args.discardHomozygous:
        cmd += " --discardHomozygous";

    if args.indelRealigned:
        cmd += " --indelRealigned";

    if args.useAdvanced:
        cmd += " --useAdvanced";

    if args.useLite:
        cmd += " --useLite";

    if args.useTables:
        cmd += " --useTables";

    if args.saveAsIntegrated:
        cmd += " --saveAsIntegrated";

    if args.useMapQ:
        cmd += " --useMapQ";

    if args.useOrientation:
        cmd += " --useOrientation";

    if args.useNpy:
        cmd += " --useNpy";

    return cmd;


def chromosomesInCluster(cluster):
    """
    Determine the chromosomes in a cluster of hotspot regions

    :param cluster: list
        List of hotspot regions

    :return: dict
        Chromosomes and first last positions within the chromosomes in the cluster
    """
    chromosomes = dict();

    for point in cluster:
        chromosome = point['chromosome'];
        position = point['position'];
        if chromosome in chromosomes:
            values = chromosomes[chromosome];
            chromosomes[chromosome] = (min(values[0], position), max(values[1], position + 1));
        else:
            chromosomes[chromosome] = (position, position + 1);

    return chromosomes;


def convertClusterToItems(cluster_):
    """
    Convert a cluster of points to items indicating a job-split

    :param cluster_: list
        List of points

    :return: tuple
        (cluster_, bedItems)
    """
    chromosomes = chromosomesInCluster(cluster_);
    bedItems = [];

    for key, value in chromosomes.items():
        bedItems.append((key, value[0] - 10, value[1] + 10));

    logging.debug("Creating cluster of %d points with bed region %s" % (len(cluster_), str(bedItems)));

    return list(cluster_), bedItems;  # Make a copy of cluster


def determineJobSplits(hotspots, minSeparation=256, minItemsPerBundle=1024):
    """
    Analyze hotspots to provide job-splits

    :param hotspots: str
        Hotspots filename

    :param minSeparation: int
        Minimum separation between any two hotspot points in two different files

    :param minItemsPerBundle: int
        Minimum number of hotspots to be included in a file

    :return: tuple
        ([hotspot0, hotspot1, ...], [bed0, bed1, ...])
    """
    clusters = [];
    bedRegions = [];

    with open(hotspots, 'r') as fhandle:
        cluster = [];

        for line in fhandle:
            point = ast.literal_eval(line);
            if len(cluster) >= minItemsPerBundle:
                if measureDistance(cluster[-1], point) >= minSeparation:
                    cluster_, bedItems = convertClusterToItems(cluster);
                    logging.debug(
                        "Pushing cluster items first, last = %s, %s" % (
                            str(cluster_[0]),
                            str(cluster_[-1])
                        )
                    );
                    clusters.append(cluster_);
                    logging.debug(
                        "Pusing bed entry = %s" % (
                            str(bedItems),
                        )
                    );
                    bedRegions.append(bedItems);
                    cluster = [];

            cluster.append(point);

        if len(cluster) > 0:
            cluster_, bedItems = convertClusterToItems(cluster);
            logging.debug(
                "Pusing cluster items first, last = %s, %s" % (
                    str(cluster_[0]),
                    str(cluster_[-1])
                )
            );
            logging.debug(
                "Pusing bed entry = %s" % (
                    str(bedItems),
                )
            );
            clusters.append(cluster_);
            bedRegions.append(bedItems);

    logging.debug("Returning %d cluster items, %d bed regions" % (len(clusters), len(bedRegions)));

    return tuple([clusters, bedRegions]);


def allelesInRecord(record):
    """
    Determines the alleles in a record

    :param record: dict
        Dictionary representing a record

    :return: set
        List of alleles in record
    """
    alleles = [record['ref']] + record['alt'];
    return set([alleles[i] for i in record['gt']]);


def measureDistance(pointA, pointB):
    """
    Determines the distance pointB - pointA

    :param pointA: dict
        Point A

    :param pointB: dict
        Point B

    :return: int
        Distance
    """
    if (pointA['chromosome'] != pointB['chromosome']):
        distance = float('inf');

    if ('position' in pointA) and ('position' in pointB):
        return pointB['position'] - pointA['position'];
    elif ('start' in pointB) and ('stop' in pointA):
        return pointB['start'] - pointA['stop'] + 1;
    else:
        raise ValueError("Bad arguments " + str(pointA) + ", " + str(pointB));


def isWithinDistance(pointA, pointB, distance=11):
    """
    Determines whether a second point is within a certain distance after the first point

    :param pointA: dict
        Point A for comparison (first point)

    :param pointB: dict
        Point B for comparison (second point)

    :param distance: int
        Distance to be compared

    :return: bool
        The status of the comparison
    """
    if 0 <= measureDistance(pointA, pointB) <= distance:
        return True;

    return False;


def groundTruthReader(vcfname):
    """
    Generator for reading ground-truth variants

    :param vcfname: str
        VCF filename

    :return: iter
        Returns an iterator
    """
    reader = vcf.Reader(open(vcfname, 'r'));

    for record in reader:
            gt0 = genotype(record);
            yield \
                {
                    'chromosome': record.CHROM,
                    'start': record.POS - 1,
                    'stop': record.POS - 1 + len(record.REF),
                    'ref': record.REF,
                    'alt': [str(s) for s in record.ALT],
                    'gt': gt0,
                };


def hotspotsReader(filename, distance=11):
    """
    Generator that provides active regions from a hotspot file

    :param filename: str
        Filename

    :param distance: int
        Distance separation between two hotspots for them to be part of the same
        active region

    :return: iter
        Iterator
    """
    with open(filename, 'r') as fhandle:
        cluster = [];

        for line in fhandle:
            point = ast.literal_eval(line);
            if len(cluster) == 0:
                cluster.append(point);
            else:
                if isWithinDistance(cluster[-1], point, distance):
                    cluster.append(point);
                else:
                    yield {
                        'chromosome': cluster[0]['chromosome'],
                        'start': cluster[0]['position'] - distance // 2,
                        'stop': cluster[-1]['position'] + distance // 2
                    };
                    cluster = [point];

        if len(cluster) != 0:
            yield {
                'chromosome': cluster[0]['chromosome'],
                'start': cluster[0]['position'] - distance // 2,
                'stop': cluster[-1]['position'] + distance // 2
            };


def candidateReader(ref, bamfile, hotspots=None, hotspotIterator=None, distance=11, pacbio=False, readRate=(1000, 30)):
    """
    Generator for determining candidate sites for variant calling

    :param ref: str
        ReferenceCache filename

    :param bamfile: str
        Bamfile location

    :param hotspots: str
        Filename for hotspots

    :param hotspotIterator: iter
        hotspot reading iterator (see hotspotsReader above)

    :param distance: int
        Distance parameter for instanciating hotspotReader

    :param pacbio: bool
        If these are for pacbio reads, then use this

    :param readRate: tuple
        (number of reads to retain, per active region size)

    :return: iter
        Iterator object
    """
    assert((
        (hotspots is not None) and (hotspotIterator is None) or
        (hotspots is None) and (hotspotIterator is not None)
    )), "Provide one of hotspots or hotspotIterator as argument";

    if hotspotIterator is None:
        hotspotIterator = hotspotsReader(hotspots, distance);

    cache = ReferenceCache(database=ref);

    for item in hotspotIterator:
        logging.debug("CandidateReader received %s" % str(item));

        # Always retain at least readRate[0] reads
        if item['stop'] - item['start'] > readRate[1]:
            numReadsToRetain = readRate[0] / readRate[1] * (item['stop'] - item['start']);
        else:
            numReadsToRetain = readRate[0];

        container = PileupContainerLite(
            bamfile, item['chromosome'], item['start'], item['stop'] - item['start'], clipReads=pacbio, maxNumReads=numReadsToRetain
        );
        adjustCigartuplesFunction(container, cache);
        # Pacbio attribution bug fixed on 190828
        # Note: indel realignment attribution is unnecessary, since it doesn't affect hotspot detection
        searcher = SearcherClass(container, item['start'], item['stop'], cache, pacbio=True);
        logging.debug(
            "-CANDIDATE READER- Obtained regions %s for start = %d, stop = %d" % (str(searcher.differingRegions), item['start'], item['stop'])
        );

        for region in searcher.differingRegions:
            yield {'chromosome': item['chromosome'], 'start': region[0], 'stop': region[1], 'reads': container};


class Trainer:
    """
    Generate training data for AlleleSearcher DNN
    """
    def __init__(
        self,
        highconf,
        hotspots,
        groundTruth,
        ref,
        bam,
        pacbio,
        readRate=(1000, 30),
        toleranceDistance=3,
        network=None,
        normalize=False,
        maxAlleleLength=60,
        padlength=0,
        discardHomozygous=False,
        indelRealigned=False,
        useAdvanced=False,
        useMapQ=False,
        useOrientation=False,
        useQEncoding=False,
        useColored=False,
        featureLength=100,
        subsampleRates=None,
    ):
        """
        :param highconf: str
            Name of high-confidence bed file

        :param hotspots: str
            Name of hotspots file

        :param groundTruth: str
            Name of ground-truth VCF file

        :param ref: str
            Reference cache file location

        :param bam: str
            Bam file location

        :param pacbio: bool
            Whether we are targeting pacbio reads or not

        :param readRate: tuple
            (Number of reads to retain, per active region size)

        :param toleranceDistance: int
            If a feature is received without an overlapping VCF record, or
            a VCF record is received without an overlapping feature, then the minimum distance
            between the last and the next features/VCF records is measured, and if it is small enough,
            the records are merged and treated as a single candidate location

        :param network: Network
            Provide this option when network must be used to evaluate top-two alleles and generate
            classifier examples

        :param normalize: bool
            Normalize features from allele searcher

        :param maxAlleleLength: int
            A setting for PacBio reads where we do not look at allele lengths over 60

        :param padlength: int
            Length to which a feature must be padded

        :param discardHomozygous: bool
            Discard clearly homozygous sites (for classifier training)

        :param indelRealigned: bool
            Input data has been indel realigned

        :param useAdvanced: bool
            Use advanced feature map creation

        :param useMapQ: bool
            Whether we plan to use MAPQ in the feature

        :param useOrientation: bool
           Whether we want to use read orientation in the feature map

        :param useQEncoding: bool
            Use quality score encoding not probability encoding

        :param useColored: bool
            Use DeepVariant-style colored features

        :param featureLength: int
            Length of the feature map

        :param subsampleRates: dict
            A dictionary of subsample rates to choose from
        """
        stream0 = peekable(bedReader(highconf));
        stream1 = peekable(candidateReader(ref, bam, hotspots=hotspots, pacbio=pacbio, readRate=readRate));
        stream2 = peekable(groundTruthReader(groundTruth));
        self.stepper = peekable(Stepper(stream0, stream1, stream2, streamToTerminate=1));  
                                                   # Terminate when no more active region are available
        self.missedCalls = [];
        self.bamfile = bam;
        self.pacbio = pacbio;
        self.readRate = readRate;
        self.ref = ReferenceCache(database=ref);
        self.toleranceDistance = toleranceDistance;
        self.network = network;
        self.normalize = normalize;
        self.maxAlleleLength = maxAlleleLength;
        self.padlength = padlength;
        self.discardHomozygous = discardHomozygous;
        self.indelRealigned = indelRealigned;
        self.useAdvanced = useAdvanced;
        self.useOrientation = useOrientation;
        self.useMapQ = useMapQ;
        self.useQEncoding = useQEncoding;
        self.useColored = useColored;
        self.featureLength = featureLength;
        self.subsampleRates = subsampleRates;

    def __iter__(self):
        return self;

    def __next__(self):
        candidate, record = next(self.stepper);
        nextItems = self.stepper.peek(None);

        logging.debug("Obtained candidate, record = %s, %s" % (str(candidate), str(record)));

        # If more than one vcf entry overlaps on the reference, discard it - it is complex, and I have decided not to deal with it
        if (type(record) is list) and (len(record) > 1):
            return None;

        # Received a record but no candidate
        if (candidate is None) and (record is not None) and (nextItems is not None):
            nextCandidate, nextRecord = nextItems;

            if (
                (nextCandidate is not None) and
                (nextRecord is None) and
                (measureDistance(record, nextCandidate) <= self.toleranceDistance)
            ):
                logging.debug("Discarding pair " + str(nextCandidate) + ", " + str(record));
                next(self.stepper);
                return None;

        # Received a candidate but no record
        if (record is None) and (nextItems is not None) and (candidate is not None):
            nextCandidate, nextRecord = nextItems;

            if (
                (nextRecord is not None) and
                (nextCandidate is None) and
                (measureDistance(candidate, nextRecord) <= self.toleranceDistance)
            ):
                logging.debug("Discarding pair " + str(nextRecord) + ", " + str(candidate));
                next(self.stepper);
                return None;

        # If record is received without candidates add to missed calls list and quit
        if (record is not None) and (candidate is None):
            self.missedCalls.append(record);
            logging.debug("Cannot call " + str(record));
            return None;

        logging.debug("Trainer received chr, start, stop = %s, %d, %d from hotspots" % (candidate['chromosome'], candidate['start'], candidate['stop']));

        if record is not None:
            logging.debug("Trainer received chr, start, stop = %s, %d, %d from records" % (record['chromosome'], record['start'], record['stop']));

        # Perform allele search
        if (record is not None):
            # start = min(record['start'], candidate['start']);
            # stop = max(record['stop'], candidate['stop']);
            # record = modifyVcfRecord(record, start, stop, self.ref);
            start = min(record['start'], candidate['start']);
            stop = max(record['stop'], candidate['stop']);

            # Note: PacBio correctly attributed here
            localSearcher = SearcherClass(
                candidate['reads'],
                start, stop, self.ref, pacbio=self.pacbio, indelRealigned=self.indelRealigned,
                useAdvanced=self.useAdvanced, useMapQ=self.useMapQ, useOrientation=self.useOrientation, useQEncoding=self.useQEncoding,
                useColored=self.useColored,
                featureLength=self.featureLength,
            );

            # Expand the region to include adjacent indels
            if (start < candidate['start']) or (stop > candidate['stop']):
                while True:
                    newRegion = localSearcher.expandRegion(start, stop);

                    if newRegion == (start, stop):
                        break;

                    start, stop = newRegion;

            # Expand candidates and records as necessary
            record = modifyVcfRecord(record, start, stop, self.ref);
            candidate = modifyCandidate(candidate, start, stop, self.ref); 
        else:
            start = candidate['start'];
            stop = candidate['stop'];
            # Create a dummy record
            self.ref.chrom = candidate['chromosome'];
            record = {
                'chromosome': candidate['chromosome'],
                'start': candidate['start'],
                'stop': candidate['stop'],
                'ref': ''.join(self.ref[start:stop]),
                'alt': ['N'],
                'gt': [0, 0],
            };

        chromosome = candidate['chromosome'];
        container = candidate['reads'];  # PileupContainerLite(self.bamfile, chromosome, start, stop - start, clipReads=self.pacbio, maxNumReads=self.readRate[0]);

        if self.subsampleRates is not None:
            rate = samplefromMultiNomial(self.subsampleRates);
            logging.debug("Subsampling reads to %f of input depth" % rate);
            container.subsample(rate);

        # Note: PacBio attribution bug fixed on 190828
        searcher = SearcherClass(
            container, start, stop, self.ref,
            indelRealigned=self.indelRealigned, pacbio=self.pacbio, useAdvanced=self.useAdvanced,
            useMapQ=self.useMapQ, useOrientation=self.useOrientation, useQEncoding=self.useQEncoding,
            useColored=self.useColored,
            featureLength=self.featureLength,
        );
        searcher.assemble(start, stop);

        # If network is provided, do not process sites with a single allele
        if (self.network is not None) and (len(searcher.allelesAtSite) < 2) and (self.discardHomozygous):
            logging.debug("Discarding homozygous site %s" % str(candidate));
            return None;

        allelesFromSearcher = searcher.allelesAtSite;

        # Check if alleles at the site form a superset over alleles in the ground-truth
        groundTruthAlleles = allelesInRecord(record);

        if len(groundTruthAlleles.difference(allelesFromSearcher)) != 0:
            logging.debug("Cannot call, because searcher cannot find relevant alleles " + str(record));
            self.missedCalls.append(record);
            return None;

        # Generate tensors for each allele and label them
        tensors = [];
        labels = [];
        alleles = [];
        scores = [];
        supportingReads = [];
        supportingReadsStrict = [];
        featureDict = dict();

        for allele in allelesFromSearcher:
            feature = searcher.computeFeatures(allele, self.normalize);
            label = 1 / len(groundTruthAlleles) if allele in groundTruthAlleles else 0;
            tensors.append(feature);
            featureDict[allele] = torch.Tensor(feature);

            labels.append(label);
            alleles.append(allele);
            supportingReads.append(searcher.numReadsSupportingAllele(allele));
            supportingReadsStrict.append(searcher.numReadsSupportingAlleleStrict(allele));

        if self.network is not None:
            evals, alleles_ = tuple(zip(*evaluateFeatures(featureDict, self.network)));
            evalDict = dict(zip(alleles_, evals));
            for allele in alleles:
                scores.append(evalDict[allele]);

            # Determine top-two alleles and decide whether they encompass the ground-truth
            forSorting = sorted(zip(scores, alleles), reverse=True)[:2];
            topTwo = set([f[1] for f in forSorting]);
            if len(groundTruthAlleles.difference(topTwo)) != 0:
                logging.debug("Cannot call, because network cannot find relevant alleles " + str(record));
                self.missedCalls.append(record);
                return None;

        siteLabel = 0 if len(groundTruthAlleles) == 1 else 1;

        if len(scores) > 0:
            logging.debug("Scores at site = %s, for alleles %s, site label is %d" % (str(scores), str(alleles), siteLabel));

        if (stop - start) > self.maxAlleleLength:
            return None;

        return chromosome, start, stop, alleles, tensors, labels, siteLabel, scores, supportingReads, supportingReadsStrict;


def addToHDF5FilePytables(fhandle, chromosome, start, stop, alleles, tensors, labels, siteLabel, scores, supportingReads, supportingReadsStrict):
    """
    Add contents to an hdf5 file with pytables interface

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
    """
    logging.debug("Writing chr, start, stop = %s, %d, %d to file" % (chromosome, start, stop));
    groupName = '_'.join(['location', chromosome, str(start), str(stop)]);  # 'location' is prefixed here to avoid "NaturalNameWarning" from pytables
    fhandle.create_group(fhandle.root, groupName);
    mainGroup = getattr(fhandle.root, groupName);

    for i, (allele, tensor, label) in enumerate(zip(alleles, tensors, labels)):
        fhandle.create_group(mainGroup, allele);
        alleleGroup = getattr(mainGroup, allele);
        fhandle.create_array(alleleGroup, 'label', [label]);
        fhandle.create_array(alleleGroup, 'feature', tensor);
        fhandle.create_array(alleleGroup, 'supportingReads', [supportingReads[i]]);
        fhandle.create_array(alleleGroup, 'supportingReadsStrict', [supportingReadsStrict[i]]);

        if len(scores) > 0:
            fhandle.create_array(alleleGroup, 'score', [scores[i]]);

        # Perform checks
        if np.logical_or.reduce(np.isnan(tensor.flatten())):
            print("-ERROR- Found tensor with NaN element at location %s" % groupName);

    fhandle.create_array(mainGroup, 'siteLabel', [siteLabel]);

    if siteLabel not in [0, 1]:
        print("-ERROR- Found site label that is neither one or zero at site %s, label = %d" % (groupName, siteLabel));


def addToHDF5File(fhandle, chromosome, start, stop, alleles, tensors, labels, siteLabel, scores, supportingReads, supportingReadsStrict):
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
    """
    logging.debug("Writing chr, start, stop = %s, %d, %d to file" % (chromosome, start, stop));
    groupName = '_'.join([chromosome, str(start), str(stop)]);  # We will maintain a flat hierarchy
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
        if len(scores) > 0:
            alleleGroup.create_dataset('score', shape=(1,), dtype='float32');
            alleleGroup['score'][:] = scores[i];

        # Perform checks
        if np.logical_or.reduce(np.isnan(tensor.flatten())):
            print("-ERROR- Found tensor with NaN element at location %s" % groupName);

    mainGroup.create_dataset('siteLabel', shape=(1,), dtype=int);
    mainGroup['siteLabel'][:] = siteLabel;

    if siteLabel not in [0, 1]:
        print("-ERROR- Found site label that is neither one or zero at site %s, label = %d" % (groupName, siteLabel));


def copy(inhandle, outhandle, location):
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
            dset = alleleOutGroup.create_dataset(
                attribute, dtype=alleleInGroup[attribute].dtype, shape=alleleInGroup[attribute].shape
            );
            dset[:] = alleleInGroup[attribute][:];


def genTrainingData(args):
    """
    Main function for generating training data

    :param args: argparse.ArgumentParser
        Arguments for the script
    """
    if args.network is not None:
        network = torch.load(args.network, map_location='cpu');
        network.eval();
    else:
        network = None;

    trainer = Trainer(
        args.highconf,
        args.hotspots,
        args.groundTruth,
        args.ref,
        args.bam,
        args.pacbio,
        readRate=args.readRate,
        network=network,
        normalize=args.normalize,
        padlength=args.padlength,
        discardHomozygous=args.discardHomozygous,
        indelRealigned=args.indelRealigned,
        useAdvanced=args.useAdvanced,
        useMapQ=args.useMapQ,
        useOrientation=args.useOrientation,
        useQEncoding=args.useQEncoding,
        useColored=args.useColored,
        featureLength=args.featureLength,
        subsampleRates=SUBSAMPLE_RATES,
    );

    if args.useTables:
        fhandle = tables.open_file(args.outputPrefix + ".table", 'w');
    else:
        fhandle = h5py.File(args.outputPrefix + '.hdf5', 'w');

    lastLocation = None;
    args_ = None;

    def main():
        for i, args_ in enumerate(iter(trainer)):
            if args_ is None:
                continue;
            addToHDF5FileFunctor(*([fhandle] + list(args_)));
            lastLocation = (args_[0], args_[1], args_[2]);

            if (((i + 1) % 100) == 0):
                logging.info("Completed %d locations" % (i + 1));

        fhandle.close();

        if args.saveAsIntegrated and (not args.useTables):
            logging.info("Converting to integrated format");
            dtypes = None;
            if args.useColored:
                dtypes = {'feature': 'uint8', 'label': 'float32'};
            MemmapperCompound(
                args.outputPrefix + ".hdf5",
                prefix=args.outputPrefix,
                keys=['feature', 'label'],
                concatenations=[False, True],
                hybrid=False,
                mode='integrated',
                dtypes=dtypes,
                memtype="memmap" if not args.useNpy else "npy",
            );
            # Copy to new hdf5 file and replace the old hdf5 file to save space
            logging.info("Releasing space ... will create a temporary file");
            inhandle = h5py.File(args.outputPrefix + ".hdf5", 'r');
            outhandle = h5py.File(args.outputPrefix + ".tmp.hdf5", 'w');
            for location in inhandle.keys():
                copy(inhandle, outhandle, location);
            inhandle.close();
            outhandle.close();
            os.rename(args.outputPrefix + ".tmp.hdf5", args.outputPrefix + ".hdf5");

    if args.debug or args.deepDebug:
        main();
    else:
        try:
            main();
        except Exception:
            logging.error("Error! Last processed location is %s" % (str(lastLocation)));


def splitIntoJobs(args):
    """
    Main function for generating training data

    :param args: argparse.ArgumentParser
        Arguments for the script
    """
    assert(args.temp is not None), "Provide temp directory for splitting into multiple jobs";

    args.temp = os.path.abspath(args.temp);

    if not os.path.exists(args.temp):
        os.makedirs(args.temp);
    else:
        if not os.path.isdir(args.temp):
            raise ValueError("%s is not a directory" % args.temp);

    logging.info("Determining split locations for jobs");

    jobSplits = determineJobSplits(args.hotspots, args.minSeparation, args.minNumLocationsPerJob);
    shellScriptName = os.path.join(args.temp, "jobs.sh")
    shellScript = open(shellScriptName, 'w');

    logging.info("Creating jobs");

    if args.numThreadsSplit <= 1:
        for i, items in enumerate(zip(*jobSplits)):
            cluster, bedItem = items;
            job = createJobSplit(bedItem, args.highconf, args.groundTruth, cluster, os.path.join(args.temp, "job%d" % i), args);
            shellScript.write(str(job) + '\n');
            logging.info("Completed creating %d jobs" % (i + 1));
    else:
        workers = Pool(args.numThreadsSplit);
        args = [(bedItem, args.highconf, args.groundTruth, cluster, os.path.join(args.temp, "job%d" % i), args) for i, (cluster, bedItem) in enumerate(zip(*jobSplits))];
        for j, job in enumerate(workers.imap_unordered(createJobSplitWrapper, args)):
            shellScript.write(str(job) + '\n');
            logging.info("Completed creating %d jobs" % (j + 1));

    shellScript.close();
    logging.info("Wrote jobs into file %s" % shellScriptName);


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate training examples for AlleleSearcher DNN");

    parser.add_argument(
        "--bam",
        help="Bam file",
        required=True,
    );

    parser.add_argument(
        "--highconf",
        help="High confidence bed region",
        required=True,
    );

    parser.add_argument(
        "--ref",
        help="Reference cache path",
        required=True,
    );

    parser.add_argument(
        "--hotspots",
        help="Hotspots file",
        required=True,
    );

    parser.add_argument(
        "--groundTruth",
        help="Ground-truth VCF",
        required=True,
    );

    parser.add_argument(
        "--pacbio",
        help="Whether this read set contains PacBio reads",
        default=False,
        action="store_true",
    );

    parser.add_argument(
        "--outputPrefix",
        help="Prefix of output file",
        required=True,
    );

    parser.add_argument(
        "--readRate",
        help="For downsampling: comma-separated Number of reads,reference length",
        default=None,
    );

    parser.add_argument(
        "--debug",
        help="Print debug information",
        default=False,
        action="store_true",
    );

    parser.add_argument(
        "--deepDebug",
        help="Print deep debug messages (including C++ section)",
        default=False,
        action="store_true",
    );

    parser.add_argument(
        "--splitIntoJobs",
        help="Split into multiple jobs for parallel processing",
        default=False,
        action="store_true",
    );

    parser.add_argument(
        "--minSeparation",
        help="Minimum separation between two hotspots when splitting into jobs",
        type=int,
        default=256,
    );

    parser.add_argument(
        "--minNumLocationsPerJob",
        help="Minimum number of locations per job when splitting",
        type=int,
        default=1024,
    );

    parser.add_argument(
        "--temp",
        help="Temp directory for splitting into multiple jobs",
    );

    parser.add_argument(
        "--numThreadsSplit",
        help="Number of threads for splitting jobs",
        default=1,
        type=int,
    );

    parser.add_argument(
        "--network",
        help="If network predictions for top-two alleles should be included, provide this",
        required=False,
    );

    parser.add_argument(
        "--normalize",
        help="Normalize feature maps from AlleleSearcher",
        default=False,
        action="store_true",
    );

    parser.add_argument(
        "--discardHomozygous",
        help="For classifier training, discard clearly homozygous sites",
        default=False,
        action="store_true",
    );

    parser.add_argument(
        "--padlength",
        help="If network expects padded input, length to which the input must be padded",
        default=0,
        type=int,
    );

    parser.add_argument(
        "--indelRealigned",
        help="(DEPRECATED; use --useLite instead) Input data has been indel realigned",
        default=False,
        action="store_true",
    );

    parser.add_argument(
        "--useLite",
        help="Use AlleleSearcherLite (experimental)",
        default=False,
        action="store_true",
    );

    parser.add_argument(
        "--doNotAdjustCigartuples",
        help="Turn off left-alignment of cigartuples",
        action="store_true",
        default=False,
    );

    parser.add_argument(
        "--useAdvanced",
        help="Use advanced feature map creation",
        default=False,
        action="store_true",
    );

    parser.add_argument(
        "--useMapQ",
        help="Use mapping quality encoding in feature maps",
        action="store_true",
        default=False,
    );

    parser.add_argument(
        "--useOrientation",
        help="Use read orientation encoding in feature maps",
        action="store_true",
        default=False,
    );

    parser.add_argument(
        "--useTables",
        help="Use PyTables instead of h5py",
        default=False,
        action="store_true",
    );

    parser.add_argument(
        "--saveAsIntegrated",
        help="Save as an integrated dataset",
        default=False,
        action="store_true",
    );

    parser.add_argument(
        "--useQEncoding",
        help="Use quality-score encoding (not probability encoding)",
        default=False,
        action="store_true",
    );

    parser.add_argument(
        "--useColored",
        help="Use DeepVariant style coloring instead of dimensionality to encode",
        default=False,
        action="store_true",
    );

    parser.add_argument(
        "--featureLength",
        help="Length of feature maps",
        default=100,
        type=int,
    );

    parser.add_argument(
        "--useNpy",
        help="Use numpy to save data instead of default memmap for --saveAsIntegrated option",
        default=False,
        action="store_true",
    );

    parser.add_argument(
        "--subsampleRates",
        help="A file containing a dictionary specifying subsample rates",
        default=None,
    );

    args = parser.parse_args();

    logging.basicConfig(level=(logging.DEBUG if args.debug else logging.INFO), format='%(asctime)-15s %(message)s');

    if args.subsampleRates is not None:
        logging.warning("trainer will not honor subsample rates");
        # with open(args.subsampleRates, 'r') as fhandle:
        #     rateText = ast.literal_eval(fhandle.getlines());
        #     SUBSAMPLE_RATES = rateText;

    if args.useTables:
        addToHDF5FileFunctor = addToHDF5FilePytables;
    else:
        addToHDF5FileFunctor = addToHDF5File;

    if args.deepDebug:
        args.debug = True;

    if args.useLite:
        SearcherClass = AlleleSearcherLite;

    if args.doNotAdjustCigartuples:
        adjustCigartuplesFunction = dummyFunction;

    libCallability.initLogging(args.deepDebug);

    if args.readRate is None:
        args.readRate = (1000, 30) if not args.pacbio else (100, 100);
    else:
        args.readRate = list(map(int, args.readRate.split(",")));

    if not args.splitIntoJobs:
        genTrainingData(args);
    else:
        splitIntoJobs(args);
