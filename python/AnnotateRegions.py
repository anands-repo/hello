import torch
from PileupContainerLite import PileupContainerLite as PileupContainer
from ReferenceCache import ReferenceCache
from more_itertools import peekable
import ast
#from evalVcfFailures import chromosomeToNumber, genotype
import argparse
import ast
import logging
import math
import torch.utils.data
#from ContigAnalyzer import QLoss
import random
import numpy as np
from functools import reduce
import vcf
from collections import namedtuple
import itertools
import copy

torch.manual_seed(13);
random.seed(13);

TOPN = 8;


VariantRecord = namedtuple('VariantRecord', ['chromosome', 'position', 'ref', 'alt', 'gt']);


def allelesAtSite(record):
    """
    Obtain the alleles at a given site from a variant record at that site

    :param record: VariantRecord
        The variant record at the site

    :return: list
        List of alleles at the site
    """
    allAlleles = [record.ref] + record.alt;
    genotypes = list(set(record.gt));
    actuals = [str(allAlleles[g]) for g in genotypes];
    return actuals;


def supportsForContig(contig, kmers, k):
    """
    Determine the support profile of kmers

    :param contig: str
        The contig string

    :param kmers: dict
        A dictionar of kmer vs occurrence

    :param k: int
        K-mer length (redundant in some sense)

    :return: list
        Support profile
    """
    numKmers = len(contig)-k+1;
    supports = [];

    for i in range(numKmers):
        kmer = contig[i:i+k];
        if kmer in kmers:
            supports.append(kmers[kmer]);
        else:
            supports.append(0);

    return supports;


# def determineCyclicKmers(kmers):
#     """
#     Determine a set of cyclic kmers
# 
#     :param kmers: set
#         A set of kmers
# 
#     :return: set
#         A set of cyclic kmers
#     """
#     cyclic = set(libCallability.searchRepeats(list(kmers)));
# 
#     return cyclic;

def determineCyclicKmers(kmers):
    raise NotImplementedError;


def getKmersList(read, k):
    numKmers = len(read) - k + 1;
    kmers = [];

    for i in range(numKmers):
        kmers.append(read[i: i + k]);

    return kmers;


def getKmersInReadSet(reads, k):
    return set(reduce(lambda x, y : x + y, [getKmersList(read, k) for read in reads], []));


def getKmersInReadList(reads, k):
    return reduce(lambda x, y : x + y, [getKmersList(read, k) for read in reads], []);


def labelSites(segment, candidates, truthset, left, maxRecords=10):
    """
    Label a set of candidate variants based on the truth-set

    :param segment: str
        Reference segment in the region

    :param candidates: list
        List of VcfRecord objects representing candidates at site

    :param truthset: list
        List of VcfRecord objects representing truthset at site

    :param left: int
        Left-most position of segment within the reference chromosome

    :param maxRecords: int
        Maximum number of records entertained at a time
    """
    if (len(candidates) > maxRecords):
        raise ValueError("Intractable computation with more than %d records" % maxRecords);

    putativeHaplotypes = createGroundTruthHaplotypes(segment, truthset, left);
    putativeCandidates = createHaplotypeCandidates(segment, candidates, left);
    alleleConfigurations = [];
    successful = False;

    for putativeHaplotypePair in putativeHaplotypes:
        haplotypeCoverage = {haplotype: False for haplotype in putativeHaplotypePair};

        passedHaplotypes = set();
        alleleConfigurations = [];

        for candidate in putativeCandidates:
            candidateHaplotype = candidate;
            candidateAlleles = putativeCandidates[candidate];

            if (candidateHaplotype in haplotypeCoverage) and (candidateHaplotype not in passedHaplotypes):
                haplotypeCoverage[candidateHaplotype] = True;
                passedHaplotypes.add(candidateHaplotype);
                alleleConfigurations.append([c[1] for c in candidateAlleles]);

            if all(haplotypeCoverage.values()):
                successful = True;
                break;

        if successful:
            break;

    if successful:
        # Create two sets of alleles if only one set of alleles is present
        if len(alleleConfigurations) == 1:
            alleleConfigurations = 2 * alleleConfigurations;

        # Convert alleleConfigurations into haplotype pairs - these are not phased alleles
        return True, [set((x, y)) for x, y in zip(*alleleConfigurations)];

    return False, [];


def createGroundTruthHaplotypes(segment, records, left):
    """
    Creates a set of putative ground-truth records in a region

    :param segment: str
        The reference segment

    :param records: list
        Sorted VcfRecord objects

    :param left: int
        The left-most position of segment (absolute value)

    :return: list
        List of 2-tuples, or 1-tuple with all possible haplotype string
        pairings
    """
    if len(records) == 0:
        return [(segment, )];

    genotypeIndices = [list(range(len(set(record.gt)))) for record in records];

    haplotypePairs = [];

    for genotypeCounter in itertools.product(*genotypeIndices):
        haplotypePrefix0 = str(segment[0: records[0].position - left]);
        haplotypePrefix1 = str(segment[0: records[0].position - left]);

        for i, (variant, index) in enumerate(zip(records, genotypeCounter)):
            alleles = allelesAtSite(variant);

            if len(alleles) == 2:
                allele0 = alleles[index];
                allele1 = alleles[(index + 1) % 2];
            else:
                allele0 = alleles[index];
                allele1 = allele0;

            haplotypePrefix0 += allele0;
            haplotypePrefix1 += allele1;

            if i < len(records) - 1:
                nextVar = records[i + 1];
                suffix = segment[variant.position + len(variant.ref) - left: nextVar.position - left];
            else:
                suffix = segment[variant.position + len(variant.ref) - left:];

            haplotypePrefix0 += suffix;
            haplotypePrefix1 += suffix;

        if haplotypePrefix0 != haplotypePrefix1:
            haplotypePairs.append((haplotypePrefix0, haplotypePrefix1));
        else:
            haplotypePairs.append((haplotypePrefix0, ));

    return haplotypePairs;


def createHaplotypeCandidates(segment, records, left):
    """
    Create haplotype candidates from segment

    :param segment: str
        The reference segment

    :param records: list
        VcfRecord objects

    :param left: int
        The left-most position of the segment

    :return: dict
        Haplotypes or sub-haplotypes with haplotypes as strings
        and the alleles involved as values
    """
    # Handle absence of any records in the region:
    if len(records) == 0:
        return {segment: ''};

    record = records[0];

    if len(records) > 1:
        nextRecords = records[1:];
        nextLeft = record.position + len(record.ref);
        nextSubsegment = segment[nextLeft - left:];
        haplotypes = createHaplotypeCandidates(nextSubsegment, nextRecords, nextLeft);
    else:
        haplotypes = {'': []};

    subsegment = segment[:record.position - left];
    alleles = allelesAtSite(record);
    ref = record.ref;
    newHaplotypes = dict();

    for allele in alleles:
        for key in haplotypes:
            newKey = subsegment + allele + key;
            oldValue = haplotypes[key];
            newValue = [(ref, allele)] + oldValue;

            # If there are no more records to be processed, include the rest of the sequence here as well
            if len(records) == 1:
                newKey += segment[record.position - left + len(record.ref):];

            newHaplotypes[newKey] = newValue;

    return newHaplotypes;


def validateHaplotypes(
        chromosome,
        left,
        right,
        segment,
        records,
        bamfile,
        minKmerLength=25,
        maxKmerLength=75,
        returnFailedHaplotypes=False,
        pacbio=False,
):
    """
    Validate the haplotypes in a region when multiple haplotypes are present

    :param left: int
        The left position of the region

    :param right: int
        The right position of the region

    :param segment: str
        The reference segment in the region (from left -> right, right exclusive)

    :param records: list
        The list of records in the region

    :param bamfile: str
        The bam file to use

    :param minKmerLength: int
        The minimum k-mer length

    :param maxKmerLength: int
        The maximum k-mer length

    :param returnFailedHaplotypes: bool
        Return a list of failed haplotypes as well

    :param pacbio: bool
        Pacbio dataset being used

    :return: list
        The list of validated haplotypes in the region
    """
    container = PileupContainer(bamfile, chromosome, left, right - left, clipReads=pacbio);
    haplotypeCandidates = createHaplotypeCandidates(segment, records, left);

    # If there are only two or fewer haplotype candidates in the region, things are unambiguous
    if len(haplotypeCandidates) <= 2:
        logging.info("Validated haplotypes %s"%(str(haplotypeCandidates)));

        tooShort = False;

        for key in haplotypeCandidates:
            if len(key) < minKmerLength:
                tooShort = True;
                break;

        if not tooShort:
            if returnFailedHaplotypes:
                return haplotypeCandidates, set(), container;
            else:
                return haplotypeCandidates, container;
        else:
            logging.info("Haplotype(s) is(are) too short. Skipping...");
            return None;

    # If there are more, filter the correct ones based on k-mer support
    # The k-mers used must be non-cyclic, and cover a distance equaling
    # the distance between (including) any two consecutive haplotypes

    # Find the smallest k-mer length at which the graph is not cyclic
    reads = [p.alignment.query_sequence for p in container.pileupreads];
    nonCyclicK = -1;

    for k in range(minKmerLength, maxKmerLength):
        kmers = getKmersInReadSet(reads, k);
        cyclics = determineCyclicKmers(kmers);
        if len(cyclics) == 0:
            nonCyclicK = k;
            break;

    if nonCyclicK < 0:
        logging.info("Couldn't find a non-cyclic k-mer within specifications, couldn't validate");
        return None;

    if len(records) > 1:
        maxPairDistance = max([
            records[i+1].position + max([len(allele) for allele in allelesAtSite(records[i+1])]) - records[i].position
            for i in range(len(records)-1)
        ]);
    else:
        raise NotImplementedError("Impossible condition happened!");

    validateKmerLength = max(nonCyclicK, maxPairDistance);
    kmerList = getKmersInReadList(reads, validateKmerLength);
    kmerDict = dict();

    if validateKmerLength >= len(segment):
        logging.info("k-mer length required to validate is too long");
        return None;

    for kmer in kmerList:
        if kmer not in kmerDict:
            kmerDict[kmer] = 1;
        else:
            kmerDict[kmer] += 1;

    supportVectors = [];

    for haplotype in haplotypeCandidates:
        support = supportsForContig(haplotype, kmerDict, validateKmerLength);

        # Somehow, short haplotypes are still slipping through, so we do not entertain these cases
        if len(support) == 0:
            return None;

        supportVectors.append(support);
        logging.debug("Haplotype %s has supports %s with alleles %s"%(str(haplotype), str(support), str(haplotypeCandidates[haplotype])));

    minSupportForHaplotypes = [min(s) for s in supportVectors];
    supportDict = dict(zip(haplotypeCandidates, minSupportForHaplotypes));

    # Find top two haplotypes defined as haplotypes with
    # the highest minimum support
    topHaplotype, topScore  = None, -1;
    secondHaplotype, secondScore = None, -1;

    # If no high-scoring sequence is found return none
    if (topScore == 0) or (secondScore == 0):
        return None;

    for key, value in supportDict.items():
        if value > topScore:
            # Move top to second
            secondScore = topScore;
            secondHaplotype = topHaplotype;

            # Move new to top
            topScore = value;
            topHaplotype = key;
        else:
            if value > secondScore:
                # Move new to second
                secondScore = value;
                secondHaplotype = key;

    # Check whether these are unique values
    numOccurrencesTop = 0;
    numOccurrencesSecond = 0;

    del supportDict[topHaplotype];
    del supportDict[secondHaplotype];

    for item in supportDict:
        if supportDict[item] == topScore:
            numOccurrencesTop += 1;

        if supportDict[item] == secondScore:
            numOccurrencesSecond += 1;

    if (numOccurrencesTop > 0) or (numOccurrencesSecond > 0):
        logging.info("Couldn't find unique top two contigs, couldn't validate");
        return None;
    else:
        # Validate that the top two haplotypes also cover all alleles in all records
        candidates0 = haplotypeCandidates[topHaplotype];
        candidates1 = haplotypeCandidates[secondHaplotype];
        covers = True;

        for r, c0, c1 in zip(records, candidates0, candidates1):
            # allelesFromCandidates = set([c0] + [c1]);
            allelesFromCandidates = set([c0[1]] + [c1[1]]); # c0, c1 is of format (ref, allele)
            allelesFromRecord = set(allelesAtSite(r));
            if len(allelesFromRecord.difference(allelesFromCandidates)) > 0:
                covers = False;
                break;

        if covers:
            passedCandidates = {topHaplotype: candidates0, secondHaplotype: candidates1};
            logging.info("Validated haplotypes %s"%(str(passedCandidates)));

            # Find haplotypes that didn't pass
            failedContigs = set(supportDict.keys()).difference(set(passedCandidates.keys()));

            if returnFailedHaplotypes:
                return passedCandidates, failedContigs, container;
            else:
                return passedCandidates, container;
        else:
            logging.info("Top two contigs do not cover all alleles, couldn't validate");
            return None;


