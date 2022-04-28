# © 2019 University of Illinois Board of Trustees.  All rights reserved
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
from collections import namedtuple
import itertools
import copy

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

    print(putativeHaplotypes)
    print(putativeCandidates)

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
        and the alleles (ref allele, replacement allele) involved as values
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
