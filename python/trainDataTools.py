# © 2019 University of Illinois Board of Trustees.  All rights reserved
"""
Classes for generating and manipulating training data for the new AlleleSearcher DNN
"""
from copy import deepcopy
import ast
# from ReferenceCache import ReferenceCache
from PySamFastaWrapper import PySamFastaWrapper as ReferenceCache
from PileupContainerLite import PileupContainerLite
import vcf
import sys
import argparse
import h5py
import logging
import pybedtools
import os
from multiprocessing import Pool
import math
import numpy as np
import AnnotateRegions
import os
import PileupDataTools
import collections
import intervaltree
from timeit import default_timer as timer
import labeler
from labeler import RegionTooLongException

try:
    profile
except Exception:
    def profile(_):
        return _;

DATAGEN_TIME = 0;
TENSOR_TIME = 0;
INTERNAL_TIME0 = 0;
GUARD_BAND = 3
STRICT_INTERSECTION = False
MAX_ITEMS_PER_GROUP = 8
HYBRID_TRUTH_EVAL = False


def genotype(record, index=0):
    """
    Parse genotype in tuple form
    """
    separator = None;

    if '|' in record.samples[index]['GT']:
        separator = '|';
    elif '/' in record.samples[index]['GT']:
        separator = '/';

    if separator is None:
        return None;

    gt = list(map(int, record.samples[index]['GT'].split(separator)));

    return gt;


def checkIntersection(hotspot, bedInterval):
    """
    Check whether a hotspot is strictly contained within a bedInterval

    :param hotspot: tuple
        Hotspot tuple

    :param bedInterval: intervaltree.IntervalTree
        High confidence intervals
    """
    bedItems = bedInterval[hotspot[0]: hotspot[1]]

    for item in bedItems:
        # Check whether any item completely contains the given hotspot
        if STRICT_INTERSECTION:
            if item[0] <= hotspot[0] < hotspot[1] <= item[1]:
                return True
        else:
            return True

    return False


def checkIntersectionRecord(record, bedInterval):
    """
    Check whether a record is strictly contained within a bedInterval

    :param hotspot: tuple
        Hotspot tuple

    :param bedInterval: intervaltree.IntervalTree
        High-confidence intervals within the given chromosome
    """
    item = (record.position, record.position + len(record.ref));
    return checkIntersection(item, bedInterval);


def bedReader(bedfile):
    bedRegions = collections.defaultdict(intervaltree.IntervalTree);

    with open(bedfile, 'r') as fhandle:
        for line in fhandle:
            entries = line.split();
            chr_, start, stop = entries[0], int(entries[1]), int(entries[2]);
            bedRegions[chr_].addi(start - GUARD_BAND, stop + GUARD_BAND, None);

    for chromosome in bedRegions:
        bedRegions[chromosome].merge_overlaps()

    return bedRegions;


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

    cmd = 'python %s' % (os.path.join(os.path.split(os.path.abspath(__file__))[0], "trainData.py"));
    cmd += " --bam %s" % args.bam;
    cmd += " --highconf %s" % hcName;
    cmd += " --ref %s" % args.ref;
    cmd += " --hotspots %s" % hotspotsName;
    cmd += " --groundTruth %s" % vcfName;
    cmd += " --pacbio" if args.pacbio else "";
    cmd += " --outputPrefix %s" % prefix;
    cmd += " --featureLength %d" % args.featureLength;
    cmd += " --hotspotMode %s" % args.hotspotMode;

    if args.bam2 is not None:
        cmd += " --bam2 %s" % args.bam2;

    if args.chrPrefixBam1 != "":
        cmd += " --chrPrefixBam1 %s" % args.chrPrefixBam1;

    if args.chrPrefixBam2 != "":
        cmd += " --chrPrefixBam1 %s" % args.chrPrefixBam2;

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


def groundTruthReader(vcfname):
    """
    Generator for reading ground-truth variants

    :param vcfname: str
        VCF filename

    :return: iter
        Returns an iterator
    """
    reader = vcf.Reader(open(vcfname, 'r'));
    truthSet = collections.defaultdict(intervaltree.IntervalTree);

    for record in reader:
        all_alleles = [record.REF]

        for alt in record.ALT:
            alt = str(alt)
            all_alleles.append(alt)

        genotypes = genotype(record)
        true_alleles = [all_alleles[i] for i in genotypes]
        sorted_alts = sorted(all_alleles[1:])
        sorted_all_alleles = [record.REF] + sorted_alts
        genotypes = [
            i for i, a in enumerate(sorted_all_alleles) if a in true_alleles
        ]
        if len(genotypes) == 1:
            genotypes = 2 * genotypes
        record_ = AnnotateRegions.VariantRecord(
            chromosome=record.CHROM,
            position=record.POS - 1,
            ref=record.REF,
            alt=sorted_alts,
            gt=genotypes,
        )

        truthSet[record.CHROM].addi(
            record.POS - 1,
            record.POS - 1 + len(record.REF),
            record_
        );

    return truthSet;


def createRecord(chromosome, position, refAllele, allelesAtSite):
    """
    Creates a candidate record

    :param chromosome: str
        Chromosome

    :param position: int
        Position in the chromosome

    :param refAllele: str
        Reference allele

    :param allelesAtSite: list
        List of alleles at the given site

    :return AnnotateRegions.VariantRecord
        VariantRecord object
    """
    allelesNoRef = deepcopy(allelesAtSite);

    if len(refAllele) == 0:
        raise ValueError("Obtained empty reference")

    if refAllele in allelesNoRef:
        allelesNoRef.remove(refAllele);

    gts = list(range(len(allelesAtSite))) if refAllele in allelesAtSite else [i + 1 for i in range(len(allelesAtSite))];

    candidateRecord = AnnotateRegions.VariantRecord(
        chromosome=chromosome,
        position=position,
        ref=refAllele,
        alt=allelesNoRef,
        gt=gts
    );

    return candidateRecord;


def findAlleleIndex(record, allele):
    """
    Find the index of an allele in a VCF record
    """
    alleles = [record.ref] + record.alt;
    if allele not in alleles:
        return -1;
    else:
        return alleles.index(allele);


def editRecord(record, dict_):
    """
    Edit a vcf-record-like object

    :param record: AnnotateRegion.VariantRecord
        The record to be edited

    :param dict_: dict
        Dictionary containing keys which need to be edited
    """
    recordToDict = {
        'chromosome': record.chromosome,
        'position': record.position,
        'ref': record.ref,
        'alt': record.alt,
        'gt': record.gt,
    };

    for key, value in dict_.items():
        recordToDict[key] = value;

    return AnnotateRegions.VariantRecord(**recordToDict);


def clusterLocations(locations, distance=PileupDataTools.MIN_DISTANCE, maxAlleleLength=80):
    """
    Similar to what hotspot reading does, this clusters a list of locations and gives them out

    :param locations: list
        List of locations (intervaltree.Interval)

    :param distance: int
        Distance for clustering candidates

    :param maxAlleleLength: int
        Maximum length of allele

    :return: iter
        Iterator over the clustered locations
    """
    cluster = [];

    for location in locations:
        if location[1] - location[0] > maxAlleleLength:
            # In this case, cleave the cluster here
            if len(cluster) > 0:
                yield cluster
                cluster = []
                continue

        if len(cluster) == 0:
            cluster.append(location);
        else:
            if (location[0] - cluster[-1][1] < distance) and (len(cluster) < MAX_ITEMS_PER_GROUP):
                cluster.append(location);
            else:
                yield cluster;
                cluster = [location];

    if len(cluster) > 0:
        yield cluster;
        cluster = [];


def split_clusters(cluster, reference):
    """
    Split clusters (again) so that the number of items in the cluster is less than MAX items

    :param cluster: list
        List of cluster tuples

    :param reference: PySamFastaWrapper
        Reference object

    :return: tuple
        List of clusters and reference segments
    """
    cluster = list(cluster)

    if len(cluster) <= MAX_ITEMS_PER_GROUP:
        ref_start = cluster[0][0] - PileupDataTools.MIN_DISTANCE // 2
        ref_stop = cluster[-1][-1] + PileupDataTools.MIN_DISTANCE // 2
        yield cluster, ''.join(reference[ref_start: ref_stop]), ref_start
    else:
        last_cluster = None
        indices = list(range(0, len(cluster), MAX_ITEMS_PER_GROUP))

        for i in range(len(indices)):
            index = indices[i]
            next_index = indices[i + 1] if i + 1 < len(indices) else -1
            current_cluster = cluster[index: index + MAX_ITEMS_PER_GROUP]
            next_cluster = cluster[next_index: next_index + MAX_ITEMS_PER_GROUP] if next_index >= 0 else None
            ref_start = max(
                current_cluster[0][0] - PileupDataTools.MIN_DISTANCE // 2,
                last_cluster[-1][-1] if last_cluster else -float('inf')
            )
            ref_stop = min(
                current_cluster[-1][-1] + PileupDataTools.MIN_DISTANCE // 2,
                next_cluster[0][0] if next_cluster else float('inf')
            )
            yield current_cluster, ''.join(reference[ref_start: ref_stop]), ref_start
            last_cluster = current_cluster


def get_labeled_candidates(
    chromosome,
    reference,
    searcher,
    cluster,
    truths=None,
    highconf=None,
    hotspotMethod="BOTH",
    maxAlleleLength=80,
):
    """
    A new version of the function below to use the new version of assembler

    :param chromosome: str
        Chromosome

    :param reference: PySamFastaWrapper
        Reference object

    :param searcher: AlleleSearcherLite
        AlleleSearcher object

    :param cluster: Noop
        This is no longer considered

    :param truths: collections.defaultdict
        Ground truth variants

    :param highconf: collections.defaultdict
        High confidence labels

    :param hotspotMethod: str
        Method to detect hotspots

    :param maxAlleleLength: int
        Maximum allele size
    """
    execStart = timer();
    global INTERNAL_TIME0;

    candidateRecords = [];
    candidateRecordsForTruthing = [];

    searcher.assemble_region()
    cluster = searcher.cluster

    results = []

    # It's possible that post-assembly the clusters are not viable any more
    if len(cluster) == 0:
        return results
    
    for cluster, segment, start in split_clusters(cluster, reference):
        stop = start + len(segment)

        for spot in cluster:
            allelesAtSpot = [];
            allelesAtSpotForTruthing = [];
            refAllele = segment[spot[0] - start: spot[1] - start];

            logging.debug(
                "Collecting reference allele for locations %d -> %d with segment start %d, end %d" % (
                    spot[0], spot[1], start, start + len(segment)
                )
            )

            if highconf is not None:
                if not checkIntersection(spot, highconf[chromosome]):
                    continue;

            candidates = searcher.determineAllelesInRegion(spot[0], spot[1]);
            allelesAtSpot += candidates;
            allelesAtSpot = list(set(allelesAtSpot));

            candidateRecords.append(
                createRecord(
                    chromosome=chromosome, position=spot[0], refAllele=refAllele, allelesAtSite=allelesAtSpot,
                )
            );

        if len(candidateRecords) == 0:
            continue

        candidateRecords = sorted(candidateRecords, key=lambda x: x.position);

        if truths is not None:
            assert(highconf is not None), "Provide highconf bed if truth is provided";

            # To determine ground-truths actually perform assembly for searcher0 and create candidates
            # We perform assembly in order to use only alleles that have non-zero support, rather than
            # alleles that have no support. Sometimes it is possible that labels are created with alleles
            # with no support (especially in STR regions).
            candidateRecordsForTruthing = [];
            # searcher = searchers[0];

            for spot in cluster:
                if not checkIntersection(spot, highconf[chromosome]):
                    continue;

                logging.debug("Performing assembly in %d -> %d" % (spot[0], spot[1]))
                searcher.assemble(spot[0], spot[1]);
                logging.debug("Completed assembly in %d -> %d" % (spot[0], spot[1]))
                refAllele = segment[spot[0] - start: spot[1] - start];

                # Only obtain allele list that has support from one sequencing technology
                allelesToTruthFrom = [];

                for allele in searcher.allelesAtSite:
                    if searcher.hybrid:
                        if HYBRID_TRUTH_EVAL or searcher.numReadsSupportingAlleleStrict(allele, 0) > 0:
                            allelesToTruthFrom.append(allele);
                    else:
                        allelesToTruthFrom.append(allele);

                candidateRecordsForTruthing.append(
                    createRecord(
                        chromosome=chromosome,
                        position=spot[0],
                        refAllele=refAllele,
                        allelesAtSite=sorted(list(set(allelesToTruthFrom)))
                    )
                );

            # Note: It seems ground-truth vcf has records outside of high-confidence regions
            groundTruth = [
                item.data for item in truths[chromosome][start: stop] \
                        if checkIntersectionRecord(item.data, highconf[chromosome])
            ];
            sortedTruths = sorted(groundTruth, key=lambda x: x.position);

            try:
                logging.debug("Candidate records %s" % str(candidateRecordsForTruthing))
                logging.debug("Ground truth records %s" % str(sortedTruths))
                logging.debug("Reference segment %s at %d" % (segment, start))
                lmachine = labeler.Labeler(sortedTruths, segment, start)
                flag, truthAlleles = lmachine(candidateRecordsForTruthing)
                logging.debug("Truths = %s" % (str(truthAlleles)));
            except RegionTooLongException:
            # except ValueError:
                logging.info("Region %s: %d - %d is too long" % (chromosome, start, stop));
                INTERNAL_TIME0 += (timer() - execStart);
                return None;

            if not flag:
                candidateRecords = [
                    editRecord(r, {'gt': [-1, -1]}) for r in candidateRecords
                ];
            else:
                # Mark the ground-truth alleles in each record
                substituteRecords = [];

                for r, t in zip(candidateRecords, truthAlleles):
                    gt = [findAlleleIndex(r, _) for _ in t];
                    assert(len(gt) >= 1), "At least one ground-truth allele should be found";
                    gt = gt * 2 if len(gt) == 1 else gt;
                    substituteRecords.append(
                        editRecord(r, {'gt': gt})
                    );

                candidateRecords = substituteRecords;

        results.extend(candidateRecords)

    INTERNAL_TIME0 += (timer() - execStart);
    return results


@profile
def getLabeledCandidates(
    chromosome,
    start,
    stop,
    segment,
    searcher,
    cluster,
    truths=None,
    highconf=None,
    hotspotMethod="BOTH",
    maxAlleleLength=80,
):
    """
    Create candidate vcf records for a set of spots, and label them if necessary

    :param chromosome: str
        Chromosomal region for which we are creating data

    :param start: int
        Start of the region (includes buffer length)

    :param stop: int
        Stop-point of the region (includes buffer length)

    :param segment: str
        Reference segment from start -> stop

    :param searcher: list
        List of AlleleSearcherLite objects

    :param cluster: list
        List of hotspot regions in a cluster

    :param truths: collections.defaultdict
        Ground-truth in the region

    :param highconf: collections.defaultdict
        Highconfidence bed regions

    :param hotspotMethod: str
        Whether to use BAM1 or both BAMs to construct hotspots

    :param maxAlleleLength: int
        Maximum allele length
    """
    execStart = timer();
    global INTERNAL_TIME0;

    candidateRecords = [];
    candidateRecordsForTruthing = [];

    for spot in cluster:
        allelesAtSpot = [];
        allelesAtSpotForTruthing = [];
        refAllele = segment[spot[0] - start: spot[1] - start];

        if highconf is not None:
            if not checkIntersection(spot, highconf[chromosome]):
                continue;

        candidates = searcher.determineAllelesInRegion(spot[0], spot[1]);
        allelesAtSpot += candidates;
        allelesAtSpot = list(set(allelesAtSpot));

        candidateRecords.append(
            createRecord(
                chromosome=chromosome, position=spot[0], refAllele=refAllele, allelesAtSite=allelesAtSpot,
            )
        );

    candidateRecords = sorted(candidateRecords, key=lambda x: x.position);

    if truths is not None:
        assert(highconf is not None), "Provide highconf bed if truth is provided";

        # To determine ground-truths actually perform assembly for searcher0 and create candidates
        # We perform assembly in order to use only alleles that have non-zero support, rather than
        # alleles that have no support. Sometimes it is possible that labels are created with alleles
        # with no support (especially in STR regions).
        candidateRecordsForTruthing = [];
        # searcher = searchers[0];

        for spot in cluster:
            if not checkIntersection(spot, highconf[chromosome]):
                continue;

            searcher.assemble(spot[0], spot[1]);
            refAllele = segment[spot[0] - start: spot[1] - start];

            # Only obtain allele list that has support from one sequencing technology
            allelesToTruthFrom = [];

            for allele in searcher.allelesAtSite:
                if searcher.hybrid:
                    if searcher.numReadsSupportingAlleleStrict(allele, 0) > 0:
                        allelesToTruthFrom.append(allele);
                else:
                    allelesToTruthFrom.append(allele);

            candidateRecordsForTruthing.append(
                createRecord(
                    chromosome=chromosome,
                    position=spot[0],
                    refAllele=refAllele,
                    allelesAtSite=list(set(allelesToTruthFrom))
                )
            );

        # Note: It seems ground-truth vcf has records outside of high-confidence regions
        groundTruth = [
            item.data for item in truths[chromosome][start: stop] \
                    if checkIntersectionRecord(item.data, highconf[chromosome])
        ];
        sortedTruths = sorted(groundTruth, key=lambda x: x.position);

        try:
            logging.debug(
                "Calculating truths %s, %s, %s, %d" % (
                    str(candidateRecordsForTruthing), str(sortedTruths), segment, start
                )
            );
            flag, truthAlleles = AnnotateRegions.labelSites(
                segment,
                candidateRecordsForTruthing,
                sortedTruths,
                left=start,
            );
            logging.debug("Truths = %s" % (str(truthAlleles)));
        except ValueError:
            logging.info("Region %s: %d - %d is too long" % (chromosome, start, stop));
            INTERNAL_TIME0 += (timer() - execStart);
            return None;

        if not flag:
            candidateRecords = [
                editRecord(r, {'gt': [-1, -1]}) for r in candidateRecords
            ];
        else:
            # Mark the ground-truth alleles in each record
            substituteRecords = [];

            for r, t in zip(candidateRecords, truthAlleles):
                gt = [findAlleleIndex(r, _) for _ in t];
                assert(len(gt) >= 1), "At least one ground-truth allele should be found";
                gt = gt * 2 if len(gt) == 1 else gt;
                substituteRecords.append(
                    editRecord(r, {'gt': gt})
                );

            candidateRecords = substituteRecords;

    INTERNAL_TIME0 += (timer() - execStart);
    return candidateRecords;


def createTensors(records, searcher, maxAlleleLength=80, hotspotMethod="BOTH"):
    """
    Create tensors for a given site (generator)

    :param records: list
        List of records

    :param searcher: AlleleSearcher
        Searcher for site

    :param maxAlleleLength: int
        Maximum size of alleles to check

    :param hotspotMethod: str
        Method to detect alleles at hostpot locations
    """
    execStart = timer();
    global TENSOR_TIME;

    logging.debug("Starting tensor retrieval")

    def allelesInRecord(record):
        alleles_ = [record.ref] + record.alt;
        return set(alleles_[gt] for gt in record.gt);

    for record in records:
        start_ = record.position;
        stop_ = start_ + len(record.ref);

        logging.debug("Performing assembly in %d -> %d for tensor retrieval" % (start_, stop_))
        searcher.assemble(start_, stop_);
        logging.debug("Completed assembly in %d -> %d for tensor retrieval" % (start_, stop_))

        tensors = [];
        labels = [];
        alleles = [];
        scores = [];
        supportingReads = [];
        supportingReadsStrict = [];
        tensors2 = [];
        supportingReads2 = [];
        supportingReadsStrict2 = [];
        truths = allelesInRecord(record);

        for allele in [record.ref] + record.alt:
            # If there is no support for allele, then we don't include any information regarding the allele
            numSupportsForAllele = searcher.numReadsSupportingAlleleStrict(allele, 0);

            if searcher.hybrid:
                numSupportsForAllele += searcher.numReadsSupportingAlleleStrict(allele, 1);

            if numSupportsForAllele == 0:
                logging.debug("Allele %s has no support from searcher" % allele);
                continue;

            if len(allele) > maxAlleleLength:
                logging.debug("Allele %s is too long" % allele);
                continue;

            # searcher = searchers[0];
            alleles.append(allele);
            labels.append(1 if (allele in truths) else 0);

            feature = searcher.computeFeatures(allele, 0);
            tensors.append(feature);
            supportingReads.append(-1);
            supportingReadsStrict.append(searcher.numReadsSupportingAlleleStrict(allele, 0));

            if searcher.hybrid:
                feature2 = searcher.computeFeatures(allele, 1);
                tensors2.append(feature2);
                supportingReads2.append(-1);
                supportingReadsStrict2.append(searcher.numReadsSupportingAlleleStrict(allele, 1));

            
        siteLabel = 0 if sum(labels) <= 1 else 1;
        # total = sum(labels);
        # labels = [i / total for i in labels];

        TENSOR_TIME += (timer() - execStart);
        yield {
            'chromosome': record.chromosome,
            'start': start_,
            'stop': stop_,
            'alleles': alleles,
            'tensors': tensors,
            'labels': labels,
            'siteLabel': siteLabel,
            'scores': scores,
            'supportingReads': supportingReads,
            'supportingReadsStrict': supportingReadsStrict,
            'tensors2': tensors2,
            'supportingReads2': supportingReads2,
            'supportingReadsStrict2': supportingReadsStrict2,
        };
        execStart = timer();

    logging.debug("Completed tensor retrieval")


def data(
    hotspots,
    readSamplers,
    searcherFactory,
    reference,
    vcf=None,
    bed=None,
    distance=PileupDataTools.MIN_DISTANCE,
    maxAlleleLength=80,
    hotspotMethod="BOTH",
    searcherCollection=None,
):
    """
    Generator for data for training/variant calling

    :param hotspots: collections.defaultdict
        Dictionary of hotspot regions

    :param readSamplers: list
        List of PileupDataTools.ReadSampler objects

    :param searcherFactory: SearcherFactory
        Factory object to create searchers

    :param reference: str
        Reference cache location

    :param vcf: str
        Ground-truth vcf if labeling is desired

    :param bed: str
        High-confidence bed file if vcf is provided

    :param distance: int
        Distance for clustering candidates

    :param maxAlleleLength: int
        Maximum size of an allele to be considered

    :param hotspotMethod: str
        Method to produce hotspots

    :param searcherCollection: collections.defaultdict
        Dictionary of intervaltree objects searchers constructed
        to determine hotspot locations
    """
    execStart = timer();
    global DATAGEN_TIME;

    ref = ReferenceCache(database=reference);

    if vcf is not None:
        assert(bed is not None), "Provide high conf regions when vcf is provided";
        bedRegions = bedReader(bed);
        truthSet = groundTruthReader(vcf);
    else:
        bedRegions = None;
        truthSet = None;

    for chromosome in hotspots:
        tree = hotspots[chromosome];
        locations = sorted(tree.all_intervals);
        clusters = clusterLocations(locations, distance, maxAlleleLength);
        ref.chrom = chromosome;

        for cluster in clusters:
            # First, prepare AlleleSearcher instances for the cluster of hotspots
            start = cluster[0][0] - distance // 2;
            stop = cluster[-1][1] + distance // 2 - 1;
            searcher = None;

            if searcherCollection is not None:
                preconstructed = searcherCollection[chromosome][start: stop];
                for pre in preconstructed:
                    if (pre[0] <= start < stop <= pre[1]) and pre[2]:
                        logging.debug("Found pre-constructed searcher for span %d, %d" % (start, stop));
                        searcher = pre[2];
                        break;

            if not searcher:
                container = [
                    R(chromosome, start, stop) for R in readSamplers
                ];
                searcher = searcherFactory(
                    container, start, stop
                );

            # Create candidate vcf records for each hotspot and label them
            # candidates = getLabeledCandidates(
            candidates = get_labeled_candidates(
                chromosome,
                ref,
                searcher,
                cluster,
                truthSet,
                bedRegions,
                hotspotMethod=hotspotMethod,
                maxAlleleLength=maxAlleleLength,
            );

            # Region that has too many candidates
            if candidates is None:
                DATAGEN_TIME += (timer() - execStart);
                yield {"type": "TOO_LONG", 'chromosome': chromosome, 'start': start, 'stop': stop};
                execStart = timer();
                continue;

            # It is possible that none of the candidates survive when checked against high confidence regions
            if len(candidates) == 0:
                continue;

            # Region that has mislabeled records
            if -1 in candidates[0].gt:
                DATAGEN_TIME += (timer() - execStart);
                yield {"type": "MISSED", 'chromosome': chromosome, 'start': start, 'stop': stop};
                execStart = timer();
                continue;

            # Generate tensors
            for tensors in createTensors(candidates, searcher, maxAlleleLength, hotspotMethod=hotspotMethod):
                DATAGEN_TIME += (timer() - execStart);
                logging.debug("Yielding")
                yield tensors;
                execStart = timer();
