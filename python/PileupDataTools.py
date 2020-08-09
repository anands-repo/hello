"""
Classes for generating and manipulating training data for the new AlleleSearcher DNN
"""
import ast
from AlleleSearcherLite import AlleleSearcherLite, LocationOutOfBounds
from PySamFastaWrapper import PySamFastaWrapper as ReferenceCache
from PileupContainerLite import PileupContainerLite
import logging
from HotspotDetector import combineOverlapping
import intervaltree
import collections
import copy
import pysam
from timeit import default_timer as timer

READ_RATE_ILLUMINA = (1000, 30);
READ_RATE_PACBIO = (100, 100);
MIN_DISTANCE = 13;
CHRPREFIX1 = "";
CHRPREFIX2 = "";
FLANKING_BASES = 75;
CANDIDATE_READER_TIME = 0;

try:
    profile
except Exception:
    def profile(_):
        return _;


class SearcherFactory:
    def __init__(self, ref, featureLength, pacbio, useInternalLeftAlignment, noAlleleLevelFilter=False, clr=False):
        """
        Factory object for creating a specific type of allele searcher again and again

        :param ref: ReferenceCache
            ReferenceCache object

        :param featureLength: int
            Length of feature map desired

        :param pacbio: bool
            Are we dealing with PacBio reads (for single read mode)?

        :param useInternalLeftAlignment: bool
            Whether we should use internal left-alignment

        :param noAlleleLevelFilter: bool
            No allele level filter to be used

        :param clr: bool
            Whether the read type is CLR for pacbio reads
        """
        self.featureLength = featureLength;
        self.pacbio = pacbio;
        self.useInternalLeftAlignment = useInternalLeftAlignment;
        self.noAlleleLevelFilter = noAlleleLevelFilter or (clr and pacbio);
        self.clr = clr;
        self.ref = ReferenceCache(database=ref);

    def __call__(self, container, start, stop):
        """
        :param container: list/PileupContainerLite
            Container object or list of container objects for hybrid mode

        :param start: int
            Start co-ordinates

        :param stop: int
            Stop co-ordinates
        """
        return AlleleSearcherLite(
            container=container[0] if ((type(container) is list) and (len(container) == 1)) else container,
            start=start,
            stop=stop,
            ref=self.ref,
            featureLength=self.featureLength,
            pacbio=self.pacbio,
            useInternalLeftAlignment=self.useInternalLeftAlignment,
            noAlleleLevelFilter=self.noAlleleLevelFilter,
        );


class ReadSampler:
    def __init__(self, bamfile, readRate, chrPrefix, pacbio, noClip=False, prorateReadsToRetain=True):
        self.readRate = readRate;
        self.chrPrefix = chrPrefix;
        self.pacbio = pacbio;
        self.bamfile = bamfile;
        self.bhandle = pysam.AlignmentFile(bamfile, 'rb');
        self.noClip = noClip;
        self.prorateReads = prorateReadsToRetain;

    def __call__(self, chromosome, start, stop):
        if self.prorateReads:
            if stop - start > self.readRate[1]:
                numReadsToRetain = self.readRate[0] / self.readRate[1] * (stop - start);
            else:
                numReadsToRetain = self.readRate[0];
        else:
            numReadsToRetain = self.readRate;

        container = PileupContainerLite(
            self.bhandle,
            self.chrPrefix + chromosome,
            start,
            stop - start,
            clipReads=(self.pacbio and not self.noClip),
            maxNumReads=numReadsToRetain,
            clipFlank=200,
        );

        return container;


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


def isWithinDistance(pointA, pointB, distance=MIN_DISTANCE):
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


def hotspotsReader(filename, distance=MIN_DISTANCE):
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
                if isWithinDistance(cluster[-1], point, distance=distance):
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


def diffIntervals(a, b):
    """
    Check whether two sets of intervals are different

    :param a: set
        First set of intervals

    :param b: set
        Second set of intervals

    :return: bool
        Whether intervals are different or not
    """
    a_ = set((x, y) for x, y, _ in a);
    b_ = set((x, y) for x, y, _ in b);
    return a_ == b_;


def obtainConsensusRegions(searchers):
    """
    Obtain a list of consensus differing locations when multiple searchers are involved

    :param searchers: list
        List of AlleeSearcherLite objects

    :return: intervaltree.IntervalTree
        Consensus intervals
    """
    intervals = intervaltree.IntervalTree();

    for searcher in searchers:
        intervals.update(
            [intervaltree.Interval(*x) for x in searcher.differingRegions]
        );

    intervals.merge_overlaps();

    while True:
        allIntervals = copy.deepcopy(intervals.all_intervals);

        for x, y, _ in allIntervals:
            for searcher in searchers:
                expandedRegion = searcher.expandRegion(x, y);
                intervals.addi(expandedRegion[0], expandedRegion[1], None);

        intervals.merge_overlaps();

        # If intervals have stopped changing, return
        if diffIntervals(allIntervals, intervals.all_intervals):
            break;

    return intervals;


@profile
def candidateReader(
    readSamplers,
    searcherFactory,
    activity,
    distance=MIN_DISTANCE,
    hotspotMode="BOTH",
    provideSearchers=False,
):
    """
    Determines all candidate variant call regions

    :param readSamplers: list
        ReadSampler objects for obtaining reads

    :param searcherFactory: SearcherFactory
        SearcherFactory object

    :param activity: str
        Filename for active regions

    :param distance: int
        Distance parameter for instanciating hotspotReader

    :param hotspotMode: str
        How to determine exact hotspots

    :param provideSearchers: bool
        Return searchers along with hotspots

    :return: tuple
        tuple of collections.defaultdict indicating all the hotspots
        and searchers constructed
    """
    execStart = timer();
    hotspotIterator = hotspotsReader(activity, distance);
    hotspots = collections.defaultdict(intervaltree.IntervalTree);
    searcherCollection = None;

    if provideSearchers:
        searcherCollection = collections.defaultdict(intervaltree.IntervalTree);

    for i, item in enumerate(hotspotIterator):
        chromosome, start, stop = item['chromosome'], item['start'], item['stop'];
        searchers = [];

        try:
            containers = [
                rS(chromosome, max(0, start - FLANKING_BASES), stop + FLANKING_BASES)
                for rS in readSamplers
            ];
            searcher = searcherFactory(containers, start, stop)

            logging.debug("Constructed searcher for span %d, %d" % (start, stop));

            if provideSearchers:
                searcherCollection[chromosome].addi(start, stop, searcher);
        except LocationOutOfBounds:
            logging.warning("Location %s, %d, %d is out of bounds" % (chromosome, start, stop));
            continue;

        differingRegions = searcher.differingRegions;
        for x, y in differingRegions:
            hotspots[chromosome].addi(x, y, None);

        # NOTE: With the merged AlleleSearcher model, there is no need to obtain a consensus anymore
        # if not searcher.hybrid:
        #     differingRegions = searcher.differingRegions;
        #     for x, y in differingRegions:
        #         hotspots[chromosome].addi(x, y, None);
        # else:
        #     hotspots[chromosome].update(obtainConsensusRegions(searchers));

        if (i + 1) % 100 == 0:
            logging.info("Completed %d hotspot items" % (i + 1));

    for key in hotspots:
        hotspots[key].merge_overlaps();

    execStop = timer();

    global CANDIDATE_READER_TIME;
    CANDIDATE_READER_TIME += (execStop - execStart);

    return hotspots, searcherCollection;
