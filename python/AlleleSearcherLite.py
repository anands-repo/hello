import libCallability
from PySamFastaWrapper import PySamFastaWrapper as ReferenceCache
import logging
import numpy as np

try:
    profile
except Exception:
    def profile(x):
        return x;


class LocationOutOfBounds(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs);


class AlleleSearcherLite:
    """
    This file provides a python wrapper for the C++ AlleleSearcherLite class
    """
    @profile
    def __init__(
        self,
        container,
        start,
        stop,
        ref,
        featureLength=150,
        pacbio=False,
        strict=True,
        useInternalLeftAlignment=False,
        noAlleleLevelFilter=False,
    ):
        """
        :param container: list/PileupContainerLite
            List of pileupcontainer objects or a single pileupcontainer object

        :param start: int
            Left limiting position of interest

        :param stop: int
            Right limiting position of interest

        :param ref: str/ReferenceCache
            Either path to the reference cache or the reference cache

        :param featureLength: int
            Length of feature map

        :param pacbio: bool
            Indicate that the reads are for PacBio if we are only using one container

        :param strict: bool
            Whether a differingRegion should strictly fall with start and stop

        :param useInternalLeftAlignment: bool
            Enable use of C++-based cigar left-alignment

        :param noAlleleLevelFilter: bool
            Do not use allele-level filters
        """
        self.start = start;
        self.stop = stop;
        self.strict = strict;
        self.featureLength = featureLength;
        self.noAlleleLevelFilter = noAlleleLevelFilter;
        containers = container if (type(container) is list) else [container];

        # Arguments for C++ searcher
        reads = [];
        names = [];
        qualities = [];
        refStarts = [];
        references = [];
        mapq = [];
        orientation = [];
        pacbio = [];
        cigartuples = [];
        self.noReads = [False for i in containers];
        self.pacbio = (len(containers) == 1) and pacbio;
        self.hybrid = len(containers) > 1;

        for i, container_ in enumerate(containers):
            if len(container_.pileupreads) != 0:
                cigartuples += [[list(x) for x in cigartuple] for cigartuple in container_.cigartuples];
                reads += [p.alignment.query_sequence for p in container_.pileupreads];
                names += [p.alignment.query_name for p in container_.pileupreads];
                qualities += [p.alignment.query_qualities for p in container_.pileupreads];
                refStarts += [p.alignment.reference_start for p in container_.pileupreads];
                mapq += [p.alignment.mapping_quality for p in container_.pileupreads];
                orientation += [-1 if p.alignment.is_reverse else 1 for p in container_.pileupreads];
            else:
                self.noReads[i] = True;

        if type(ref) is str:
            self.ref = ReferenceCache(database=ref, chrom=containers[0].chromosome);
        else:
            self.ref = ref;
            self.ref.chrom = containers[0].chromosome;

        windowStart = min(refStarts + [start]) - 10;
        windowEnd = -float('inf');  # max([container_.referenceEnd for container_ in containers]) + 10;

        for c in containers:
            if len(c.pileupreads) > 0:
                windowEnd = max(windowEnd, c.referenceEnd);

        if windowStart < 0:
            raise LocationOutOfBounds;

        if windowEnd > len(self.ref):
            raise LocationOutOfBounds;

        if windowEnd < 0:
            raise LocationOutOfBounds;

        windowEnd += 10;
        reference = ''.join(self.ref[windowStart: windowEnd]);

        if len(containers) == 1:
            pacbio = [pacbio for i in reads] 
        else:
            pacbio = [False for i in containers[0].pileupreads] + [True for i in containers[1].pileupreads];

        self.searcher = libCallability.AlleleSearcherLite(
            reads,
            names,
            qualities,
            cigartuples,
            refStarts,
            mapq,
            orientation,
            pacbio,
            reference,
            windowStart,
            10,
            useInternalLeftAlignment
        );

    @property
    def refAllele(self):
        return self.searcher.refAllele;

    @property
    def differingRegions(self):
        if all(self.noReads):
            return [];

        if hasattr(self, 'regions'):
            return self.regions;

        self.searcher.determineDifferingRegions();

        self.regions = [];

        for item in self.searcher.differingRegions:
            logging.debug("Received differing region %s" % (str((item.first, item.second))));
            if self.strict:
                if self.start <= item.first < item.second <= self.stop:
                    self.regions.append((item.first, item.second));
                else:
                    logging.debug("Discarding region due to strict requirements");
            else:
                start = max(self.start, item.first);
                stop = min(item.second, self.stop);
                if start < stop:
                    self.regions.append((start, stop));

        return self.regions;

    @property
    def allelesAtSite(self):
        alleles = set();

        if all(self.noReads):
            return alleles;

        for item in self.searcher.allelesAtSite:
            alleles.add(item);

        return alleles;

    def addAlleleForAssembly(self, allele):
        """
        Set an allele for assembly. When alleles are set, assembly uses
        only the set alleles.

        :param allele: str
            Allele to be used for assembly
        """
        if not all(self.noReads):
            self.searcher.addAlleleForAssembly(allele);

    @profile
    def computeFeatures(self, allele, index=0):
        """
        Computes features for a given allele

        :param allele: str
            Allele for which features are to be computed

        :param index: int
            The read set for which features are to be released

        :return: np.ndarray
            Feature using numpy
        """
        if self.noReads[index if self.hybrid else 0]:
            return np.zeros(shape=(1, self.featureLength, 6), dtype=np.uint8);
        else:
            index = index == 1 if self.hybrid else self.pacbio;
            return self.searcher.computeFeaturesColoredSimple(allele, self.featureLength, index);

    def coverage(self, position):
        """
        Provide the coverage at an absolute position in the reference, provided the
        position is within the region being analyzed

        :param position: int
            Position where coverage data is desired

        :return: int
            Coverage value
        """
        return self.searcher.coverage(position);

    @profile
    def assemble(self, start=None, stop=None):
        """
        Performs assembly between start and stop

        :param start: int
            start postion of suspected allele

        :param stop: int
            stop position of suspected allele

        :return: iterable
            List-like object
        """
        if all(self.noReads):
            return [];

        if start is None:
            start = self.start;

        if stop is None:
            stop = self.stop;

        if self.noAlleleLevelFilter:
            return self.searcher.assemble(start, stop, True);
        else:
            return self.searcher.assemble(start, stop, False);

    def numReadsSupportingAlleleStrict(self, allele, index):
        """
        Provides the number of reads fully encapsulating an allele

        :param allele: str
            Allele for which the number of supporting reads is desired

        :param index: int
            Read set for which query is being made

        :return: int
            Number of reads supporting the given allele
        """
        if self.noReads[index if self.hybrid else 0]:
            return 0;

        index = index == 1 if self.hybrid else self.pacbio;
        return self.searcher.numReadsSupportingAlleleStrict(allele, index);

    def determineAllelesInRegion(self, start, stop):
        """
        Function determines alleles in region without performing assembly (unlike property allelesAtSite)

        :param start: int
            Start of region

        :param stop: int
            End of region

        :return: list
            List of alleles from reads
        """
        if all(self.noReads):
            return [];
        else:
            return list(self.searcher.determineAllelesAtSite(start, stop));

    def clearAllelesForAssembly(self):
        """
        Clears preset alleles for assembly
        """
        if not all(self.noReads):
            self.searcher.clearAllelesForAssembly();

