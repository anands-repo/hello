import libCallability
from ReferenceCache import ReferenceCache
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
        featureLength=100,
        pacbio=False,
        strict=True,
        indelRealigned=False,
        useInternalLeftAlignment=False,
        useAdvanced=False,
        useColored=False,
        useMapQ=False,
        useOrientation=False,
        useQEncoding=False,
        useColoredSimple=False,
        noAlleleLevelFilter=False,
    ):
        """
        :param container: PileupContainer/PileupContainerLite
            Container for reads from which to create wrapper

        :param start: int
            Left limiting position of interest

        :param stop: int
            Right limiting position of interest

        :param ref: str/ReferenceCache
            Either path to the reference cache or the reference cache

        :param featureLength: int
            Length of feature map

        :param pacbio: bool
            Whether we are talking pacbio reads or not

        :param strict: bool
            Whether a differingRegion should strictly fall with start and stop

        :param indelRealigned: bool
            Whether input reads have been indel realigned

        :param useInternalLeftAlignment: bool
            Enable use of C++-based cigar left-alignment

        :param useAdvanced: bool
            Use advanced feature map

        :param useColored: bool
            Use colored feature maps

        :param useMapQ: bool
            Whether we want to use MapQ encoded into DNN feature maps

        :param useOrientation: bool
            Whether we want to use orientation of reads in the input feature maps

        :param useQEncoding: bool
            Encode quality scores (MAPQ and Q) as Q-scores, not probabilities

        :param useColoredSimple: bool
            Use colored feature maps but indels are squashed into single bases

        :param noAlleleLevelFilter: bool
            Do not use allele-level filters
        """
        self.container = container;
        self.start = start;
        self.stop = stop;
        self.strict = strict;
        self.indelRealigned = indelRealigned;
        self.featureLength = featureLength;
        self.useAdvanced = useAdvanced;
        self.useMapQ = useMapQ;
        self.useOrientation = useOrientation;
        self.useQEncoding = useQEncoding;
        self.useColored = useColored;
        self.useColoredSimple = useColoredSimple;
        self.noAlleleLevelFilter = noAlleleLevelFilter;
        self.pacbio = pacbio;

        if len(self.container.pileupreads) != 0:
            # Convert cigartuples, and alignedPairs to list format
            cigartuples = [[list(x) for x in cigartuple] for cigartuple in self.container.cigartuples];

            if type(ref) is str:
                self.ref = ReferenceCache(database=ref, chrom=self.container.chromosome);
            else:
                self.ref = ref;

                # Setting chromosome in reference cache
                # incurs computations downstream
                if self.ref.chrom != container.chromosome:
                    self.ref.chrom = container.chromosome;

            def noneToMinus1(x):
                return -1 if x is None else x;

            # Check whether we are within acceptable bounds
            if min(self.container.referenceStart, start) - 10 < 0:
                raise LocationOutOfBounds;

            if max(self.container.referenceEnd, stop) + 10 > len(self.ref):
                raise LocationOutOfBounds;

            # Remaining arguments
            windowStart = min(self.container.referenceStart, start) - 10;
            reads = [p.alignment.query_sequence for p in self.container.pileupreads];
            names = [p.alignment.query_name for p in self.container.pileupreads];
            qualities = [p.alignment.query_qualities for p in self.container.pileupreads];
            refStarts = [p.alignment.reference_start for p in self.container.pileupreads];
            reference = ''.join(self.ref[windowStart: max(stop, self.container.referenceEnd) + 10]);  # Provide a wide berth
            mapq = [p.alignment.mapping_quality for p in self.container.pileupreads];
            orientation = [-1 if p.alignment.is_reverse else 1 for p in self.container.pileupreads];

            self.searcher = libCallability.AlleleSearcherLite(
                reads,
                names,
                qualities,
                cigartuples,
                refStarts,
                mapq,
                orientation,
                reference,
                windowStart,
                10,
                useInternalLeftAlignment,
                useMapQ,
                useOrientation,
                useQEncoding
            );

            self.searcher.mismatchScore = 1;
            self.searcher.insertScore = 4 if not pacbio else 1;
            self.searcher.deleteScore = 4 if not pacbio else 1;
            # self.searcher.scoreLocations();
            # self.searcher.determineDifferingRegions();
            self.noReads = False;
        else:
            self.noReads = True;

    @property
    def refAllele(self):
        return self.searcher.refAllele;

    @property
    def differingRegions(self):
        if self.noReads:
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
    def mismatchScore(self):
        return self.searcher.mismatchScore;

    @mismatchScore.setter
    def mismatchScore(self, _score):
        self.searcher.mismatchScore = _score;

    @property
    def insertScore(self):
        return self.searcher.insertScore;

    @insertScore.setter
    def insertScore(self, _score):
        self.searcher.insertScore = _score;

    @property
    def deleteScore(self):
        return self.searcher.deleteScore;

    @deleteScore.setter
    def deleteScore(self, _score):
        self.searcher.deleteScore = _score;

    @property
    def allelesAtSite(self):
        alleles = set();
        if self.noReads:
            return alleles;
        for item in self.searcher.allelesAtSite:
            alleles.add(item);
        return alleles;

    def addAlleleForAssembly(self, allele):
        """
        Set an allele for assembly. When alleles are set, assembly uses
        only the set alleles

        :param allele: str
            Allele to be used for assembly
        """
        if not self.noReads:
            self.searcher.addAlleleForAssembly(allele);

    @profile
    def computeFeatures(self, allele, *args):
        """
        Computes features for a given allele

        :param allele: str
            Allele for which features are to be computed

        :return: np.ndarray
            Feature using numpy
        """
        if self.useColoredSimple:
            if self.noReads:
                return np.zeros(shape=(1, self.featureLength, 6), dtype=np.uint8);
            else:
                return self.searcher.computeFeaturesColoredSimple(allele, self.featureLength);
        elif (not self.useAdvanced) and (not self.useColored):
            if self.noReads:
                return np.zeros(shape=(self.featureLength, 8));
            else:
                return self.searcher.computeFeatures(allele, self.featureLength);
        elif self.useColored:
            if self.noReads:
                return np.zeros(shape=(1, self.featureLength, 6), dtype=np.uint8);
            else:
                return self.searcher.computeFeaturesColored(allele, self.featureLength);
        else:
            if self.noReads:
                return np.zeros(shape=(1, self.featureLength, 18));
            else:
                return self.searcher.computeFeaturesAdvanced(allele, self.featureLength);

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
        if self.noReads:
            return [];

        if start is None:
            start = self.start;

        if stop is None:
            stop = self.stop;

        if self.noAlleleLevelFilter:
            return self.searcher.assemble(start, stop, True);
        else:
            return self.searcher.assemble(start, stop, False);

    def numReadsSupportingAllele(self, allele):
        """
        Provides the number of reads supporting each allele

        :param allele: str
            Allele for which the number of supporting reads is desired
        """
        if self.noReads:
            return 0;

        return self.searcher.numReadsSupportingAllele(allele);

    def numReadsSupportingAlleleStrict(self, allele):
        """
        Provides the number of reads fully encapsulating an allele

        :param allele: str
            Allele for which the number of supporting reads is desired
        """
        if self.noReads:
            return 0;

        return self.searcher.numReadsSupportingAlleleStrict(allele);

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
        if self.noReads:
            return [];
        else:
            return list(self.searcher.determineAllelesAtSite(start, stop));

    def expandRegion(self, start, stop):
        """
        Expand a region to adjacent indels
        """
        if self.noReads:
            return (start, stop);

        logging.debug("Number of reads at site = %d" % len(self.container.pileupreads));
        region = self.searcher.expandRegion(start, stop);
        return int(region.first), int(region.second);

    def clearAllelesForAssembly(self):
        """
        Clears preset alleles for assembly
        """
        if not self.noReads:
            self.searcher.clearAllelesForAssembly();

    def __str__(self):
        string = "Chromosome %s, start %d, stop %d, type %s, number of reads %d" %(
            self.container.chromosome,
            self.start,
            self.stop,
            'pacbio' if self.pacbio else 'illumina',
            len(self.container.pileupreads),
        );
        return string;
