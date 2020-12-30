# © 2019 University of Illinois Board of Trustees.  All rights reserved
import libCallability
from ReferenceCache import ReferenceCache
import logging


class AlleleSearcher:
    # Class variable indicating whether debug should be enabled
    debug = False;
    """
    This file provides a python wrapper for the C++ AlleleSearcher
    """
    def __init__(
        self,
        container,
        start,
        stop,
        ref,
        contextLength=20,
        pacbio=False,
        strict=True,
        indelRealigned=False,
        **kwargs,
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

        :param contextLength: int
            Context length to use for AlleleSearcher

        :param pacbio: bool
            Whether we are talking pacbio reads or not

        :param debug: bool
            Enable debug messages or not

        :param strict: bool
            Whether a differingRegion should strictly fall with start and stop

        :param indelRealigned: bool
            Whether input reads have been indel realigned
        """
        self.container = container;
        self.start = start;
        self.stop = stop;
        self.strict = strict;
        self.indelRealigned = indelRealigned;

        if len(self.container.pileupreads) != 0:
            # Convert cigartuples, and alignedPairs to list format
            cigartuples = [[list(x) for x in cigartuple] for cigartuple in self.container.cigartuples];

            if type(ref) is str:
                self.ref = ReferenceCache(database=ref, chrom=self.container.chromosome);
            else:
                self.ref = ref;
                self.ref.chrom = container.chromosome;

            def noneToMinus1(x):
                return -1 if x is None else x;

            def convert(alignedPair):
                return [noneToMinus1(x) for x in alignedPair];

            # Remaining arguments
            readsStart = self.container.referenceStart;
            reads = [p.alignment.query_sequence for p in self.container.pileupreads];
            names = [p.alignment.query_name for p in self.container.pileupreads];
            qualities = [p.alignment.query_qualities for p in self.container.pileupreads];
            refStarts = [p.alignment.reference_start for p in self.container.pileupreads];
            reference = ''.join(self.ref[readsStart:self.container.referenceEnd + 1]);

            self.searcher = libCallability.AlleleSearcher(
                reads,
                names,
                qualities,
                cigartuples,
                refStarts,
                reference,
                readsStart,
                start,
                stop,
                contextLength,
                2,
                10,
                pacbio or indelRealigned,
                AlleleSearcher.debug,
            );

            # self.searcher.mismatchScore = (1 if not pacbio else 4);
            self.searcher.mismatchScore = 1;  # Experimenting with '1'; should be sufficient for SNV calling with PacBio
            self.searcher.insertScore = 4 if not pacbio else 1;
            self.searcher.deleteScore = 4 if not pacbio else 1;
            self.searcher.scoreLocations();
            self.searcher.determineDifferingRegions();
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

        self.regions = [];
        for item in self.searcher.differingRegions:
            if self.strict:
                if self.start <= item.first < item.second <= self.stop:
                    self.regions.append((item.first, item.second));
            else:
                start = max(self.start, item.first);
                stop = min(item.second, self.stop);
                if start < stop:
                    self.regions.append((start, stop));

        return self.regions;

    @property
    def filteredContigs(self):
        return list(self.searcher.filteredContigs);

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

    def printMatrix(self):
        self.searcher.printMatrix();

    def computeFeatures(self, allele, normalize=False):
        """
        Computes features for a given allele

        :param allele: str
            Allele for which features are to be computed

        :param normalize: bool
            Whether features need to be normalized

        :return: np.ndarray
            Feature using numpy
        """
        return self.searcher.computeFeatures(allele, normalize);

    def computeContigs(self, left, right):
        """
        Computes filtered contigs in the region

        :param left: int
            Left position to the left of which flanks of contigs need to be computed

        :param right: int
            Right position to the right of which flanks of contigs need to be computed

        :return: NoneType
            Side-effects
        """
        self.searcher.computeContigs(left, right);

    def allelePositionInContig(self, contig):
        """
        Given contig, provide the position of the allele of interest within this contig

        :param contig: str
            Contig from filteredContigs list

        :return: int
            Position of the allele of interest
        """
        return self.searcher.allelePositionInContig(contig);

    def alleleInContig(self, contig):
        """
        Given contig, provide the allele of interest in the contig

        :param contig: str
            Contig from filtered contigs list

        :return: str
            The allele of interest
        """
        return self.searcher.alleleInContig(contig);

    def supportPositions(self, contig):
        """
        For a contig, which are the positions supported by each read in container

        :param contig: str
            Contig for which information is desired

        :return: vector<pair<int,int> >
            List of support positions
        """
        return self.searcher.supportPositions(contig);

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

        return self.searcher.assemble(start, stop);

    def setAlleleForAssembly(self, allele):
        """
        Force-set allele for assembly. This is for sites which are triallelic or greater,
        and we pick the top two alleles and force assembly on those two alleles.

        :param allele: str
            Allele to set for assembly
        """
        self.searcher.setAlleleForAssembly(allele);

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
