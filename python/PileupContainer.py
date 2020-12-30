# © 2019 University of Illinois Board of Trustees.  All rights reserved
import pysam
import warnings
import random
from functools import reduce
import logging

BAM_CMATCH     = 0;
BAM_CINS       = 1;
BAM_CDEL       = 2;
BAM_CREF_SKIP  = 3;
BAM_CSOFT_CLIP = 4;
BAM_CHARD_CLIP = 5;
BAM_CPAD       = 6;
BAM_CEQUAL     = 7;
BAM_CDIFF      = 8;
BAM_CBACK      = 9;

def is_usable_read(pileupread):
    """
    Taken from DeepVariant. Note from DeepVariant paper (https://www.biorxiv.org/content/biorxiv/early/2018/03/20/092890.full.pdf):-
    We filter away any unusable reads (see is_usable_read() below) if it is marked as a duplicate,
    as failing vendor quality checks, isn't aligned or if this isn't the primary alignment, mapping
    quality is less than 10, or the read is paired and not marked as properly placed. We further only
    include read bases as potential alleles if all of the bases in the alleles have a base quality >= 10.

    :param pileupread: pysam.PileupRead
        A pileupread

    :return: bool
        Flag indicating whether read may be used
    """
    alignment = pileupread.alignment;
    usable    = True;
    usable    = usable and not alignment.is_unmapped;
    usable    = usable and not (alignment.is_secondary or alignment.is_supplementary);
    usable    = usable and not alignment.is_duplicate;
    usable    = usable and (not alignment.is_paired or alignment.is_proper_pair);

    # Anand - modified on 2019/12/16 to reads with non-zero mapping quality
    usable    = usable and (alignment.mapping_quality > 0);
    # Missing: failed_vendor_quality_checks

    # Anand - commented on 2019/12/16
    # # TBD: No 'N's; remove later
    # usable = usable and not ('N' in alignment.query_sequence);

    # # Keep only reads with at least 5% of their length unclipped
    # usable = usable and (len(alignment.query_sequence) * 0.05 < len(alignment.query_alignment_sequence));

    return usable;


class PileupContainer(object):
    """
    Object acts as interface for pileup locations, collecting filtered reads from the location. Objects can
    subsample reads as necessary from the location.
    """
    def __init__(self, bamfile, chromosome, position, span, maxNumReads=10000, limitColumns = False):
        """
        Init function for class

        :param bamfile: str
            Bamfilename

        :param chromosome: str
            Chromosome name

        :param position: int
            Location in the bam file (0-based)

        :param span: int
            Length of the reference sequence of interest
            from which reads should be collected

        :param maxNumReads: int
            Limit on the number of reads. Reservoir sampling is used to 
            limit to this number if there are too many reads

        :param limitColumns: bool
            Limit columns read to specific positions (see __get_reads)
        """
        self.read_list    = [];
        self.bamname      = bamfile;
        self.chromosome   = chromosome;
        self.position     = position;
        self.span         = span;
        self.avg_cov      = 0;
        self.maxNumReads  = maxNumReads;
        self.limitColumns = limitColumns;
        self.__get_reads();

    def __get_reads(self):
        records_with_name = set();
        coverages_in_span = dict();
        itemEncountered   = 0;

        with pysam.AlignmentFile(self.bamname, 'rb') as bhandle:
            for column in bhandle.pileup(self.chromosome, self.position, self.position + self.span):
                # ANAND - tentative bug fix, many reads may be excluded if we keep these lines; 06/06/2019
                if self.limitColumns:
                    if column.pos not in list(range(self.position, self.position + self.span)):
                        continue;

                coverage_at_column = 0;

                for pileupread in column.pileups:
                    query_name = pileupread.alignment.query_name;
                    direction  = pileupread.alignment.is_reverse;

                    if query_name is None:
                        raise ValueError("No query name found! BAM file not compatible");

                    if not is_usable_read(pileupread):
                        continue;

                    coverage_at_column += 1;

                    if (query_name, direction) in records_with_name:
                        continue;

                    records_with_name.add((query_name, direction));

                    pKeep    = self.maxNumReads / (itemEncountered+1);

                    if random.uniform(0,1) < pKeep:
                        if len(self.read_list) < self.maxNumReads:
                            self.read_list.append(pileupread);
                        else:
                            indexToReplace = random.sample(range(len(self.read_list)), 1)[0];
                            self.read_list[indexToReplace] = pileupread;

                    itemEncountered += 1;

                coverages_in_span[column.pos] = coverage_at_column;

        if len(coverages_in_span) > 0:
            self.avg_cov = sum(coverages_in_span.values()) / len(coverages_in_span);

        self._coverages = coverages_in_span;

    @property
    def pileupreads(self):
        return self.read_list;

    @pileupreads.setter
    def pileupreads(self, item):
        self.read_list = item;

    @property
    def coverages(self):
        return self._coverages;

    def pileupInColumns(self, start, end):
        raise NotImplementedError();

    @property
    def referenceStart(self):
        return min([r.alignment.reference_start for r in self.read_list]);

    @property
    def referenceEnd(self):
        return max([r.alignment.reference_end for r in self.read_list]);

def determineNumMismatches(read, ref, cigar):
    """
    Determine the number of mismatches in a read given the cigar string

    :param read: str
        Read string (full read)

    :param ref: str
        Reference string

    :param cigar: tuple
        Cigar tuple

    :return: int
        Number of mismatches
    """
    referencePosition = 0;
    readPosition = 0;
    numMismatches = 0;

    for item in cigar:
        operation, length = item;

        if operation in [BAM_CMATCH, BAM_CEQUAL, BAM_CDIFF]:
            for i in range(length):
                if read[readPosition] != ref[referencePosition]:
                    numMismatches += 1;

                readPosition += 1;
                referencePosition += 1;

        elif operation == BAM_CDEL:
            referencePosition += length;

        elif operation == BAM_CINS:
            readPosition += length;

        elif operation == BAM_CSOFT_CLIP:
            readPosition += length;

        elif operation == BAM_CREF_SKIP:
            referencePosition += length;

        elif operation == BAM_CPAD:
            logging.warning("BAM_CPAD is not being used, results may be erroneous");

    return numMismatches;

def leftShiftedCigar(cigar, position):
    """
    Left shift a cigar entry at a given position

    :param cigar: list
        List of cigartuples

    :param position: int
        Position of the cigar entry of interest

    :return: tuple
        New cigar tuple, new position of cigar entry
    """
    if position == 0:
        return None;

    lastCigarLeft = cigar[position - 1];

    if lastCigarLeft[0] in [BAM_CMATCH, BAM_CEQUAL, BAM_CDIFF]:
        # newLastCigarLeft = [(BAM_CMATCH, lastCigarLeft[1]-1)];
        newLastCigarLeft = [(lastCigarLeft[0], lastCigarLeft[1] - 1)];

        if newLastCigarLeft[0][1] == 0:
            newLastCigarLeft = [];

        newNextCigarRight = [];

        if position < len(cigar) - 1:
            # This means, there was something to the right of position.
            # We need to adjust that as well
            if cigar[position + 1][0] in [BAM_CMATCH, BAM_CEQUAL, BAM_CDIFF]:
                newNextCigarRight = [(cigar[position + 1][0], cigar[position + 1][1] + 1)];
            else:
                # The immediate left position was one of BAM_CMATCH, BAM_CEQUAL, BAM_CDIFF.
                # We shift it to the right of our cigar. However, we use BAM_CMATCH
                # which covers everything. In this case, BAM_CMATCH doesn't get absorbed into
                # whatever is on the right, so its okay (the representation is still parsimonious).
                newNextCigarRight = [(BAM_CMATCH, 1), cigar[position + 1]];
        else:
            # This means there is nothing else to the right, so we put whatever was on the left
            # of the cigar (which is a match) to the right side.
            newNextCigarRight = [(BAM_CMATCH, 1)];

        leftCigarEntries  = (cigar[: position - 1] if (position - 2 >= 0) else []) + newLastCigarLeft;
        rightCigarEntries = newNextCigarRight + cigar[position + 2:];

        newCigar          = leftCigarEntries + [cigar[position]] + rightCigarEntries;
        newPosition       = len(leftCigarEntries);

        return newCigar, newPosition;
    else:
        return None;

def clusterToCigar(cluster):
    """
    Convert a cluster of cigar items into a single cigar
    """
    if len(cluster) == 0: return cluster;

    operations, lengths = zip(*cluster);
    identical = lambda array : reduce(
                                    lambda x, y : x and y, 
                                    [a==array[0] for a in array],
                                    True
                               );

    if identical(operations):
        newLength = sum(lengths);
        return [[operations[0], newLength]];
    else:
        if reduce(lambda x, y : x and y,
                    [o in [BAM_CINS, BAM_CDEL] for o in operations],
                        True):
            numDels = sum([l for o, l in cluster if o == BAM_CDEL]);
            numIns  = sum([l for o, l in cluster if o == BAM_CINS]);

            newCigarCluster = cluster;

            if numIns > numDels:
                if numDels > 0:
                    newCigarCluster = [[BAM_CINS, numIns-numDels], [BAM_CMATCH, numDels]];
                else:
                    newCigarCluster = [[BAM_CINS, numIns]];
            else:
                if numIns > 0:
                    newCigarCluster = [[BAM_CDEL, numDels-numIns], [BAM_CMATCH, numIns]];
                else:
                    newCigarCluster = [[BAM_CDEL, numDels]];

            return newCigarCluster;
        else:
            return cluster;

def removeCigarRedundancies(cigar, read, ref):
    """
    Merge multiple adjancent cigars of the same type.
    Absorb adjacent indels if no new mismatches are introduced.
    """
    cigarPosition = 0;
    cigarNew      = [];
    cluster       = [];

    numDifferencesOrig = determineNumMismatches(read, ref, cigar);

    try:
        while (len(cigarNew) == 0) or (0 in tuple(zip(*cigarNew))[1]):
            if len(cigarNew) > 0:
                cigar = [];

                for item in cigarNew:
                    if item[1] != 0:
                        cigar.append(item);

                cigarNew = [];

            for i, cigarItem in enumerate(cigar):
                if len(cluster) == 0:
                    cluster.append(cigarItem);
                    continue;

                if cluster[-1][0] == cigarItem[0]:
                    cluster.append(cigarItem);

                elif (cluster[-1][0] in [BAM_CINS, BAM_CDEL]) and \
                        (cigarItem[0] in [BAM_CINS, BAM_CDEL]):
                    cluster.append(cigarItem);

                else:
                    newCigarItem = clusterToCigar(cluster);
                    newPartialCigar = cigarNew + newCigarItem + cigar[i+1:];

                    numDifferencesNew = determineNumMismatches(
                                        read, ref, newPartialCigar
                                     );

                    if numDifferencesNew == numDifferencesOrig:
                        cigarNew += newCigarItem;
                    else:
                        cigarNew += cluster;
 
                    cluster = [cigarItem];

            if len(cluster) > 0:
                newCigarItem = clusterToCigar(cluster);
                newPartialCigar = cigarNew + newCigarItem + cigar[i+1:];

                numDifferencesNew = determineNumMismatches(
                                    read, ref, newPartialCigar
                                 );

                if numDifferencesNew == numDifferencesOrig:
                    cigarNew += newCigarItem;
                else:
                    cigarNew += cluster;
                cluster = [];
    except IndexError:
        print(cigarNew);
        raise ValueError();

    return cigarNew;

def cigarToAln(read, ref, cigartuples):
    """
    Convert a cigar-based alignment to a pairwise2.align type alignment

    :param contig: str
        Contig string

    :param ref: str
        Reference sequence string

    :param cigartuples: list
        List of cigartuples
    """
    alignedPairs = [];
    readPosition = 0;
    refPosition  = 0;

    for cigaritem in cigartuples:
        operation, length = cigaritem;

        if operation not in [BAM_CMATCH, BAM_CINS, BAM_CDEL]:
            raise ValueError("Cigar operation should be match, insert, or delete");

        if operation == BAM_CMATCH:
            for i in range(length):
                alignedPairs.append((ref[refPosition], read[readPosition]));
                readPosition += 1;
                refPosition  += 1;

        elif operation == BAM_CINS:
            for i in range(length):
                alignedPairs.append(('-', read[readPosition]));
                readPosition += 1;

        elif operation == BAM_CDEL:
            for i in range(length):
                alignedPairs.append((ref[refPosition], '-'));
                refPosition += 1;

    return alignedPairs;

def alnToCigar(alignment):
    """
    From a Bio.pairwise2 alignment, obtain cigartuples

    :param alignment: list
        Alignment between two sequences

    :return: list
        Cigartuples list
    """
    state = None;
    counter = 0;
    cigars = [];

    for item in alignment:
        if (item[0] != '-') and (item[1] != '-'):
            nextState = BAM_CMATCH;
        elif (item[0] == '-') and (item[1] != '-'):
            nextState = BAM_CINS;
        elif (item[0] != '-') and (item[1] == '-'):
            nextState = BAM_CDEL;
        else:
            raise ValueError("Unknown type of alignment entry");

        if state is None:
            state = nextState;
            counter = 1;
        else:
            if (state == nextState):
                counter += 1;
            else:
                cigars.append((state, counter));
                counter = 1;
                state = nextState;

    if counter > 0:
        cigars.append((state, counter));
        counter = 0;
        state = None;

    return cigars;

def adjustCigartuplesBioPythonWrapper(
    alignments,
    reference,
    starts,
    returnContainer = False,
):
    """
    Wrapper for performing cigar adjustment on Bio.pairwise2 alignments

    :param alignments: list
        List of alignment results from pairwise2

    :param reference: ReferenceCache
        ReferenceCache object representing reference sequence
        (or similar interface)

    :param starts: list
        List of reference coordinates of each alignment's 
        start position in "alignments" list

    :param returnContainer: bool
        If set to True, the function also returns the adjusted dummy container

    :return: list 
        List of adjusted alignments
    """
    readSequence = lambda aln : ''.join([x[1] for x in aln if x[1] != '-']);
    refSequence  = lambda aln : ''.join([x[0] for x in aln if x[0] != '-']);
    alnLength    = lambda aln : len([1 for x in aln if x[0] != '-']);
    container    = PileupContainerDummy();
    container.chromosome = reference.chrom;
    container.pileupreads = [];

    # Create dummy PileupContainer objects for each alignment
    for alignment, start in zip(alignments, starts):
        read = PileupReadDummy();
        read.alignment = AlignedSegmentDummy();
        read.alignment.reference_start = start;
        read.alignment.query_sequence = readSequence(alignment);
        read.alignment.reference_end = start + alnLength(alignment);
        read.alignment.cigartuples = alnToCigar(alignment);
        read.alignment.ref_sequence = refSequence(alignment);
        container.pileupreads.append(read);

    # Call base function for cigartuple adjustment
    adjustCigartuples(container, reference);

    # Reformat adjusted cigartuples
    newAlignments = [];

    for pileupread, start, cigartuples in zip(
                              container.pileupreads,
                              starts,
                              container.cigartuples,
                             ):

        newStart = pileupread.alignment.reference_start;
        assert(newStart == start), \
            "Alignments are not supposed to have shifted start positions";
        newAln = cigarToAln(
                    read = pileupread.alignment.query_sequence,
                    ref = pileupread.alignment.ref_sequence,
                    cigartuples = cigartuples,
                 );
        newAlignments.append(newAln);

    if not returnContainer:
        return newAlignments;
    else:
        return newAlignments, container;

def adjustAlignedPairs(container):
    """
    Function determines new list of aligned pairs from modified cigartuples

    :param container: PileupContainer
        Object that has scanned all the reads

    :return: NoneType
        Operates through side-effects
    """
    container.alignedPairs = [];

    for pileupread, cigar in zip(container.pileupreads, container.cigartuples):
        refCounter   = pileupread.alignment.reference_start;
        readCounter  = 0;
        alignedPairs = [];

        for (operation, length) in cigar:
            if operation in [BAM_CSOFT_CLIP, BAM_CINS]:
                for i in range(length):
                    alignedPairs.append((readCounter, None));
                    readCounter += 1;

            elif operation in [BAM_CDEL, BAM_CREF_SKIP]:
                for i in range(length):
                    alignedPairs.append((None, refCounter));
                    refCounter += 1;

            elif operation in [BAM_CMATCH, BAM_CEQUAL, BAM_CDIFF]:
                for i in range(length):
                    alignedPairs.append((readCounter, refCounter));
                    readCounter += 1;
                    refCounter  += 1;

            elif operation in [BAM_CHARD_CLIP, BAM_CPAD]:
                # Do nothing, these sequences aren't present anywhere in the pileupread object.
                # Refer https://samtools.github.io/hts-specs/SAMv1.pdf, numbered page 7
                pass;

            elif operation == BAM_CBACK:
                raise ValueError("BAM_CBACK operation encountered. Do not know what to do.");

        container.alignedPairs.append(alignedPairs);

def adjustCigartuples(container, reference, enableAdjustAlignedPairs=True):
    """
    Function to adjust cigartuples by left aligning all indels

    :param container: PileupContainer
        Object that has scanned all the reads

    :param reference: ReferenceCache
        ReferenceCache object representing reference sequence
        (or similar interface)

    :param enableAdjustAlignedPairs: bool
        Enable/disable adjustment of alignedPairs

    :return: NoneType
        Operates through side-effects
    """
    reference.chrom = container.chromosome;
    container.cigartuples = [];

    for pileupread in container.pileupreads:
        referenceStart = pileupread.alignment.reference_start;
        referenceEnd   = pileupread.alignment.reference_end;
        refSegment     = ''.join(reference[referenceStart:referenceEnd]);
        read           = pileupread.alignment.query_sequence;
        cigarOrig      = pileupread.alignment.cigartuples;
        operations, _  = tuple(zip(*cigarOrig));
        cigarNew       = list(cigarOrig);

        if (BAM_CINS in operations) or (BAM_CDEL in operations):
            numDifferences = determineNumMismatches(read, refSegment, cigarOrig);
            cigarPosition  = 0;
    
            while (cigarPosition < len(cigarNew)):
                operation, length = cigarNew[cigarPosition];
    
                if (operation == BAM_CINS) or (operation == BAM_CDEL):
                    if cigarPosition == 0:
                        cigarPosition += 1;
                        continue;
    
                    position = cigarPosition;
    
                    returns  = leftShiftedCigar(cigarNew, position);

                    if returns is not None:
                        leftCigar, leftPosition = returns;
                        numMismatches = determineNumMismatches(
                                            read,
                                            refSegment,
                                            leftCigar
                                        );

                        while numMismatches == numDifferences:
                            cigarNew  = leftCigar;
                            position  = leftPosition;
                            returns   = leftShiftedCigar(cigarNew, position);
    
                            if returns is None:
                                break;
    
                            leftCigar, leftPosition = returns;
                            numMismatches = determineNumMismatches(
                                                read,
                                                refSegment,
                                                leftCigar
                                            );
    
                    cigarPosition = position + 1;
                else:
                    cigarPosition += 1;

        # Fix cigar redundancies
        cigarNew = removeCigarRedundancies(cigarNew, read, refSegment);

        # If left-most item is a deletion, remove it and adjust reference_start
        # accordingly
        if cigarNew[0][0] == BAM_CDEL:
            pileupread.alignment.reference_start += cigarNew[0][1];
            cigarNew = cigarNew[1:];

        # Add new left-parsimonious cigartuples
        container.cigartuples.append([tuple(c) for c in cigarNew]);

    # Add adjusted cigar aligned pairs
    if enableAdjustAlignedPairs:
        adjustAlignedPairs(container);

# Dummy classes for mimicking containers / reference cache
class AlignedSegmentDummy:
    def __init__(self):
        pass;

class PileupReadDummy:
    def __init__(self):
        pass;

class PileupContainerDummy:
    def __init__(self):
        pass;

class ReferenceCacheDummy:
    def __init__(self):
        self.string = [];

    def __getitem__(self, index):
        return self.string[index];

if __name__ == "__main__":
    """
    The following is a testcase for adjustCigartuples
    """
    # Create dummy dataset
    # Reference:   ACGATATATACCAGTA--TATATATATATATATATATATATAGGATACGATA
    # Read1    :         TATACCAGTA--TATATATATATATATATATATATAGGA
    # Read2    :         TATACCAGTA--TATATATATATATAT--ATATATAGGA
    # Read3    :         TATACCAGTATATATATATATATATAT--ATATATAGGA
    # Read5    :                 TA----TATATATATATATATATATATAGGATACTTTT
    # Try a mismatch masquerading as an indel
    # Reference: ACGATATATACCAGTATATA-TATATATATATATATATATAGGATACGATA
    # Read4    :         TATACCAGTATAG-ATATATATATATATATATATAGGA 
    reference = ReferenceCacheDummy();
    reference.string = list("ACGATATATACCAGTATATATATATATATATATATATATAGGATACGATA");
    read1     = PileupReadDummy();
    read1.alignment = AlignedSegmentDummy();
    read1.alignment.reference_start = 6;
    read1.alignment.query_sequence  = "TATACCAGTATATATATATATATATATATATATAGGA";
    read1.alignment.reference_end   = 6 + len(read1.alignment.query_sequence);
    read1.alignment.cigartuples     = [(BAM_CMATCH,
                                        len(read1.alignment.query_sequence))
                                      ];

    read2     = PileupReadDummy();
    read2.alignment = AlignedSegmentDummy();
    read2.alignment.reference_start = 6;
    read2.alignment.query_sequence  = "TATACCAGTATATATATATATATATATATATAGGA";
    read2.alignment.reference_end   = 6 + 25 + 2 + 10;
    read2.alignment.cigartuples     = [(BAM_CMATCH, 25),
                                       (BAM_CDEL,2),
                                       (BAM_CMATCH,10)
                                      ];

    read3     = PileupReadDummy();
    read3.alignment = AlignedSegmentDummy();
    read3.alignment.reference_start = 6;
    read3.alignment.query_sequence  = "TATACCAGTATATATATATATATATATATATATAGGA";
    read3.alignment.reference_end   = 6 + 10 + 15 + 2 + 10;
    read3.alignment.cigartuples     = [(BAM_CMATCH,10),
                                       (BAM_CINS,2),
                                       (BAM_CMATCH,15),
                                       (BAM_CDEL,2),
                                       (BAM_CMATCH,10)
                                      ];

    read4     = PileupReadDummy();
    read4.alignment = AlignedSegmentDummy();
    read4.alignment.reference_start = 6;
    read4.alignment.query_sequence  = "TATACCAGTATAGATATATATATATATATATATAGGA";
    read4.alignment.reference_end   = 6 + 12 + 1 + 24;
    read4.alignment.cigartuples     = [(BAM_CMATCH,12),
                                       (BAM_CINS,1),
                                       (BAM_CDEL,1),
                                       (BAM_CMATCH,24)
                                      ];

    read5     = PileupReadDummy();
    read5.alignment = AlignedSegmentDummy();
    read5.alignment.reference_start = 14;
    read5.alignment.query_sequence  = "TATATATATATATATATATATATAGGATACTTTT";
    read5.alignment.reference_end   = 14 + 2 + 2 + 28;
    read5.alignment.cigartuples     = [(BAM_CMATCH,2),
                                       (BAM_CDEL,2),
                                       (BAM_CMATCH,28),
                                       (BAM_CSOFT_CLIP,4),
                                      ];

    container = PileupContainerDummy();
    container.chromosome = '1';
    container.pileupreads = [read1, read2, read3, read4, read5];

    adjustCigartuples(container, reference);

    print("New Cigars are", container.cigartuples);
