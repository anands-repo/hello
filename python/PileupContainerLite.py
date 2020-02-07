"""
This file presents a light version of PileupContainer that doesn't collect reads column-by-column
"""
import pysam
import random
from functools import reduce
from PileupContainer import is_usable_read, AlignedSegmentDummy
import PileupContainer
import logging
import copy
from timeit import default_timer as timer

try:
    profile
except Exception:
    def profile(x):
        return x;


class PileupReadDummy:
    """
    Dummy class for putting AlignedSegment from pileupread.fetch in
    """
    def __init__(self, alignment):
        self.alignment = alignment;


class AlignedSegmentFunctional(AlignedSegmentDummy):
    def __init__(
        self,
        referenceStart,
        referenceEnd,
        cigartuples,
        query_sequence,
        query_qualities,
        query_alignment_sequence,
        query_alignment_qualities,
        query_name,
        is_reverse,
        mapping_quality,
    ):
        super().__init__();
        self.reference_start = referenceStart;
        self.reference_end = referenceEnd;
        self.cigartuples = cigartuples;
        self.query_sequence = query_sequence;
        self.query_qualities = query_qualities if (query_qualities is not None) else list(len(query_sequence) * [40]);
        self.query_alignment_sequence = query_alignment_sequence;
        self.query_alignment_qualities = query_alignment_qualities if (query_alignment_qualities is not None) else list(len(query_alignment_sequence) * [40]);
        self.alignedPairs = None;
        self.query_name = query_name;
        self.is_reverse = is_reverse;
        self.mapping_quality = mapping_quality;

    def get_aligned_pairs(self):
        if self.alignedPairs is None:
            self.alignedPairs = [];
            readCounter = 0;
            refCounter = self.reference_start;

            for operation, length in self.cigartuples:
                if operation in [PileupContainer.BAM_CMATCH, PileupContainer.BAM_CEQUAL, PileupContainer.BAM_CDIFF]:
                    self.alignedPairs.extend(
                        zip(range(readCounter, readCounter + length), range(refCounter, refCounter + length))
                    );
                    readCounter += length;
                    refCounter += length;
                elif operation in [PileupContainer.BAM_CDEL, PileupContainer.BAM_CREF_SKIP]:
                    self.alignedPairs.extend(
                        zip([None for i in range(length)], range(refCounter, refCounter + length))
                    );
                    refCounter += length;
                elif operation in [PileupContainer.BAM_CINS]:
                    self.alignedPairs.extend(
                        zip(range(readCounter, readCounter + length), [None for i in range(length)])
                    );
                    readCounter += length;
                elif operation in [PileupContainer.BAM_CSOFT_CLIP]:
                    self.alignedPairs.extend(
                        zip(range(readCounter, readCounter + length), [None for i in range(length)])
                    );
                    readCounter += length;

        return self.alignedPairs;


def createClippedReadObject(pileupread, leftFlank, rightFlank):
    """
    Clips reads to between left flank and right flank

    :param pileupread: PileupRead/PileupReadDummy
        Read object

    :param leftFlank: int
        Left flanking position

    :param rightFlank: int
        Right flanking position
    """
    cigartuples = [];    
    referenceStart = None;
    referenceEnd = None;
    readSequence = '';
    qualSequence = [];
    alnReadSeq = '';
    alnQualSeq = [];
    readCounter = 0;
    refCounter = pileupread.alignment.reference_start;
    origRefStart = pileupread.alignment.reference_start;
    query_name = pileupread.alignment.query_name if hasattr(pileupread.alignment, 'query_name') else None;
    is_reverse = pileupread.alignment.is_reverse if hasattr(pileupread.alignment, 'is_reverse') else False;

    # Traverse reads and collect data within flank limits
    for i, (operation, length) in enumerate(pileupread.alignment.cigartuples):
        if operation in [PileupContainer.BAM_CMATCH, PileupContainer.BAM_CEQUAL, PileupContainer.BAM_CDIFF]:
            if refCounter < leftFlank:
                # If entire cigartuple falls completely to the left of the left flank, discard it
                if refCounter + length - 1 < leftFlank:
                    refCounter += length;
                    readCounter += length;
                    continue;
                # If the cigartuple falls partially across the left flank boundary, shift it and use it
                else:
                    lengthDiff = leftFlank - refCounter;
                    refCounter += lengthDiff;
                    readCounter += lengthDiff;
                    length -= lengthDiff;

                    # While this happens, the cigartuple may also fall partially to the right-side of the right flank
                    if refCounter + length >= rightFlank:
                        length = rightFlank - refCounter;

            elif refCounter >= rightFlank:
                # If entire cigartuple falls outside of the right flank, discard it (end the iterations)
                break;
            elif leftFlank <= refCounter < refCounter + length <= rightFlank:
                # If entire cigartuple falls within the flanks, do nothing, and use directly
                pass;
            elif leftFlank <= refCounter < rightFlank:
                # The cigartuple falls partially within the flanks with the right side jutting out
                length = rightFlank - refCounter;
            else:
                raise ValueError("Unknown cigar condition for match");

            cigartuples.append((operation, length));            

            readSequence += pileupread.alignment.query_sequence[readCounter:readCounter + length];
            qualSequence += pileupread.alignment.query_qualities[readCounter:readCounter + length];
            alnReadSeq += pileupread.alignment.query_sequence[readCounter:readCounter + length];
            alnQualSeq += pileupread.alignment.query_qualities[readCounter:readCounter + length];

            if referenceStart is None:
                referenceStart = refCounter;

            refCounter += length;
            readCounter += length;
            referenceEnd = refCounter;

        elif operation in [PileupContainer.BAM_CDEL, PileupContainer.BAM_CREF_SKIP]:
            if refCounter < leftFlank + 1:
                # If a deletion only partially overlaps at the left-side, or if the first base within
                # the (leftFlank - rightFlank) frame is a deletion, discard it. In these cases the alignment
                # will start with a read deletion. It doesn't make sense to have an alignment starting
                # with a deletion at the left. Also, hotspot analysis must have precluded such
                # deletions only if they didn't matter (well, with a high degree of certainty anyway)
                refCounter += length;
                continue;

            elif refCounter + length >= rightFlank:
                # Similar arguments hold for deletions partially overlapping the right side, or when the
                # alignment within the (leftFlank-rightFlank) region ends with a deletion
                refCounter += length;
                break;

            elif leftFlank <= refCounter < refCounter + length <= rightFlank:
                # If entire cigartuple falls within the flanks, do nothing, and use directly
                pass;
            else:
                raise ValueError("Unknown cigar condition for deletion");

            cigartuples.append((operation, length));

            if referenceStart is None:
                raise ValueError("Alignment starts with a deletion");

            refCounter += length;

        elif operation in [PileupContainer.BAM_CINS]:
            if refCounter <= leftFlank:
                # Insertions are by definition single base events in the reference.
                # Hence we can safely discard any that occurs before the leftFlank.
                # Note that when refCounter = leftFlank, it means the inserted bases
                # "take off" from the position to the left of the leftFlank.
                readCounter += length;
                continue;
            elif refCounter >= rightFlank:
                # We allow inserted bases at the last position of the alignment
                # otherwise we should have rightFlank - 1, since rightFlank -1 is the last
                # position of the alignment
                readCounter += length;
                continue;

            if referenceStart is None:
                # Oddly enough, this condition happens in the BAM file
                # In this case, convert the leading insertion to a soft-clip
                operation = PileupContainer.BAM_CSOFT_CLIP;

            cigartuples.append((operation, length));

            readSequence += pileupread.alignment.query_sequence[readCounter:readCounter + length];
            qualSequence += pileupread.alignment.query_qualities[readCounter:readCounter + length];
            alnReadSeq += pileupread.alignment.query_sequence[readCounter:readCounter + length];
            alnQualSeq += pileupread.alignment.query_qualities[readCounter:readCounter + length];
            readCounter += length;
            referenceEnd = refCounter;  # We allow alignments to end with insertions

        elif operation in [PileupContainer.BAM_CSOFT_CLIP]:
            # Accept a soft-clip only if the "landing" or "take-off" base of the soft clip
            # falls within the flanks. Soft-clip doesn't determine alignment start or alignment end
            if i == 0:
                if leftFlank < origRefStart < rightFlank:
                    cigartuples.append((operation, length));
                    readSequence += pileupread.alignment.query_sequence[readCounter:readCounter + length];
                    qualSequence += pileupread.alignment.query_qualities[readCounter:readCounter + length];
            else:
                if leftFlank < refCounter < rightFlank:
                    cigartuples.append((operation, length));
                    readSequence += pileupread.alignment.query_sequence[readCounter:readCounter + length];
                    qualSequence += pileupread.alignment.query_qualities[readCounter:readCounter + length];

            readCounter += length;

        if refCounter >= rightFlank:
            break;

    newAlignedSegment = AlignedSegmentFunctional(
        referenceStart,
        referenceEnd,
        cigartuples,
        readSequence,
        qualSequence,
        alnReadSeq,
        alnQualSeq,
        query_name,
        is_reverse,
    );

    return PileupReadDummy(newAlignedSegment);


@profile
def strictClipFn(cigars, limitLength, left):
    """
    Determines the cigar tuples after clipping, and provides the number of bases to retain
    in both query_sequence and query_alignment_sequence

    :param cigars: list
        List of cigartuples to limit

    :param limitLength: int
        Length of the sequences to be limited to

    :param left: bool
        Whether this is a left flank or a right flank

    :return: tuple
        (cigars to keep, num bases pruned from query sequence, num bases pruned from aligned sequence, #reference pruned)
    """
    cigartuples = cigars if not left else list(reversed(cigars));
    readCounter = 0;
    cigarsToKeep = [];
    cigarsToDiscard = [];

    # Determine cigartuples to keep
    for i, (operation, length) in enumerate(cigartuples):
        readCounterNew = readCounter + \
            (length if operation in [
                PileupContainer.BAM_CMATCH,
                PileupContainer.BAM_CINS,
                PileupContainer.BAM_CSOFT_CLIP,
                PileupContainer.BAM_CEQUAL,
                PileupContainer.BAM_CDIFF
            ] else 0);

        if readCounter <= limitLength < readCounterNew:
            # We have chanced upon the point at which the read should be left-truncated
            cigarToKeep = (operation, limitLength - readCounter + 1);
            cigarToDiscard = (operation, length - cigarToKeep[1]);

            if cigarToKeep[1] > 0:
                cigarsToKeep.append(cigarToKeep);

            if cigarToDiscard[1] > 0:
                cigarsToDiscard.append(cigarToDiscard);

            cigarsToDiscard.extend(cigartuples[i + 1:]);

            break;
        else:
            cigarsToKeep.append((operation, length));

        readCounter = readCounterNew;

    cigarsToKeep = cigarsToKeep if not left else list(reversed(cigarsToKeep));

    # Adjust cigarsToKeep so that a leading, or trailing insertion is converted to a soft-clip
    # for left or right clipping respectively
    # Record this number which may be used to change reference start and reference end
    # for aligned segments
    numInsToSoftClip = 0;

    if left:
        if cigarsToKeep[0][0] == PileupContainer.BAM_CINS:
            newCigar = (PileupContainer.BAM_CSOFT_CLIP, cigarsToKeep[0][1]);
            cigarsToKeep[0] = newCigar;
            numInsToSoftClip += newCigar[1];
    else:
        if cigarsToKeep[-1][0] == PileupContainer.BAM_CINS:
            newCigar = (PileupContainer.BAM_CSOFT_CLIP, cigarsToKeep[-1][1]);
            cigarsToKeep[-1] = newCigar;
            numInsToSoftClip += newCigar[1];

    # Determine number of bases that are pruned from query_sequence
    numPrunedQuerySequence = 0;

    for operation, length in cigarsToDiscard:
        if operation in [
            PileupContainer.BAM_CMATCH,
            PileupContainer.BAM_CINS,
            PileupContainer.BAM_CSOFT_CLIP,
            PileupContainer.BAM_CEQUAL,
            PileupContainer.BAM_CDIFF
        ]:
            numPrunedQuerySequence += length;

    # Determine number of bases that are pruned from query_alignment_sequence (BAM_CSOFT_CLIP entries in original cigar need not be included)
    # However, do include number of insertions that were converted to soft-clips
    numPrunedQueryAlignmentSequence = sum(
        [length for operation, length in cigarsToDiscard if operation in [
            PileupContainer.BAM_CMATCH,
            PileupContainer.BAM_CINS,
            PileupContainer.BAM_CEQUAL,
            PileupContainer.BAM_CDIFF]]
    );
    numPrunedQueryAlignmentSequence += numInsToSoftClip;

    # Determine the number of reference bases pruned
    numReferenceBasesPruned = 0;

    for operation, length in cigarsToDiscard:
        if operation in [
            PileupContainer.BAM_CMATCH,
            PileupContainer.BAM_CEQUAL,
            PileupContainer.BAM_CDIFF,
            PileupContainer.BAM_CDEL,
            PileupContainer.BAM_CREF_SKIP,
        ]:
            numReferenceBasesPruned += length;

    return cigarsToKeep, numPrunedQuerySequence, numPrunedQueryAlignmentSequence, numReferenceBasesPruned;


@profile
def strictClipRead(pileupread, position, left, flankLength=150):
    """
    Strictly clip the flank length of a read to the given flank length

    :param pileupread: PileupReadDummy
        PileupRead object which needs to be trimmed

    :param position: int
        Position (in the reference) from which to measure flank length

    :param left: bool
        Whether we are pruning the left flank or the right flank

    :param flankLength: int
        Length of the flank
    """
    alignedSegment = pileupread.alignment;

    # Sanity check: is position with reference_start->reference_end
    if not (alignedSegment.reference_start <= position < alignedSegment.reference_end):
        return;

    # Locate the cigartuple that indicates "position", segment cigartuple at the position
    refCounter = alignedSegment.reference_start;
    leftCigars = [];
    rightCigars = [];

    for i, (operation, length) in enumerate(alignedSegment.cigartuples):
        refCounterAfter = refCounter + \
            (length if operation in [
                PileupContainer.BAM_CDEL,
                PileupContainer.BAM_CREF_SKIP,
                PileupContainer.BAM_CMATCH,
                PileupContainer.BAM_CEQUAL,
                PileupContainer.BAM_CDIFF
            ] else 0);

        if (refCounter <= position < refCounterAfter):
            cigarLeftPart = (operation, position - refCounter + 1);
            cigarRightPart = (operation, length - cigarLeftPart[1]);

            if cigarLeftPart[1] > 0:
                leftCigars.append(cigarLeftPart);

            if cigarRightPart[1] > 0:
                rightCigars.append(cigarRightPart);

            rightCigars.extend(list(alignedSegment.cigartuples[i + 1:]));

            break;
        else:
            leftCigars.append((operation, length));

        refCounter = refCounterAfter;

    modifyCigars = False;

    if left and (len(leftCigars) > 0):
        leftCigarsToKeep, numPrunedQuerySequenceLeft, numPrunedQueryAlignmentSequenceLeft, numReferencePrunedLeft = \
            strictClipFn(leftCigars, flankLength, left=True);
        alignedSegment.reference_start += numReferencePrunedLeft;
        alignedSegment.query_alignment_sequence = alignedSegment.query_alignment_sequence[numPrunedQueryAlignmentSequenceLeft:];
        alignedSegment.query_alignment_qualities = alignedSegment.query_alignment_qualities[numPrunedQueryAlignmentSequenceLeft:];
        alignedSegment.query_sequence = alignedSegment.query_sequence[numPrunedQuerySequenceLeft:];
        alignedSegment.query_qualities = alignedSegment.query_qualities[numPrunedQuerySequenceLeft:];
        rightCigarsToKeep = rightCigars;
        modifyCigars = True;
    elif (len(rightCigars) > 0):
        rightCigarsToKeep, numPrunedQuerySequenceRight, numPrunedQueryAlignmentSequenceRight, numReferencePrunedRight = \
            strictClipFn(rightCigars, flankLength, left=False);

        alignedSegment.reference_end -= numReferencePrunedRight;

        if numPrunedQuerySequenceRight > 0:
            alignedSegment.query_sequence = alignedSegment.query_sequence[:-numPrunedQuerySequenceRight];
            alignedSegment.query_qualities = alignedSegment.query_qualities[:-numPrunedQuerySequenceRight];

        if numPrunedQueryAlignmentSequenceRight > 0:
            alignedSegment.query_alignment_sequence = alignedSegment.query_alignment_sequence[:-numPrunedQueryAlignmentSequenceRight];
            alignedSegment.query_alignment_qualities = alignedSegment.query_alignment_qualities[:-numPrunedQueryAlignmentSequenceRight];

        leftCigarsToKeep = leftCigars;
        modifyCigars = True;

    # Merge left and right cigartuples
    if modifyCigars:
        if (len(leftCigarsToKeep) > 0) and (len(rightCigarsToKeep) > 0):
            cigarCenter = [leftCigarsToKeep[-1], rightCigarsToKeep[0]];
            (operation0, length0), (operation1, length1) = cigarCenter;

            if operation0 == operation1:
                cigarCenter = [(operation0, length0 + length1)];
        else:
            cigarCenter = [];

            if len(leftCigarsToKeep) > 0:
                cigarCenter = [leftCigarsToKeep[-1]];

            if len(rightCigarsToKeep) > 0:
                cigarCenter = [rightCigarsToKeep[0]];

        alignedSegment.cigartuples = leftCigarsToKeep[:-1] + cigarCenter + rightCigarsToKeep[1:];


class PileupContainerLite:
    execTime = 0;
    """
    Object acts as interface for pileup locations, collecting filtered reads from the location. Object can
    subsample reads as necessary from the location. This is a lite version of PileupContainer which does
    column-by-column analysis. This object can also be reused to obtain caching performance.
    """
    def __init__(self, bamfile, chromosome, position, span, maxNumReads=10000, clipReads=False, clipFlank=100):
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
            limit to this number if there are too many reads (similar to DeepVariant)

        :param clipReads: bool
            Clip reads to +/- flanks

        :param clipFlank: int
            Flank length to clip reads to
        """
        execStart = timer();
        self.read_list = [];
        self.bamname = bamfile;
        self.chromosome = chromosome;
        self.position = position;
        self.span = span;
        self.avg_cov = 0;
        self.maxNumReads = maxNumReads;
        self.clipReads = clipReads;
        self.clipFlank = clipFlank;

        if type(bamfile) is str:
            self.bhandle = pysam.AlignmentFile(bamfile, 'rb');
        else:
            self.bhandle = bamfile;

        self.__cigartuples = None;
        self.__get_reads();
        execStop = timer();
        PileupContainerLite.execTime += (execStop - execStart);

    @profile
    def __get_reads(self):
        records_with_name = set();
        itemEncountered = 0;  # Counter for reservoir sampling

        for alignment in self.bhandle.fetch(
            self.chromosome,
            self.position,
            self.position + self.span,
        ):
            pileupread = PileupReadDummy(alignment);
            query_name = pileupread.alignment.query_name;
            direction = pileupread.alignment.is_reverse;

            if query_name is None:
                raise ValueError("No query name found! BAM file not compatible");

            if not is_usable_read(pileupread):
                continue;

            if (query_name, direction) in records_with_name:
                continue;

            records_with_name.add((query_name, direction));

            pKeep = self.maxNumReads / (itemEncountered + 1);

            if random.uniform(0, 1) < pKeep:
                if self.clipReads:
                    newAlignedSegment = AlignedSegmentFunctional(
                        pileupread.alignment.reference_start,
                        pileupread.alignment.reference_end,
                        pileupread.alignment.cigartuples,
                        pileupread.alignment.query_sequence,
                        pileupread.alignment.query_qualities,
                        pileupread.alignment.query_alignment_sequence,
                        pileupread.alignment.query_alignment_qualities,
                        pileupread.alignment.query_name,
                        pileupread.alignment.is_reverse,
                        pileupread.alignment.mapping_quality,
                    );

                    pileupread = PileupReadDummy(newAlignedSegment);

                    # New test code for strict read clipping
                    strictClipRead(pileupread, self.position, left=True, flankLength=self.clipFlank);
                    strictClipRead(pileupread, self.position + self.span, left=False, flankLength=self.clipFlank);

                if len(self.read_list) < self.maxNumReads:
                    self.read_list.append(pileupread);
                else:
                    indexToReplace = random.sample(range(len(self.read_list)), 1)[0];
                    self.read_list[indexToReplace] = pileupread;

            itemEncountered += 1;

    def subsample(self, rate):
        if rate is not None:
            self.read_list = random.sample(
                self.read_list,
                int(rate * len(self.read_list))
            );

    @property
    def pileupreads(self):
        return self.read_list;

    @pileupreads.setter
    def pileupreads(self, item):
        self.read_list = item;

    @property
    def referenceStart(self):
        return min([r.alignment.reference_start for r in self.read_list]);

    @property
    def referenceEnd(self):
        return max([r.alignment.reference_end for r in self.read_list]);

    @property
    def cigartuples(self):
        if self.__cigartuples is None:
            return [r.alignment.cigartuples for r in self.read_list];
        else:
            return self.__cigartuples;

    @cigartuples.setter
    def cigartuples(self, _cigartuples):
        self.__cigartuples = _cigartuples;

    @property
    def alignedPairs(self):
        if not hasattr(self, '__alignedPairs'):
            return [r.alignment.get_aligned_pairs() for r in self.read_list];
        else:
            return self.__alignedPairs;

    @alignedPairs.setter
    def alignedPairs(self, _alignedPairs):
        self.__alignedPairs = _alignedPairs;

    def intersect(self, start, stop):
        """
        Create a copy of this object and intersect each read with start, stop coordinates (or intersection)
        If read clipping is in force, do reclipping

        :param start: int
            Start coordinate to intersect

        :param stop: int
            Stop coordinate to intersect

        :return: PileupContainerLite
            A copy of the object modified as necessary
        """
        newObj = copy.deepcopy(self);
        newObj.position = start;
        newObj.span = stop - start;
        newReadList = [];

        for read in newObj.read_list:
            alignedPairs = read.alignment.get_aligned_pairs();
            refPositionsAligned = set([a[1] for a in alignedPairs if a[1] is not None]);
            minPosition = min(refPositionsAligned);
            maxPosition = max(refPositionsAligned);

            if (minPosition <= start <= maxPosition) or (minPosition <= stop - 1 <= maxPosition):
                # Accept read if either condition holds, and reclip reads if necessary
                if newObj.clipReads:
                    strictClipRead(read, start, left=True, flankLength=newObj.clipFlank);
                    strictClipRead(read, stop, left=False, flankLength=newObj.clipFlank);

                newReadList.append(read);

        newObj.read_list = newReadList;

        return newObj;


if __name__ == "__main__":
    """
    Test read clipping functionality
    """
    # Positions   0         10           23                                         66
    # Reference = ACATTAACGATCGACTGCACATCGACATCGACTAGCATACACTAGCACATCGACATCACGATCCAC
    # Region    =          |            |
    # Read0     = ACATTAACG---GACTGCA
    # Read1     =             GACTGCACATCGttttt
    # Read2     =       ACGATCGACTGCACATCGACA
    # Read3     =       ACGATCGACTGCACA----CA
    # Read4     =       ACIATCGACTGCACAiCGA;   I = GT, i = TA (both should be neglected)
    # Read5     =       ACGATCGA-TGCAIATCGACA; I = CT

    # Create Read0
    alignment = AlignedSegmentDummy();
    alignment.query_sequence = "ACATTAACG---GACTGCA".replace("-","");
    alignment.query_qualities = list([30] * len(alignment.query_sequence));
    alignment.cigartuples = [(PileupContainer.BAM_CMATCH, 9), (PileupContainer.BAM_CDEL, 3), (PileupContainer.BAM_CMATCH, 7)];
    alignment.reference_start = 0;
    read0 = PileupReadDummy(alignment);

    # Create Read1
    alignment = AlignedSegmentDummy();
    alignment.query_sequence = "GACTGCACATCGttttt".upper();
    alignment.query_qualities = list([30] * len(alignment.query_sequence));
    alignment.cigartuples = [(PileupContainer.BAM_CMATCH, 12), (PileupContainer.BAM_CSOFT_CLIP, 5)];
    alignment.reference_start = 12;
    read1 = PileupReadDummy(alignment);

    # Create Read2
    alignment = AlignedSegmentDummy();
    alignment.query_sequence = "ACGATCGACTGCACATCGACA";
    alignment.query_qualities = list([30] * len(alignment.query_sequence));
    alignment.cigartuples = [(PileupContainer.BAM_CMATCH, 21)];
    alignment.reference_start = 6;
    read2 = PileupReadDummy(alignment);

    # Create Read3
    alignment = AlignedSegmentDummy();
    alignment.query_sequence = "ACGATCGACTGCACA----CA".replace("-", "");
    alignment.query_qualities = list([30] * len(alignment.query_sequence));
    alignment.cigartuples = [(PileupContainer.BAM_CMATCH, 15), (PileupContainer.BAM_CDEL, 4), (PileupContainer.BAM_CMATCH, 2)];
    alignment.reference_start = 6;
    read3 = PileupReadDummy(alignment);

    # Create Read4
    alignment = AlignedSegmentDummy();
    alignment.query_sequence = "ACIATCGACTGCACAiCGA".replace("I", "GT").replace("i", "TA");
    alignment.query_qualities = list([30] * len(alignment.query_sequence));
    alignment.cigartuples = [
        (PileupContainer.BAM_CMATCH, 3),
        (PileupContainer.BAM_CINS, 1),
        (PileupContainer.BAM_CMATCH, 13),
        (PileupContainer.BAM_CINS, 1),
        (PileupContainer.BAM_CMATCH, 3)
    ];
    alignment.reference_start = 6;
    read4 = PileupReadDummy(alignment);

    # Create Read5
    alignment = AlignedSegmentDummy();
    alignment.query_sequence = "ACGATCGA-TGCAIATCGACA".replace("-", "").replace("I", "CT");
    alignment.query_qualities = list([30] * len(alignment.query_sequence));
    alignment.cigartuples = [
        (PileupContainer.BAM_CMATCH, 8),
        (PileupContainer.BAM_CDEL, 1),
        (PileupContainer.BAM_CMATCH, 5),
        (PileupContainer.BAM_CINS, 1),
        (PileupContainer.BAM_CMATCH, 7)
    ];
    alignment.reference_start = 6;
    read5 = PileupReadDummy(alignment);

    # Call function and print output
    def test(read):
        result = createClippedReadObject(read, 9, 23);
        print("Results");
        print(result.alignment.query_sequence, result.alignment.query_alignment_sequence);
        print(result.alignment.query_qualities, result.alignment.query_alignment_qualities);
        print(result.alignment.reference_start, result.alignment.reference_end);
        print(result.alignment.cigartuples, result.alignment.get_aligned_pairs());
        print();

    test(read0);
    test(read1);
    test(read2);
    test(read3);
    test(read4);
    test(read5);

    # Test with real data
    container = PileupContainerLite(
        "/root/storage/HG002PacbioSequel2/HG002.SequelII.pbmm2.hs37d5.whatshap.haplotag.RTG.10x.trio.bam",
        '20',
        74730,
        1,
        clipReads=True,
        clipFlank=10,
    );

    for read in container.pileupreads:
        print("Read name: ", read.alignment.query_name, ", cigar :", read.alignment.cigartuples, ", reference start: ", read.alignment.reference_start, ", reference end:", read.alignment.reference_end, ", aligned pairs: ", read.alignment.get_aligned_pairs());
