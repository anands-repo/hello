# © 2019 University of Illinois Board of Trustees.  All rights reserved
"""
New hotspot detection method that uses the C++ AlleleSearcher class
"""
import argparse
from AlleleSearcherLite import AlleleSearcherLite, LocationOutOfBounds
from ReferenceCache import ReferenceCache
from PileupContainerLite import PileupContainerLite
import logging
import libCallability
import collections
import PileupDataTools
from PySamFastaWrapper import PySamFastaWrapper
import math

CHUNK_SIZE_ILLUMINA = 400
CHUNK_SIZE_PACBIO = 10000
MAX_NUM_READS_ILLUMINA = 10000
MAX_NUM_READS_PACBIO = 1000
_DEFAULT_HYBRID_HOTSPOT = False
_DEFAULT_MIN_MAPQ = 10
_DEFAULT_Q_THRESHOLD = 10

try:
    profile
except Exception:
    def profile(x):
        return x


@profile
def doOneChunk(
    chromosome,
    begin,
    end,
    positions,
    readFactory,
    cache,
    pacbio=False,
    q_threshold=_DEFAULT_Q_THRESHOLD,
    min_mapq=_DEFAULT_MIN_MAPQ,
    hybrid_hotspot=_DEFAULT_HYBRID_HOTSPOT,
):
    """
    Perform hotspot detection for a single chunk

    :param chromosome: str
        Chromosome in which to perform analysis

    :param begin: int
        Beginning of region

    :param end: int
        End of region

    :param positions: collections.OrderedDict
        Current set of positions into which to add differing
        regions from this

    :param readFactory: list/str
        Read factory or list of read factories in case of hybrid mode

    :param cache: PySamFastaWrapper
        Reference reader

    :param pacbio: bool
        Whether the reads are pacbio reads

    :param q_threshold: int
        Quality score cutoff

    :param min_mapq: int
        Minimum mapping quality

    :param hybrid_hotspot: bool
        Whether to use hybrid_hotspot or not
    """
    if type(readFactory) is list:
        container = [
            rF(
                chromosome, begin, end
            ) for rF in readFactory
        ]
    else:
        container = [readFactory(chromosome, begin, end)]

    if all(len(c.pileupreads) == 0 for c in container):
        return

    try:
        searcher = AlleleSearcherLite(
            container, begin, end, cache, strict=False, pacbio=pacbio,
            hybrid_hotspot=hybrid_hotspot, q_threshold=q_threshold, mapq_threshold=min_mapq,
        )

        if hasattr(searcher, 'searcher'):
            assert(searcher.searcher.check_q_threshold(q_threshold)), "Q threshold not set correctly"
            assert(searcher.searcher.check_mapq_threshold(min_mapq)), "MAPQ threshold not set correctly"

    except LocationOutOfBounds:
        logging.warning("Out of bounds locations found for chunk %s, %d, %d" % (chromosome, begin, end))
        return

    for left, right in searcher.differingRegions:
        for j in range(left, right):
            positions[j] = None


def hotspotGeneratorSingle(
    readFactory,
    cache,
    chromosome,
    start,
    stop,
    chunkSize,
    pacbio=False,
    q_threshold=_DEFAULT_Q_THRESHOLD,
    min_mapq=_DEFAULT_MIN_MAPQ,
    hybrid_hotspot=_DEFAULT_HYBRID_HOTSPOT,
):
    numChunks = math.ceil((stop - start) / chunkSize)
    positions = collections.OrderedDict()

    for i in range(numChunks):
        begin_ = start + chunkSize * i
        end_ = min(begin_ + chunkSize, stop)
        doOneChunk(
            chromosome,
            begin_,
            end_,
            positions,
            readFactory,
            cache,
            pacbio=pacbio,
            q_threshold=q_threshold,
            min_mapq=min_mapq,
            hybrid_hotspot=hybrid_hotspot,
        )

    sortedPositions = sorted(list(positions.keys()))

    for p in sortedPositions:
        yield p


def hotspotGeneratorHybrid(
    readFactory0,
    readFactory1,
    cache,
    chromosome,
    start,
    stop,
    chunkSize0,
    chunkSize1,
    q_threshold=_DEFAULT_Q_THRESHOLD,
    min_mapq=_DEFAULT_MIN_MAPQ,
    hybrid_hotspot=_DEFAULT_HYBRID_HOTSPOT,
):
    numChunks = math.ceil((stop - start) / chunkSize1)
    positions = collections.OrderedDict()

    # Perform outer loop over bam1 (PacBio reads)
    for i in range(numChunks):
        begin_ = start + chunkSize1 * i
        end_ = min(begin_ + chunkSize1, stop)
        doOneChunk(
            chromosome,
            begin_,
            end_,
            positions,
            [readFactory0, readFactory1],
            cache,
            pacbio=False,
            q_threshold=q_threshold,
            min_mapq=min_mapq,
            hybrid_hotspot=hybrid_hotspot,
        )

    sortedPositions = sorted(list(positions.keys()))

    for p in sortedPositions:
        yield p


def main(args):
    logging.info("Started script")

    cache = PySamFastaWrapper(args.ref)

    if len(args.region.split(",")) == 1:
        chromosome = args.region
        cache.chrom = chromosome
        start = 0
        stop = len(cache)
    else:
        chromosome, start, stop = args.region.split(",")
        start = int(start)
        stop = int(stop)

    bams = args.bam.split(",")

    if len(bams) > 1:
        args.bam, args.bam2 = bams
        args.bam = PileupDataTools.ReadSampler(
            args.bam,
            MAX_NUM_READS_ILLUMINA,
            chrPrefix="",
            pacbio=False,
            noClip=True,
            prorateReadsToRetain=False,
        )
        args.bam2 = PileupDataTools.ReadSampler(
            args.bam2,
            MAX_NUM_READS_PACBIO,
            chrPrefix="",
            pacbio=True,
            noClip=True,
            prorateReadsToRetain=False,
        )
    else:
        args.bam = bams[0]
        args.bam2 = None
        args.bam = PileupDataTools.ReadSampler(
            args.bam,
            MAX_NUM_READS_ILLUMINA if not args.pacbio else MAX_NUM_READS_PACBIO,
            chrPrefix="",
            pacbio=args.pacbio,
            noClip=True,
            prorateReadsToRetain=False,
        )

    if args.bam2 is None:
        iterator = hotspotGeneratorSingle(
            args.bam,
            cache,
            chromosome,
            start,
            stop,
            CHUNK_SIZE_PACBIO if args.pacbio else CHUNK_SIZE_ILLUMINA,
            pacbio=args.pacbio,
            q_threshold=args.q_threshold,
            min_mapq=args.mapq_threshold,
            hybrid_hotspot=args.hybrid_hotspot,
        )
    else:
        iterator = hotspotGeneratorHybrid(
            args.bam,
            args.bam2,
            cache,
            chromosome,
            start,
            stop,
            CHUNK_SIZE_ILLUMINA,
            CHUNK_SIZE_PACBIO,
            q_threshold=args.q_threshold,
            min_mapq=args.mapq_threshold,
            hybrid_hotspot=args.hybrid_hotspot,
        )

    with open(args.output, 'w') as ohandle:
        for i, item in enumerate(iterator):
            ohandle.write(
                str({'chromosome': chromosome, 'position': item}) + '\n'
            )

            if (i + 1) % 100 == 0:
                logging.info("Completed %d chunks" % (i + 1))

    logging.info("Completed running the script")

    return args.output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hotspot detection using C++ AlleleSearcher class")

    parser.add_argument(
        "--bam",
        help="Comma-separated list of BAM files from which to call hotspots",
        required=True
    )

    parser.add_argument(
        "--ref",
        help="Reference cache location",
        required=True
    )

    parser.add_argument(
        "--region",
        help="Chromosome,start,stop or simply Chromosome",
        required=True
    )

    parser.add_argument(
        "--pacbio",
        help="Indicate that we are using PacBio reads (for a single file)",
        default=False,
        action="store_true"
    )

    parser.add_argument(
        "--output",
        help="Path to the output file",
        required=True
    )

    parser.add_argument(
        "--debug",
        help="Display debug messages",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--hybrid_hotspot",
        help="Enable hybrid hotspot detection",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--q_threshold",
        help="Quality score threshold",
        default=_DEFAULT_Q_THRESHOLD,
        type=int        
    )

    parser.add_argument(
        "--mapq_threshold",
        help="Mapping quality threshold",
        default=_DEFAULT_MIN_MAPQ,
        type=int,
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=(logging.INFO if not args.debug else logging.DEBUG),
        format='%(asctime)-15s %(message)s')
    libCallability.initLogging(args.debug)

    main(args)
