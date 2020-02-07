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

CHUNK_SIZE_ILLUMINA = 400;
CHUNK_SIZE_PACBIO = 10000;
MAX_NUM_READS_ILLUMINA = 10000;
MAX_NUM_READS_PACBIO = 1000;

try:
    profile
except Exception:
    def profile(x):
        return x;


@profile
def doOneChunkSingle(chromosome, begin, end, positions, readFactory, pacbio=False):
    """
    Perform hotspot detection for a single chunk for a single bam file

    :param chromosome: str
        Chromosome in which to perform analysis

    :param begin: int
        Beginning of region

    :param end: int
        End of region

    :param positions: collections.OrderedDict
        Current set of positions into which to add differing
        regions from this

    :param readFactory: str
        Read factory

    :param pacbio: bool
        Whether the reads are pacbio reads
    """
    container = readFactory(
        chromosome, begin, end
    );

    if len(container.pileupreads) == 0:
        return;

    try:
        searcher = AlleleSearcherLite(
            container, begin, end, cache, strict=False
        );
    except LocationOutOfBounds:
        logging.warning("Out of bounds locations found for chunk %s, %d, %d" % (chromosome, begin, end));
        return;

    if pacbio:
        # Raise indel threshold for PacBio reads
        # for hotspot detection, otherwise too many erroneous locations
        # get involved
        logging.debug("Setting indel threshold for pacbio reads");
        searcher.searcher.indelThreshold = 0.12;

    for left, right in searcher.differingRegions:
        for j in range(left, right):
            positions[j] = None;


def hotspotGeneratorSingle(
    readFactory,
    chromosome,
    start,
    stop,
    chunkSize,
    pacbio=False,
):
    numChunks = (stop - start) // chunkSize;

    if numChunks * chunkSize < (stop - start):
        numChunks += 1;

    positions = collections.OrderedDict();

    for i in range(numChunks):
        begin_ = start + chunkSize * i;
        end_ = min(begin_ + chunkSize, stop);
        doOneChunkSingle(chromosome, begin_, end_, positions, readFactory, pacbio=pacbio);

    sortedPositions = sorted(list(positions.keys()));

    for p in sortedPositions:
        yield p;


def hotspotGeneratorHybrid(
    readFactory0,
    readFactory1,
    chromosome,
    start,
    stop,
    chunkSize0,
    chunkSize1,
):
    numChunks = (stop - start) // chunkSize1;

    if numChunks * chunkSize1 < (stop - start):
        numChunks += 1;

    positions = collections.OrderedDict();

    # Perform outer loop over bam1 (PacBio reads)
    for i in range(numChunks):
        begin_ = start + chunkSize1 * i;
        end_ = min(begin_ + chunkSize1, stop);

        # Collect all TGS positions
        doOneChunkSingle(chromosome, begin_, end_, positions, readFactory1,  pacbio=True);

        # Collect NGS positions using an iterator
        ngsIter = hotspotGeneratorSingle(
            readFactory0,
            chromosome,
            begin_,
            end_,
            chunkSize0,
            pacbio=False,
        );

        for p_ in ngsIter:
            positions[p_] = None;

    sortedPositions = sorted(list(positions.keys()));

    for p in sortedPositions:
        yield p;


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hotspot detection using C++ AlleleSearcher class");

    parser.add_argument(
        "--bam",
        help="Comma-separated list of BAM files from which to call hotspots",
        required=True
    );

    parser.add_argument(
        "--ref",
        help="Reference cache location",
        required=True
    );

    parser.add_argument(
        "--region",
        help="Chromosome,start,stop or simply Chromosome",
        required=True
    );

    parser.add_argument(
        "--pacbio",
        help="Indicate that we are using PacBio reads (for a single file)",
        default=False,
        action="store_true"
    );

    parser.add_argument(
        "--output",
        help="Path to the output file",
        required=True
    );

    parser.add_argument(
        "--debug",
        help="Display debug messages",
        action="store_true",
        default=False,
    );

    cache = None;

    @profile
    def main():
        global cache;
        args = parser.parse_args();

        # logging.basicConfig(level=(logging.DEBUG if args.debug else logging.INFO), format='%(asctime)-15s %(message)s');
        logging.basicConfig(level=(logging.INFO if not args.debug else logging.DEBUG), format='%(asctime)-15s %(message)s');
        libCallability.initLogging(args.debug);

        logging.info("Started script");

        # cache = ReferenceCache(database=args.ref);
        cache = PySamFastaWrapper(args.ref);

        if len(args.region.split(",")) == 1:
            chromosome = args.region;
            cache.chrom = chromosome;
            start = 0;
            stop = len(cache);
        else:
            chromosome, start, stop = args.region.split(",");
            start = int(start);
            stop = int(stop);

        bams = args.bam.split(",");

        # Setup searcher factories
        if len(bams) > 1:
            args.bam, args.bam2 = bams;
            args.bam = PileupDataTools.ReadSampler(
                args.bam,
                MAX_NUM_READS_ILLUMINA,
                chrPrefix="",
                pacbio=False,
                noClip=True,
                prorateReadsToRetain=False,
            );
            args.bam2 = PileupDataTools.ReadSampler(
                args.bam2,
                MAX_NUM_READS_PACBIO,
                chrPrefix="",
                pacbio=True,
                noClip=True,
                prorateReadsToRetain=False,
            );
        else:
            args.bam = bams[0];
            args.bam2 = None;
            args.bam = PileupDataTools.ReadSampler(
                args.bam,
                MAX_NUM_READS_ILLUMINA if not args.pacbio else MAX_NUM_READS_PACBIO,
                chrPrefix="",
                pacbio=args.pacbio,
                noClip=True,
                prorateReadsToRetain=False,
            );

        if args.bam2 is None:
            iterator = hotspotGeneratorSingle(
                args.bam,
                chromosome,
                start,
                stop,
                CHUNK_SIZE_PACBIO if args.pacbio else CHUNK_SIZE_ILLUMINA,
                pacbio=args.pacbio,
            );
        else:
            iterator = hotspotGeneratorHybrid(
                args.bam,
                args.bam2,
                chromosome,
                start,
                stop,
                CHUNK_SIZE_ILLUMINA,
                CHUNK_SIZE_PACBIO,
            );

        with open(args.output, 'w') as ohandle:
            for i, item in enumerate(iterator):
                ohandle.write(
                    str({'chromosome': chromosome, 'position': item}) + '\n'
                );

                if (i + 1) % 100 == 0:
                    logging.info("Completed %d chunks" % (i + 1));

        logging.info("Completed running the script");

    main();
