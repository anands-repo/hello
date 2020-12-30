# © 2019 University of Illinois Board of Trustees.  All rights reserved
"""
New hotspot detection method that uses the C++ AlleleSearcher class
"""
import argparse
#from AlleleSearcher import AlleleSearcher
from ReferenceCache import ReferenceCache
from PileupContainerLite import PileupContainerLite
from PileupContainer import adjustCigartuples
import logging
import libCallability


def combineOverlapping(regions):
    """
    Combine overlapping clusters into one

    :param regions: list
        List of differing regions

    :return: list
        Combined region
    """
    unclustered = [];
    cluster = [];

    def clusterToRegion(cluster):
        small = min([c[0] for c in cluster]);
        large = max([c[-1] for c in cluster]);
        return small, large;

    for (start, stop) in regions:
        if len(cluster) == 0:
            cluster.append((start, stop));
        else:
            if max([c[1] for c in cluster]) > start:
                cluster.append((start, stop));
            else:
                small, large = clusterToRegion(cluster);
                unclustered.append((small, large));
                cluster = [(start, stop)];

    if len(cluster) > 0:
        small, large = clusterToRegion(cluster);
        unclustered.append((small, large));

    return unclustered;


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hotspot detection using C++ AlleleSearcher class");

    parser.add_argument(
        "--bam",
        help="BAM file in which to call hotspots",
        required=True
    );

    parser.add_argument(
        "--bam2",
        help="Second BAM file if necessary for hybrid calling. In this case, the second BAM file is PacBio and the first is Illumina",
        required=False,
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
        "--chunkSize",
        help="Size of each chunk to analyze",
        required=True,
        type=int
    );

    parser.add_argument(
        "--numReadsPerChunk",
        help="Maximum number of reads to retain per chunk",
        required=True,
        type=int,
    );

    parser.add_argument(
        "--pacbio",
        help="Indicate that these are PacBio reads",
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
        help="Enable logging messages",
        default=False,
        action="store_true",
    );

    args = parser.parse_args();

    if args.bam2 is not None:
        args.pacbio = False;

    logging.basicConfig(level=(logging.DEBUG if args.debug else logging.INFO), format='%(asctime)-15s %(message)s');
    libCallability.initLogging(args.debug);

    cache = ReferenceCache(database=args.ref);

    if len(args.region.split(",")) == 1:
        chromosome = args.region;
        cache.chrom = chromosome;
        start = 0;
        stop = len(cache);
    else:
        chromosome, start, stop = args.region.split(",");
        start = int(start);
        stop = int(stop);

    numChunks = (stop - start) // args.chunkSize;
    partialLastChunk = False;

    if numChunks * args.chunkSize < (stop - start):
        partialLastChunk = True;

    ohandle = open(args.output, 'w');

    def doOneChunkSingle(begin, end, i):
        container = PileupContainerLite(args.bam, chromosome, begin, end - begin, args.numReadsPerChunk);
        logging.info("Found %d reads" % len(container.pileupreads));
        if len(container.pileupreads) == 0:
            return;
        # adjustCigartuples(container, cache);
        logging.info("Completed adjusting cigartuples");
        logging.info("Start, End = %d, %d" % (container.referenceStart, container.referenceEnd));
        searcher = AlleleSearcher(container, begin, end, cache, contextLength=3, pacbio=args.pacbio, strict=False);

        logging.info("Completed %d chunks" % (i + 1));

        for left, right in searcher.differingRegions:
            logging.debug("Obtained differing regions %d, %d" % (left, right));
            for j in range(left, right):
                hotspot = dict({'chromosome': chromosome, 'position': j});
                ohandle.write(str(hotspot) + '\n');

    def doOneChunkHybrid(begin, end, i):
        container = PileupContainerLite(
            args.bam,
            chromosome,
            begin,
            end - begin,
            args.numReadsPerChunk,
        );
        container2 = PileupContainerLite(
            args.bam2,
            chromosome,
            begin,
            end - begin,
            args.numReadsPerChunk,
            clipReads=True,
            clipFlank=200
        );
        # adjustCigartuples(container, cache);
        # adjustCigartuples(container2, cache);
        logging.info("Completed adjusting cigartuples");
        logging.info("Start, End = %d, %d" % (container.referenceStart, container.referenceEnd));
        searcher = AlleleSearcher(container, begin, end, cache, contextLength=3, pacbio=False, strict=False);
        searcher2 = AlleleSearcher(container, begin, end, cache, contextLength=3, pacbio=True, strict=False);

        logging.info("Completed %d chunks" % (i + 1));

        # Combine differing regions and sort
        differingRegions = sorted(searcher.differingRegions + searcher2.differingRegions);

        # Print out the results
        for left, right in combineOverlapping(differingRegions):
            for j in range(left, right):
                hotspot = dict({'chromosome': chromosome, 'position': j});
                ohandle.write(str(hotspot) + '\n');

    for i in range(numChunks):
        begin_ = start + args.chunkSize * i;
        end_ = begin_ + args.chunkSize;

        if args.bam2 is not None:
            doOneChunkHybrid(begin_, end_, i);
        else:
            doOneChunkSingle(begin_, end_, i);

    if partialLastChunk:
        begin_ = start + args.chunkSize * numChunks;
        end_ = stop;

        if args.bam2 is not None:
            doOneChunkHybrid(begin_, end_, numChunks);
        else:
            doOneChunkSingle(begin_, end_, numChunks);

    ohandle.close();
