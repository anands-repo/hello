import os
from argparse import ArgumentParser
from ReferenceCache import ReferenceCache
import logging
from PySamFastaWrapper import PySamFastaWrapper


def cmd(chromosome, start, stop, bam, ref, output, pacbio=False, log=False, bam2=None, hybrid_hotspot=False):
    command = "python %s/HotspotDetectorDVFiltered.py" % os.path.split(os.path.abspath(__file__))[0]
    command += " --bam %s" % (bam if bam2 is None else (bam + "," + bam2))
    command += " --ref %s" % ref
    command += " --region %s,%d,%d" % (chromosome, start, stop)
    command += " --output %s" % output
    command += " --pacbio" if pacbio else ""
    command += " --hybrid_hotspot" if hybrid_hotspot else ""
    command += " --q_threshold %d" % args.q_threshold
    command += " --mapq_threshold %d" % args.mapq_threshold
    command += " >& %s" % os.path.splitext(output)[0] + ".log" if log else ""

    return command


def doOneRegion(args, chromosome, start, stop, outputPrefix=None):
    ref = args.ref

    if (start is not None) and (stop is not None):
        start = int(start)
        stop = int(stop)
    else:
        cache = PySamFastaWrapper(ref)
        cache.chrom = chromosome
        stop = len(cache)
        start = 0

    nJobs = args.nJobs
    bam = args.bam

    splitSize = (stop - start) // nJobs
    lastChunk = False

    if nJobs * splitSize < (stop - start):
        lastChunk = True

    if outputPrefix is None:
        writeCmd = print
        fname = None
    else:
        fname = outputPrefix + "_chromosome%s.sh" % chromosome
        fhandle = open(fname, 'w')

        def writeCmd(x):
            fhandle.write(x + "\n")

    for i in range(nJobs):
        begin = start + splitSize * i
        end = begin + splitSize
        outputname = os.path.join(os.getcwd(), "output%d.txt" % i) if (outputPrefix is None) else outputPrefix + "_chromosome%s_job%d.txt" % (chromosome, i)
        command = cmd(
            chromosome,
            begin,
            end,
            bam,
            ref,
            outputname,
            args.pacbio,
            log=args.log,
            bam2=args.bam2,
            hybrid_hotspot=args.hybrid_hotspot,
        )
        writeCmd(command)

    if lastChunk:
        begin = start + nJobs * splitSize
        end = stop
        outputname = os.path.join(os.getcwd(), "output%d.txt" % (i + 1))
        outputname = os.path.join(os.getcwd(), "output%d.txt" % (i + 1)) if (outputPrefix is None) else outputPrefix + "_chromosome%s_job%d.txt" % (chromosome, i + 1)
        command = cmd(chromosome, begin, end, bam, ref, outputname, args.pacbio, log=args.log, bam2=args.bam2)
        writeCmd(command)

    if outputPrefix is not None:
        fhandle.close()

    return fname


def main(args):
    chromosome = args.chromosome

    if args.chromosome == 'wgs':
        assert(args.outputDir is not None), "Provide output directory for wgs"
        outputDir = os.path.abspath(args.outputDir)
        allcmds = []

        # TBD: change to accurate determination of chromosomal contigs
        # from bamfile later
        for chromosome in [str(x) for x in range(1, 23)]:
            chrdir = os.path.join(outputDir, 'chr%s' % chromosome)
            if not os.path.exists(chrdir):
                os.makedirs(chrdir)
            else:
                logging.info("Found %s, using it" % chrdir)
            allcmds.append(doOneRegion(args, chromosome, None, None, os.path.join(chrdir, 'jobs')))

        # Print all commands into one file
        whandle = open(os.path.join(args.outputDir, 'allcommands.sh'), 'w')

        for cmdfile in allcmds:
            with open(cmdfile, 'r') as handle:
                for line in handle:
                    whandle.write(line)

        whandle.close()
    else:
        if args.region is not None:
            start, stop = args.region.split(",")
            start = int(start)
            stop = int(stop)
        else:
            start, stop = None, None

        outputPrefix = os.path.join(os.path.abspath(args.outputDir), 'jobs') \
            if (args.outputDir is not None) else None

        doOneRegion(args, chromosome, None, None, outputPrefix)


if __name__ == "__main__":
    parser = ArgumentParser(description="Create multiple HotspotDetector Jobs")
    parser.add_argument("--chromosome", help="Chromosome", required=True)
    parser.add_argument("--region", help="start,stop", required=False)
    parser.add_argument("--nJobs", help="Number of jobs", required=True, type=int)
    parser.add_argument("--bam", help="Bam file", required=True)
    parser.add_argument("--bam2", help="Second bam file for hybrid calling", required=False)
    parser.add_argument("--ref", help="Reference cache location", required=True)
    parser.add_argument("--pacbio", help="Indicate that this is for PacBio reads", action="store_true", default=False)
    parser.add_argument("--log", help="Enable logging", default=False, action="store_true")
    parser.add_argument("--outputDir", help="Output directory", default=None)
    parser.add_argument("--hybrid_hotspot", help="Use hybrid hotspot detection", default=False, action="store_true")
    parser.add_argument("--q_threshold", help="Quality score threshold", default=10, type=int)
    parser.add_argument("--mapq_threshold", help="Mapping quality threshold", default=10, type=int)
    args = parser.parse_args()
    main(args)
