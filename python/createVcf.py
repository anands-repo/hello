import argparse
import subprocess
import re
import ast
from ReferenceCache import ReferenceCache
from functools import reduce
import os
import subprocess
import pysam

SNV_THRESHOLD=0.1
INDEL_THRESHOLD=0.1

def correctRepresentation(variant, ref):
    """
    Correct the variant representation (variant normalization)
    Follows https://genome.sph.umich.edu/wiki/Variant_Normalization

    :param variant: dict
        Dictionary representation of the given variant

    :param ref: str
        Reference cache location

    :return: dict
        Corrected dictionary representation
    """
    referenceCache = ReferenceCache(database=ref);
    referenceCache.chrom = variant['CHROM'];
    allAlleles = [variant['REF']] + variant['ALT'];
    ALLELES = [allAlleles[gt] for gt in sorted(list(set(map(int, variant['GT'].split('/')))))];

    if reduce(lambda x, y : x and y, [a == variant['REF'] for a in ALLELES], True):
        return variant;
    else:
        runLoop = True;

        # Left align and right parsimony
        while runLoop:
            runLoop = False;

            # If alleles end with the same base, truncate rightmost base
            rightBases = [variant['REF'][-1]] + [a[-1] for a in ALLELES];
            if reduce(lambda x, y : x and y, [r == rightBases[0] for r in rightBases]):
                variant['REF'] = variant['REF'][:-1];
                ALLELES = [a[:-1] for a in ALLELES];
                runLoop = True;

            # If any allele is empty, incorporate a base from the reference to its left
            if reduce(lambda x, y : x or y, [len(a) == 0 for a in ALLELES]) or (len(variant['REF']) == 0):
                variant['POS'] -= 1;
                leftBase = referenceCache[variant['POS']];
                variant['REF'] = leftBase + variant['REF'];
                ALLELES = [leftBase + a for a in ALLELES];
                runLoop = True;

        # Left parsimony
        runLoop = True;

        while runLoop:
            # Every allele is of length at least 2
            if reduce(lambda x, y : x and y, [len(a) >= 2 for a in ALLELES]) and (len(variant['REF']) >= 2):
                leftBases = [variant['REF'][0]] + [a[0] for a in ALLELES];
                if reduce(lambda x, y : x and y, [l == leftBases[0] for l in leftBases]):
                    variant['POS'] += 1;
                    variant['REF'] = variant['REF'][1:];
                    ALLELES = [a[1:] for a in ALLELES];
                else:
                    runLoop = False;
            else:
                runLoop = False;

        # Form new alt alleles
        ALT = [a for a in ALLELES if a != variant['REF']];
        variant['ALT'] = ALT;

        newGT = [];

        for a in ALLELES:
            if a != variant['REF']:
                index = ALT.index(a) + 1;
            else:
                index = 0;
            newGT.append(index);

        if len(newGT) == 1:
            newGT = newGT * 2;

        newGT = '/'.join(map(str, newGT));
        variant['GT'] = newGT;

        return variant;

def generate_vcf_header(contig_lengths, chromosomes):
    string  = "##fileformat=VCFv4.1\n";

    for chr_ in chromosomes:
        if chr_ in contig_lengths:
            string += "##contig=<ID=%s,length=%d>\n"%(chr_,contig_lengths[chr_]);

    string += "##INFO=<ID=AssemblyResults,Description=Obtained probabilistic graph assembly>\n";
    string += "##FORMAT=<ID=GT,Number=1,Type=String,Description=Genotype>\n";
    string += "##FILTER=<ID=FAIL,Description=Failed allele fraction test>\n";
    string += "#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	SAMPLE1\n";

    return string;

def get_contig_lengths(bam):
    # command = ['samtools', 'view', '-H', bam];
    # proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE);
    # msg, err = proc.communicate();
    # contig_length = re.compile("^\s*\@SQ\s+SN:(\S+)\s+LN:([0-9]+)\s*.*$");

    # length_dict = {};

    # for line in msg.splitlines():
    #     line = line.decode('ascii');
    #     result = contig_length.match(line);

    #     if result is not None:
    #         length_dict[str(result.group(1))] = int(result.group(2));
    chromosomes = list(map(str, range(1,23))) + ['X', 'Y'];
    fhandle     = pysam.AlignmentFile(bam, 'rb');
    lengths     = [fhandle.get_reference_length(c) for c in chromosomes];
    return dict(zip(chromosomes, lengths));

def createVcfEntry(line, ref, normalize = False):
    entries = ast.literal_eval(line);
    passes  = lambda a, flag : a > INDEL_THRESHOLD if flag else a > SNV_THRESHOLD;

    if 'CHROM' in entries:
        fracs = list(map(float, entries['FRAC'].split(':')));
        gts   = list(map(int, entries['GT'].split('/')));
        fDict = dict(zip(gts, fracs));
        indel = False;

        for a in entries['ALT']:
            if len(a) != len(entries['REF']):
                indel = True;

        # for a, f in zip(entries['GT'].split('/'), fracs):
        #     if f > (SNV_THRESHOLD if not indel else INDEL_THRESHOLD):
        #         passingAlleles.append(int(a));

        # Choose alleles with top two signal levels, and threshold them
        # if len(passingAlleles) > 2:
        topTwo = [];
        fZip   = zip(fracs, gts);
        sZip   = sorted(fZip);
        topTwo.append(sZip[-1][1]);

        if len(sZip) > 1:
            topTwo.append(sZip[-2][1]);

        passingAlleles = [p for p in topTwo if passes(fDict[p], indel)];

        if sum(passingAlleles) > 0:
            flag  = 'PASS';
            if len(passingAlleles) == 1: passingAlleles = passingAlleles * 2;
            newGt = '/'.join([str(x) for x in passingAlleles]);
        else:
            flag  = 'FAIL';
            newGt = '/'.join([str(x) for x in topTwo]);

        variantDict = {
            'CHROM':str(entries['CHROM']),
            'POS':entries['POS'],
            'REF':entries['REF'],
            'ALT':entries['ALT'],
            'GT':newGt
        };

        if normalize:
            variantDict = correctRepresentation(variantDict, ref);

        entry = "%s\t%d\t.\t%s\t%s\t30\t%s\tAssemblyResults\tGT\t%s\n"%(str(variantDict['CHROM']), variantDict['POS']+1, variantDict['REF'], ','.join(variantDict['ALT']), flag, variantDict['GT']);
    else:
        entry = None;

    return entry;

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create VCF file from assembly results");

    parser.add_argument(
        "--assembly",
        help = "Assembly results file",
        required = True,
    );

    parser.add_argument(
        "--bam",
        help = "Bam file used for assembly",
        required = True,
    );

    parser.add_argument(
        "--threshold",
        default = 0.1,
        type = float,
        help = "Threshold for signal cutoff"
    );

    parser.add_argument(
        "--ref",
        help = "Location of reference cache database",
        required = True,
    );

    parser.add_argument(
        "--normalize",
        action = 'store_true',
        help = "Normalize variant representation",
        default = False,
    );

    parser.add_argument(
        "--vcf",
        required = True,
        help = "Output VCF filename",
    );

    args = parser.parse_args();

    SNV_THRESHOLD = args.threshold;
    INDEL_THRESHOLD = args.threshold;

    contigLengths = get_contig_lengths(args.bam);
    header        = generate_vcf_header(contigLengths, ['18']);
    tmpvcf        = os.path.join('/tmp', 'tmp.vcf');

    with open(args.assembly, 'r') as fhandle, open(tmpvcf, 'w') as whandle:
        # whandle.write(header);
        for line in fhandle:
            results = createVcfEntry(line, args.ref, args.normalize);
            if results is not None:
                whandle.write(str(results));

    with open(args.vcf, 'w') as fhandle:
        fhandle.write(header);

    with open(args.vcf, 'a') as fhandle:
        # sort -k1,1d -k2,2n
        subprocess.call(['sort', '-k1,1d', '-k2,2n', tmpvcf], stdout=fhandle);