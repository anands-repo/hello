from more_itertools import peekable
import vcf
import ast
from ReferenceCache import ReferenceCache
import argparse
import logging


def modifyFeatures(record, newStart, newStop, cache):
    cache.chrom = record['chromosome'];
    allelicMapping = newAlleles(
        record['features'].keys(),
        record['start'],
        record['stop'],
        newStart,
        newStop,
        cache,
    );
    newRecord = dict(record);
    newRecord['ref'] = ''.join(cache[newStart:newStop]);
    newRecord['features'] = dict();

    for allele, newAllele in allelicMapping.items():
        newRecord['features'][newAllele] = record['features'][allele];

    newRecord['start'] = newStart;
    newRecord['stop'] = newStop;

    return newRecord;


def modifyVcfRecord(record, newStart, newStop, cache):
    cache.chrom = record['chromosome'];
    allelicMapping = newAlleles(
        record['alt'],
        record['start'],
        record['stop'],
        newStart,
        newStop,
        cache,
    );
    newRecord = dict(record);
    newRecord['ref'] = ''.join(cache[newStart:newStop]);
    newRecord['alt'] = list();

    for allele, newAllele in allelicMapping.items():
        if allele in record['alt']:
            newRecord['alt'].append(newAllele);

    newRecord['start'] = newStart;
    newRecord['stop'] = newStop;

    return newRecord;


def getAlleles(ref, alt, gt):
    """
    Given ref alt and gt, return a list of alleles at site

    :param ref: str
        Reference allele

    :param alt: list
        List of alt alleles at site

    :param gt: list
        Genotype at site

    :return: set
        Set of alleles at site
    """
    if gt is None:
        return None;
    allAlleles = [ref] + alt;
    return set([allAlleles[i] for i in gt]);

featureIndices = {
    "likelihood": 0,
    "unionMapping": 1,
    "maxMapping": 2,
    "unionUniqueMapping": 3,
    "maxUniqueMapping": 4,
};


def topAlleles(record, criterion="likelihood"):
    """
    Determine the top alleles from a feature record

    :param record: dict
        Feature dictionary

    :param criterion: str
        One of "likelihood", "maxMapping", "unionMapping", "maxUniqueMapping", "unionUniqueMapping"

    :return: tuple
        Top-two alleles
    """
    zipped = [
        (
            value[featureIndices[criterion]],
            allele
        )
        for allele, value in record['features'].items()
    ];
    sortedZipped = sorted(zipped, reverse=True);
    values, alleles = tuple(zip(*sortedZipped));
    return alleles[:2];


def newAlleles(alleles, oldStart, oldStop, newStart, newStop, refCache):
    """
    Given old and new start and stop positions of a record, expand alleles to fill the new space

    :param alleles: iterable
        List of old alleles

    :param oldStart: int
        Old start position

    :param oldStop: int
        Old stop position

    :param newStart: int
        New start position

    :param newStop: int
        New stop position

    :param refCache: ReferenceCache
        ReferenceCache object initialized to the correct chromosome

    :return: dict
        Mapping of old alleles to new alleles
    """
    numRefPrefix = oldStart - newStart;
    numRefSuffix = newStop - oldStop;
    refPrefix = '' if numRefPrefix == 0 else ''.join(refCache[newStart:oldStart])
    refSuffix = '' if numRefSuffix == 0 else ''.join(refCache[oldStop:newStop]);
    allelicMapping = dict();
    for allele in alleles:
        allelicMapping[allele] = refPrefix + allele + refSuffix;
    return allelicMapping;


def genotype(record, index=0):
    """
    Parse genotype in tuple form
    """
    separator = None;

    if '|' in record.samples[index]['GT']:
        separator = '|';
    elif '/' in record.samples[index]['GT']:
        separator = '/';

    if separator is None:
        return None;

    gt = list(map(int, record.samples[index]['GT'].split(separator)));

    return gt;


def failVcfReader(failList):
    """
    Reads a failure list
    """
    reader = vcf.Reader(open(failList, 'r'));

    for record in reader:
        if (record.samples[0]['BD'] == 'FN') or (record.samples[1]['BD'] == 'FP'):
            gt0 = None if record.samples[0]['BVT'] == 'NOCALL' else genotype(record, index=0);
            gt1 = None if record.samples[1]['BVT'] == 'NOCALL' else genotype(record, index=1);
            ftype = (record.samples[0]['BD'], record.samples[1]['BD']);
            yield \
                {
                    'chromosome': record.CHROM,
                    'start': record.POS - 1,
                    'stop': record.POS - 1 + len(record.REF),
                    'ref': record.REF,
                    'alt': [str(s) for s in record.ALT],
                    'ftype': ftype,
                    'vtype': (record.samples[0]['BVT'], record.samples[1]['BVT']),
                    'gt0': gt0,
                    'gt1': gt1,
                };


def bedReader(bedname):
    """
    Reads a bed file
    """
    with open(bedname, 'r') as fhandle:
        for line in fhandle:
            items = line.split();
            yield {
                'chromosome': items[0],
                'start': int(items[1]),
                'stop': int(items[2]),
            };


def featureReader(featureFile):
    """
    Reads a features file
    """
    with open(featureFile, 'r') as fhandle:
        for line in fhandle:
            yield ast.literal_eval(line);


def chromosomeToNumber(chromosome):
    """
    Establish chromosomal ordering
    """
    chromDict = dict({str(x) : x for x in range(1, 23)});
    chromDict['X'] = 23;
    chromDict['Y'] = 24;
    return chromDict[chromosome]


class Stepper:
    a_before_b = 0;
    a_partially_before_b = 1;
    a_identical_to_b = 2;
    a_partially_after_b = 3;
    a_after_b = 4;
    a_contains_b = 5;
    a_in_b = 6;

    """
    Steps through streams
    """
    def __init__(self, stream0, stream1, stream2, streamToTerminate=2):
        """
        :param stream0: peekable
            stream0 is the larger region (active region or high-confidence bed)

        :param stream1: peekable
            stream1 is a smaller region that falls within the
            larger region (e.g., failure report from hap.py)

        :param stream2: peekable
            stream2 is a smaller region similar to stream1 (e.g., features from call)

        :param streamToTerminate: int
            The stream (other than stream0) running out of which, we should terminate
        """
        self.stream0 = stream0;
        self.stream1 = stream1;
        self.stream2 = stream2;
        # Reading each stream should provide dictionary entries with
        # {'chromosome', 'start', 'stop'} entries
        self.streamToTerminate = streamToTerminate;

    def __iter__(self):
        return self;

    def isSmallRegionAfterLargeRegion(self, smallRegion, largeRegion):
        """
        Is an item read from stream1/stream2 after an item read from stream0

        :param smallRegion: dict
            Item read from stream1/stream2

        :param largeRegion: dict
            Item read from stream0

        :return: bool
            The result of the comparison
        """
        if chromosomeToNumber(smallRegion['chromosome']) > chromosomeToNumber(largeRegion['chromosome']):
            return True;

        if chromosomeToNumber(smallRegion['chromosome']) < chromosomeToNumber(largeRegion['chromosome']):
            return False;

        if smallRegion['stop'] > largeRegion['stop']:
            return True;

        return False;

    def isSmallRegionBeforeLargeRegion(self, smallRegion, largeRegion):
        """
        Is an item read from stream1/stream2 before an item read from stream0

        :param smallRegion: dict
            Item read from stream1/stream2

        :param largeRegion: dict
            Item read from stream0

        :return: bool
            The result of the comparison
        """
        if chromosomeToNumber(smallRegion['chromosome']) > chromosomeToNumber(largeRegion['chromosome']):
            return False;

        if chromosomeToNumber(smallRegion['chromosome']) < chromosomeToNumber(largeRegion['chromosome']):
            return True;

        if smallRegion['start'] < largeRegion['start']:
            return True;

        return False;

    def overlapAnalysis(self, regionA, regionB):
        """
        Conduct overlap analysis between two regions

        :param regionA: dict
            One of the regions read from one of the streams

        :param regionB: dict
            One of the regions read from one of the streams

        :return: int
            Integer representing code
        """
        if chromosomeToNumber(regionA['chromosome']) < chromosomeToNumber(regionB['chromosome']):
            return Stepper.a_before_b;

        if regionA['chromosome'] == regionB['chromosome']:
            if regionA['stop'] <= regionB['start']:
                return Stepper.a_before_b;

            if regionA['start'] <= regionB['start'] < regionA['stop'] < regionB['stop']:
                return Stepper.a_partially_before_b;

            if regionB['start'] <= regionA['start'] < regionB['stop'] < regionA['stop']:
                return Stepper.a_partially_after_b;

            if (regionA['start'] == regionB['start']) and (regionA['stop'] == regionB['stop']):
                return Stepper.a_identical_to_b;

            if (regionA['start'] <= regionB['start'] < regionB['stop'] <= regionA['stop']):
                return Stepper.a_contains_b;

            if (regionB['start'] <= regionA['start'] < regionA['stop'] <= regionB['stop']):
                return Stepper.a_in_b;

            return Stepper.a_after_b;

        return Stepper.a_after_b;

    def smaller(self, s1, s2):
        """
        Pick the smaller or earlier one from s1, s2

        :param s1: dict
            Chromosomal locus

        :param s2: dict
            Chromosomal locus

        :return: dict
            Smaller of the loci
        """
        if s1 is None:
            return s2;

        if s2 is None:
            return s1;

        if s1['chromosome'] != s2['chromosome']:
            if chromosomeToNumber(s1['chromosome']) < chromosomeToNumber(s2['chromosome']):
                return s1;
            else:
                return s2;

        if s1['start'] < s2['start']:
            return s1;
        else:
            return s2;

    def __next__(self):
        ordering = None;
        overlapSet = set([
            Stepper.a_partially_after_b,
            Stepper.a_partially_before_b,
            Stepper.a_identical_to_b,
            Stepper.a_contains_b,
            Stepper.a_in_b,
        ]);

        # Search for s0, s1, s2 such that sx, x > 0, is either None, or within s0
        while True:
            s1 = self.stream1.peek(None);
            s2 = self.stream2.peek(None);
            s0 = self.stream0.peek(None);

            streams = [s0, s1, s2];

            logging.debug("Stepper received %s, %s, %s" % (str(s0), str(s1), str(s2)));

            if s0 is None:
                raise StopIteration("Completed reading data");

            if streams[self.streamToTerminate] is None:
                raise StopIteration;

            while(
                ((s1 is not None) and self.isSmallRegionAfterLargeRegion(s1, s0)) or
                ((s2 is not None) and self.isSmallRegionAfterLargeRegion(s2, s0))
            ):
                next(self.stream0);
                s0 = self.stream0.peek(None);
                if s0 is None:
                    raise StopIteration("Completed reading data");

            # If either s1 or s2 is None, act appropriately --- this code has been moved
            # from below to help the hybrid case. In fact, I do not know why this wasn't causing problems
            # before. This should have been, because s1, or s2 can be None
            if (s1 is None) or (s2 is None):
                if s1 is not None:
                    next(self.stream1);

                if s2 is not None:
                    next(self.stream2);

                logging.debug("Returning (1) %s, %s" % (str(s1), str(s2)));
                return (s1, s2);

            # If both s1 and s2 results are empty, raise StopIteration
            if (s1 is None) and (s2 is None):
                raise StopIteration;

            # If s1, s2 overlap, then ensure that both s1, and s2 are within s0
            ordering = self.overlapAnalysis(s1, s2);

            if (s1 is not None) and (s2 is not None) and (ordering in overlapSet):
                if (self.overlapAnalysis(s0, s1) == Stepper.a_contains_b) and \
                        (self.overlapAnalysis(s0, s2) == Stepper.a_contains_b):
                    break;
                else:
                    next(self.stream1);
                    next(self.stream2);
            else:
                smaller = self.smaller(s1, s2);
                if self.overlapAnalysis(s0, smaller) == Stepper.a_contains_b:
                    break;
                else:
                    if smaller == s1:
                        next(self.stream1);
                    else:
                        next(self.stream2);

        # Compare s1 and s2 and return the lower one and advance the stream
        # If both are overlapping or identical, return both
        if ordering == Stepper.a_before_b:
            value = (s1, None);
            next(self.stream1);
        elif ordering == Stepper.a_after_b:
            value = (None, s2);
            next(self.stream2);
        elif ordering in overlapSet:
            next(self.stream1);
            next(self.stream2);

            # Collect all s2 which overlap with the curent s2
            overlappingS2s = [s2];
            nextS2 = self.stream2.peek(None);

            while (nextS2 is not None) and (self.overlapAnalysis(overlappingS2s[-1], nextS2) in overlapSet):
                overlappingS2s.append(nextS2);
                next(self.stream2);
                nextS2 = self.stream2.peek(None);

            if len(overlappingS2s) > 1:
                value = (s1, overlappingS2s);
            else:
                value = (s1, s2);
        else:
            raise StopIteration;

        logging.debug("Returning (2) %s, %s" % (str(value[0]), str(value[1])));
        return value;


class Analyzer:
    """
    Analyzes failure list from hap.py for PileupAnalyzer, providing reasons for failure
    """
    def __init__(self, features, failures, highconf, ref):
        """
        :param features: str
            Features file

        :param failures: str
            Failures file

        :param highconf: str
            High confidence bed file

        :param ref: str
            ReferenceCache path
        """
        self.stream0 = peekable(bedReader(highconf));
        self.stream1 = peekable(featureReader(features));
        self.stream2 = peekable(failVcfReader(failures));
        self.stepper = Stepper(self.stream0, self.stream1, self.stream2);
        self.cache = ReferenceCache(database=ref);

    def __iter__(self):
        return self;

    def expandRegions(self, s1, s2):
        """
        Expands overlapping alleles from feature list and vcf records

        :param s1: dict
            Feature dictionary

        :param s2: dict
            VCF record

        :return tuple
            Modified s1, s2
        """
        self.cache.chrom = s1['chromosome'];
        start = min(s1['start'], s2['start']);
        stop = max(s1['stop'], s2['stop']);

        return modifyFeatures(s1, start, stop, self.cache), modifyVcfRecord(s2, start, stop, self.cache);

    def __next__(self):
        s1, s2 = next(self.stepper);

        if (s1 is None) and (s2 is None):
            return None;

        while s2 is None:
            # We are only interested in failures,
            # so a feature accompanied without a fail vcf entry
            # is discarded. This almost makes it so that we do not
            # need a high confidence bed file
            s1, s2 = next(self.stepper);

        if s1 is None:
            return {
                'chromosome': s2['chromosome'],
                'start': s2['start'],
                'stop': s2['stop'],
                'vtype': s2['vtype'],
                'reason': 'DID_NOT_RUN',
            };

        # If multiple overlapping s2s are received, decide if exactly one of them contains TP fields
        if type(s2) is list:
            s2WithTP = None;
            minPosition = min([s2_['start'] for s2_ in s2]);
            maxPosition = max([s2_['stop'] for s2_ in s2]);

            for s2_ in s2:
                if s2_['gt0'] is not None:
                    if s2WithTP is None:
                        s2WithTP = s2_;
                    else:
                        return {
                            'chromosome': s2[0]['chromosome'],
                            'start': minPosition,
                            'stop': maxPosition,
                            'vtype': s2[0]['vtype'],
                            'reason': 'CANNOT_ANALYSE'
                        };

            if s2WithTP is None:
                return {
                    'chromosome': s1['chromosome'],
                    'start': minPosition,
                    'stop': maxPosition,
                    'vtype': 'UNKNOWN',
                    'reason': 'CANNOT_ANALYSE'
                };
            else:
                self.cache.chrom = s2WithTP['chromosome'];
                s2 = modifyVcfRecord(s2WithTP, minPosition, maxPosition, self.cache);

        # Expand overlapping entries to cover the whole reference region
        s1, s2 = self.expandRegions(s1, s2);
        assert(s2['ref'] == s1['ref']);

        # Ground-truth and called alleles
        gtAlleles = getAlleles(s2['ref'], s2['alt'], s2['gt0']);
        calledAlleles = getAlleles(s2['ref'], s2['alt'], s2['gt1']);

        # Obtain top alleles in call (do it according to likelihood for now)
        topTwo = topAlleles(s1);

        # Now categorize the error
        if (s2['gt0'] is None) or (len(gtAlleles) == 1):
            # Only one allele in ground-truth
            gtAllele = next(iter(gtAlleles)) if gtAlleles is not None else s2['ref'];

            if (s2['gt1'] is None):
                assert(s2['gt0'] is not None), "Found NOCALL->NOCALL site";

            # Eventhough the "s1 is None" check exists above, it doesn't mean gt1 is called
            # This is because 0/0 is a possible genotype field. However, further analysis here
            # is based on s1, so this should be okay.
            # This condition can still happen despite the (s1 is None) check above
            # (s1 is not None) doesn't imply gt1 is called because [0/0] is still possible

            if gtAllele in set(s1['features'].keys()):
                # If ground-truth allele is the top allele, this is a classifier problem
                if gtAllele == topTwo[0]:
                    return {
                        'chromosome': s2['chromosome'],
                        'start': s2['start'],
                        'stop': s2['stop'],
                        'vtype': s2['vtype'],
                        'reason': 'CLASSIFICATION',
                        'siteType': 'HOMOZYGOUS,'
                    };
                # If ground-truth allele is not the top allele then this is a modeling problem
                else:
                    return {
                        'chromosome': s2['chromosome'],
                        'start': s2['start'],
                        'stop': s2['stop'],
                        'vtype': s2['vtype'],
                        'reason': 'MODELING',
                        'siteType': 'HOMOZYGOUS,'
                    };
            else:
                # Ground-truth allele is not in the allele list at all
                # This is an assembly issue
                return {
                    'chromosome': s2['chromosome'],
                    'start': s2['start'],
                    'stop': s2['stop'],
                    'vtype': s2['vtype'],
                    'reason': 'ASSEMBLY',
                    'siteType': 'HOMOZYGOUS,'
                };
        else:
            # Two alleles in ground-truth

            # If gtAlleles are in the set of all alleles, then this is not an assembly issue
            if len(gtAlleles.difference(set(s1['features'].keys()))) == 0:
                if len(gtAlleles.difference(set(topTwo))) == 0:
                    # Both alleles are found in top-two => thresholding/classifier issue
                    # One could also say this is a modeling issue because the score could be too low
                    # But we could simply collect the likelihoods in these sites and plot a histogram
                    return {
                        'chromosome': s2['chromosome'],
                        'start': s2['start'],
                        'stop': s2['stop'],
                        'vtype': s2['vtype'],
                        'reason': 'CLASSIFICATION',
                        'siteType': 'HETEROZYGOUS',
                    };
                else:
                    return {
                        'chromosome': s2['chromosome'],
                        'start': s2['start'],
                        'stop': s2['stop'],
                        'vtype': s2['vtype'],
                        'reason': 'MODELING',
                        'siteType': 'HETEROZYGOUS',
                    };
            else:
                # Both alleles aren't found in the top-two => assembly issue
                return {
                    'chromosome': s2['chromosome'],
                    'start': s2['start'],
                    'stop': s2['stop'],
                    'vtype': s2['vtype'],
                    'reason': 'ASSEMBLY',
                    'siteType': 'HETEROZYGOUS',
                };


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate hap.py results for PileupAnalyzer");

    parser.add_argument("--happy", help="Happy results", required=True);
    parser.add_argument("--bed", help="Bed region within which to evaluate", required=True);
    parser.add_argument("--features", help="Features from PileupAnalyzer", required=True);
    parser.add_argument("--ref", help="Reference cache location", required=True);
    parser.add_argument("--output", help="Name of the output file", required=True);

    args = parser.parse_args();

    analyzer = Analyzer(args.features, args.happy, args.bed, args.ref);

    with open(args.output, 'w') as fhandle:
        for item in analyzer:
            fhandle.write(str(item) + '\n');
            print(item);
