# © 2019 University of Illinois Board of Trustees.  All rights reserved
import argparse
import torch
from Bio.pairwise2 import align
from ReferenceCache import ReferenceCache
from functools import reduce
from createVcf import correctRepresentation
import ast
#from checkBeamSearchAccuracy import convertToFraction
import logging
#from HaplotypeAnalyzer import HaplotypeAnalyzer
import numpy as np
import pickle

def rerank(dict_, rf, topN = 4):
    avg_cov  = dict_['avg_cov'];
    std_cov  = dict_['std_cov'];
    min_cov  = dict_['min_cov'];
    ref_len  = dict_['region'][1] - dict_['region'][0];
    scores   = [];
    lengths  = [];
    supports = [];

    for score, contig in dict_['contigs']:
        scores.append(score);
        lengths.append(len(contig));
        supports.append(dict_['supports'][contig]);

    features = sorted(list(zip(scores, supports, lengths)), reverse=True);
    features = reduce(lambda x, y : x + y, [list(x) for x in features], []);

    if len(features) < 3 * topN:
        for i in range(topN - len(features)//3):
            features += [-100,-1,-1];
    else:
        features = features[:3 * topN];

    features += [avg_cov, min_cov, std_cov, ref_len];
    features  = np.array([features]);
    predict   = rf.predict_proba(features)[1].flatten();
    indices   = np.argsort(predict);
    topTwo    = [dict_['contigs'][i] for i in indices[:2]];

    return topTwo;

def determinePassingModel(dict_, topTwo, rf = False):
    avg_cov  = dict_['avg_cov'];
    std_cov  = dict_['std_cov'];
    min_cov  = dict_['min_cov'];
    l0, len0 = topTwo[0][0], len(topTwo[0][1]);
    l1, len1 = topTwo[1][0], len(topTwo[1][1]);
    reflen   = dict_['region'][1] - dict_['region'][0];
    s0, s1   = dict_['uniqueSupports'][topTwo[0][1]], dict_['uniqueSupports'][topTwo[1][1]];
    # s0, s1   = dict_['supports'][topTwo[0][1]], dict_['supports'][topTwo[1][1]];
    tensor   = torch.Tensor([[l0, len0, s0, l1, len1, s1, avg_cov, min_cov, std_cov, reflen]]);
    
    if rf:
        predict = model.predict_proba(tensor.numpy())[1];
    else:
        with torch.no_grad():
            predict = model(tensor).cpu().data.numpy();

    label   = np.argmax(predict.flatten());                

    passing  = [c for s,c in topTwo] if label == 1 else [topTwo[0][1]];

    return passing;

def createRefMatrix(alignments, refSeq):
    """
    Create a matrix from alignments. Expects two alignments one for each haplotype.
    If haplotypes are identical, duplicate the alignments
    """
    matrix    = [{'REF':[i], 'ALT':[[],[]]} for i in refSeq];

    for i, alignment in enumerate(alignments):
        refCounter = 0;
        for position in alignment:
            ref, alt = position;
            if refCounter < len(matrix):
                matrix[refCounter]['ALT'][i] += [alt];
            else:
                break;
            if ref != '-':
                refCounter += 1;

    return matrix;

def processGroupedVariants(variants):
    """
    Converts grouped variants to VCF-like records
    """
    refs    = [];
    alts    = [];
    gts     = [];
    offsets = [];

    for variant in variants:
        refSequence = ''.join(reduce(lambda x, y : x + y, [v[1] for v in variant], []));
        alt0Seq     = ''.join(reduce(lambda x, y : x + y, [v[2][0] for v in variant], []));
        alt1Seq     = ''.join(reduce(lambda x, y : x + y, [v[2][1] for v in variant], []));
        gt0, gt1    = None, None;

        if alt0Seq == refSequence:
            gt0 = 0;
        else:
            gt0 = 1;

        if alt1Seq == refSequence:
            gt1 = 0;
        else:
            if alt0Seq == refSequence:
                gt1 = 1;
            elif alt0Seq == alt1Seq:
                gt1 = 1;
            else:
                gt1 = 2;

        alt = [];

        if alt0Seq != refSequence:
            alt.append(alt0Seq);
        
        if (alt1Seq != refSequence) and (alt1Seq != alt0Seq):
            alt.append(alt1Seq);

        if len(alt) == 0:
            # It won't be used, but alt will be invalid, so add everything, and mark it 0/0
            alt.append(alt0Seq);
            alt.append(alt1Seq);

        refs.append(refSequence);
        alts.append(alt);
        gts.append((gt0,gt1));
        offsets.append(variant[0][0]);

    return offsets, refs, alts, gts;

def fixEmptyAlleles(chromosome, location, ref, alts, referenceCache):
    """
    Add a base to the left if there is an empty allele
    """
    foundEmptyAllele = False;

    alts = [a.replace('-', '') for a in alts];

    for item in [ref] + alts:
        if len(item) == 0:
            foundEmptyAllele = True;
            break;

    if foundEmptyAllele:
        location   -= 1;
        cache       = ReferenceCache(database=referenceCache) if type(referenceCache) is str else referenceCache;
        cache.chrom = chromosome;
        prependage  = cache[location];
        ref         = prependage + ref;
        alts        = [prependage + a for a in alts];

    return foundEmptyAllele, chromosome, location, ref, alts;

def createVcfRecord(
    chromosome, base, referenceCache, offsets, refs, alts, gts, string="BeamSearchResults", qual=30, qualifier="PASS"
):
    """
    Creates normalized VCF-like records from processed variants
    """
    records = [];

    for offset, ref, alt, gt in zip(offsets, refs, alts, gts):
        changeInAlleles = True;

        # Ensure that there is no empty allele
        _, chromosome, location, ref, alt = fixEmptyAlleles(chromosome, base+offset, ref, alt, referenceCache);

        # If all alleles are identical, skip any parsimony checks
        if len(alt) == 0 or all(a == ref for a in alt):
            pass
        else:
            # Right parsimony
            while changeInAlleles:
                changeInAlleles = False;

                # If right-most bases are identical, remove them and continue to normalize
                rightBase = set();
                rightBase.add(ref[-1]);
                for a in alt: rightBase.add(a[-1]);

                if len(rightBase) == 1:
                    ref = ref[:-1];
                    alt = [a[:-1] for a in alt];
                    changeInAlleles = True;

                # If there is an empty allele, extend all alleles
                _, chromosome, location, ref, alt = fixEmptyAlleles(chromosome, location, ref, alt, referenceCache);
                changeInAlleles = changeInAlleles or _;

            leftParsimonious = False;

            while not leftParsimonious:
                if (len(ref) > 1) and (min([len(a) for a in alt]) > 1):
                    leftBase = set();
                    leftBase.add(ref[0]);
                    for a in alt: leftBase.add(a[0]);

                    if len(leftBase) == 1:
                        location += 1;
                        ref = ref[1:];
                        alt = [a[1:] for a in alt];
                    else:
                        leftParsimonious = True;
                else:
                    leftParsimonious = True;

            entry = "%s\t%d\t.\t%s\t%s\t%f\t%s\t%s\tGT\t%s"%(
                str(chromosome),
                location + 1,
                ref,
                ','.join(alt),
                qual,
                qualifier,  # 'PASS'
                string,
                '/'.join([str(x) for x in gt]),
            );
            records.append(entry);

    return records;

def variantsFromMatrix(matrix):
    """
    Creates variants from matrix - groups consecutive locations with variation and outputs the variants involved
    """
    state   = None;
    variant = [];
    records = [];

    for i, position in enumerate(matrix):
        ref, alts      = position['REF'], position['ALT'];
        deviationFound = False;

        for alt in alts:
            if alt != ref:
                deviationFound = True;
                break;

        if not deviationFound:
            if len(variant) > 0:
                records.append(variant);
                variant = [];
        else:
            variant.append((i, ref, alts));

    if len(variant) > 0:
        records.append(variant);
        variant = [];

    return processGroupedVariants(records);

def variantsFromContigs(refSeq, contigs, refCache, start, chromosome):
    # 1. Align each sequence to reference sequence
    alignments = [];

    for contig in contigs:
        alignment = align.globalms(refSeq, contig, 1, -4, -5, -1)[0];
        alignment = list(zip(alignment[0], alignment[1]));
        alignments.append(alignment);

    if len(alignments) == 1:
        alignments = 2 * alignments;

    # 2. Create a matrix into which we fill the alignment entries
    matrix = createRefMatrix(alignments, refSeq);
    
    # 3. Create variant groups
    unnormalized = variantsFromMatrix(matrix);

    # 4. Normalize variants
    normalized =  createVcfRecord(chromosome, start, refCache, *unnormalized);

    return normalized;

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(message)s');

    parser = argparse.ArgumentParser(description = "Create VCF entries from a contig file");
    
    parser.add_argument(
        "--contig",
        help = "Contig file",
        required = True,
    );
    
    parser.add_argument(
        "--referenceCache",
        help = "Reference cache",
        required = True,
    );

    parser.add_argument(
        "--thresholdMethod",
        help = "Type of thresholding method to use",
        choices = ["fraction", "log"],
        default = "log",
    );
    
    parser.add_argument(
        "--threshold",
        help = "Cutoff for contigs",
        type = float,
        default = -3,
    );

    parser.add_argument(
        "--model",
        help = "NN/sklearn model for cutoff",
        default = None,
    );

    parser.add_argument(
        "--rf",
        default = False,
        action = "store_true",
        help = "Random forest classifier to be used for thresholding",
    );

    parser.add_argument(
        "--rerankRf",
        action = "store",
        default = None,
        help = "Random forest classifier for reranking if provided",
    );
    
    args = parser.parse_args();

    cache = ReferenceCache(database = args.referenceCache);

    model = None;
    modelRerank = None;

    if args.model is not None:
        if args.rf:
            model = pickle.load(open(args.model, 'rb'));
        else:
            model = HaplotypeAnalyzer();
            with open(args.model, 'rb') as fhandle:
                params = torch.load(fhandle, map_location='cpu');
                model.load_state_dict(params);
            model.train(False);

    if args.rerankRf is not None:
        modelRerank = pickle.load(open(args.rerankRf, 'rb'));

    with open(args.contig, 'r') as fhandle:
        for line in fhandle:
            dict_ = ast.literal_eval(line);
            chromosome  = dict_['chromosome'];
            cache.chrom = chromosome;
            location    = dict_['region'][0];
            refSeq      = ''.join(cache[location:dict_['region'][1]]);

            if (args.rerankRf is not None) and (len(dict_['contigs']) > 1):
                topTwo = rerank(dict_, modelRerank);
            else:
                topTwo = sorted(dict_['contigs'][:2], reverse = True);

            passing     = [];

            if args.thresholdMethod == 'fraction':
                if args.model is None:
                    topTwo = convertToFraction(topTwo) if args.thresholdMethod == "fraction" else topTwo;
                else:
                    if len(topTwo) == 2:
                        (s0, c0), (s1, c1) = topTwo;
                        maxS = max(s0, s1);
                        topTwo = [(s0-maxS, c0), (s1-maxS, c1)];
                    elif len(topTwo) == 1:
                        topTwo = [(0, topTwo[0][1])];

            if len(topTwo) == 0:
                logging.warning("-WARNING- Cannot find any contigs in region %s"%(str((chromosome, dict_['region']))));

            if model is None:
                for score, contig in topTwo:
                    if score > args.threshold:
                        passing.append(contig);
            else:
                if len(topTwo) > 1:
                    passing = determinePassingModel(dict_, topTwo, args.rf);
                else:
                    if len(topTwo) == 1:
                        passing = [topTwo[0][1]];
                    else:
                        passing = [];

            if len(passing) > 0:
                normalized = variantsFromContigs(refSeq, passing, args.referenceCache, location, chromosome);
                for n in normalized: print(n);
