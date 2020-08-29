import logging
import libCallability
import copy
import itertools

MAX_SEGMENT_SIZE = 10 


class RegionTooLongException(Exception):
    def __init__(self, message):
        super().__init__(message)


def unique_list(items):
    itemset = set()
    unique = []

    for i in items:
        if i not in itemset:
            unique.append(i)

        itemset.add(i)

    return unique


def last_position(record):
    return record.position + len(record.ref)


class ReferenceSegment:
    def __init__(self, segment, start):
        self.segment = segment
        self.start = start

    @property
    def last_position(self):
        return self.start + len(self.segment)

    def __len__(self):
        return len(self.segment)

    def __getitem__(self, index):
        if type(index) is slice:
            new_start = index.start - self.start if (index.start is not None) else None
            new_stop = index.stop - self.start if (index.stop is not None) else None
            index = slice(new_start, new_stop, index.step)
        elif type(index) is int:
            index -= self.start
        else:
            # raise ValueError("Unsupported index type " + str(type(index)))
            logging.error("Something wrong happened with indexing")
            raise RegionTooLongException("Something wrong happened with indexing")

        return self.segment[index]


def deduplicate_ground_truth_haplotypes(results):
    """
    Remove duplicate haplotype pairs

    :param results: list
        List of haplotype pairs from ground-truth enumeration
    """
    deduplicated = []
    prev = set()

    invert = lambda x: (x[1], x[0])

    for h, n1, n2 in results:
        if (h not in prev) and (invert(h) not in prev):
            deduplicated.append((h, n1, n2))
        
        prev.add(h)

    return deduplicated


def enumerate_haplotypes(
    variant_set, ref, anchor, call_level=0, call_type='ground_truth'
):
    """
    Enumerate haplotypes for ground truths or for candidate set

    :param variant_set: list
        List of records

    :param ref: ReferenceSegment
        Reference indexing object

    :param anchor: int
        Left anchor point for invocation

    :param call_level: int
        Indicate recursive depth for this invocation

    :return: list/dict
        List of tuples: ((haplotype0, haplotype1), num_missed alts) for ground-truths
        map: candidate_haplotype_pair => variant record list
    """
    if not variant_set:
        if call_type == 'ground_truth':
            return [((ref[anchor:], ref[anchor:]), 0, 0)]
        else:
            return {
                (ref[anchor:], ref[anchor:]): []
            }

    prefix = ref[anchor: variant_set[0].position] if call_level == 0 else ""

    recursed = enumerate_haplotypes(
        variant_set[1: ],
        ref,
        variant_set[0].position + len(variant_set[0].ref),
        call_level=call_level + 1,
        call_type=call_type,
    )

    genotypes = variant_set[0].gt
    alts = set(genotypes) - {0}
    all_alleles = [variant_set[0].ref] + variant_set[0].alt
    true_alleles = [all_alleles[i] for i in genotypes]

    if len(variant_set) > 1:
        addendum = ref[variant_set[0].position + len(variant_set[0].ref): variant_set[1].position]
    else:
        addendum = ""

    if call_type == 'ground_truth':
        gts_for_site = [(0, 0), genotypes]

        if genotypes[0] != 0:
            gts_for_site.append((0, genotypes[1]))

        if genotypes[1] != 0:
            gts_for_site.append((genotypes[0], 0))
    else:
        gts_for_site = list(itertools.product(range(len(all_alleles)), range(len(all_alleles))))

    if call_type != 'ground_truth':
        iterand = recursed.items()
        results = dict()
    else:
        iterand = iter(recursed)
        results = []

    for stuff in iterand:
        if call_type == 'ground_truth':
            haplotype_pair, num_missed, num_extra = stuff
        else:
            haplotype_pair, selected_alleles_prev = stuff

        haplotype0, haplotype1 = haplotype_pair

        for gt in gts_for_site:
            selected_alleles = [all_alleles[i] for i in gt]

            if len(selected_alleles) == 1:
                selected_alleles = [selected_alleles[0], selected_alleles[0]]

            if len(selected_alleles) == 0:
                logging.warning("Site cannot be labeled due to empty variant record")
                raise ValueError("Site cannot be labeled due to empty variant record")

            new_haplotype0 = prefix + selected_alleles[0] + addendum + haplotype0
            new_haplotype1 = prefix + selected_alleles[1] + addendum + haplotype1

            if call_type == 'ground_truth':
                num_missed_new = num_missed + len(set(true_alleles) - set(selected_alleles))
                num_extra_new = num_extra + len(set(selected_alleles) - set(true_alleles))
                results.append(
                    ((new_haplotype0, new_haplotype1), num_missed_new, num_extra_new)
                )
            else:
                new_selected_alleles = [tuple(selected_alleles)] + selected_alleles_prev
                results[(new_haplotype0, new_haplotype1)] = new_selected_alleles

    logging.debug("Returning from level %d with %d items" % (call_level, len(results)))

    return results


class Labeler:
    def __init__(self, truths, segment, start):
        self.ref = ReferenceSegment(segment, start)
        self.truths = truths
        self._construct_gt_haplotypes()

        if len(self.truths) > 0 and last_position(self.truths[-1]) > self.ref.last_position:
            raise RegionTooLongException("Something wrong happened with indexing")

    def _construct_gt_haplotypes(self):
        if len(self.truths) > MAX_SEGMENT_SIZE:
            logging.warning("Too many truth variants to evaluate")
            raise RegionTooLongException("Too many truth variants to evaluate")

        logging.debug("Evaluating ground-truth candidates with %d records" % len(self.truths))

        self.gt_candidates = deduplicate_ground_truth_haplotypes(
            enumerate_haplotypes(self.truths, self.ref, self.ref.start, call_type='ground_truth')
        )

        logging.debug("Got ground-truth candidates %s" % (str(self.gt_candidates)))

    def __call__(self, candidate_records):
        if len(candidate_records) > MAX_SEGMENT_SIZE:
            logging.warning("Too many candidates to evaluate")
            raise RegionTooLongException("Too many candidates to evaluate")

        if len(candidate_records) > 0 and last_position(candidate_records[-1]) > self.ref.last_position:
            raise RegionTooLongException("Too many candidates to evaluate")

        site_records = [
            libCallability.SiteRecord(
                [r.ref] + r.alt,
                r.position,
                r.position + len(r.ref)
            ) for r in candidate_records
        ]

        trie = libCallability.VariantTrie(site_records, self.ref.segment, self.ref.start)

        for ((h0, h1), n1, n2) in self.gt_candidates:
            trie.search_haplotype_pair(h0, h1, n1, n2)

        if trie.success():
            allele_candidates = trie.get_best_matching_variants()
            alleles = []

            for item in allele_candidates:
                alleles.append(
                    (str(item.first), str(item.second))
                )

            total_errors = int(trie.num_errors.first) + int(trie.num_errors.second)

            if len(candidate_records) == 1 and total_errors > 0:
                return False, []
            else:
                return True, alleles
        else:
            return False, []

        # haplotypes = enumerate_haplotypes(
        #     candidate_records,
        #     self.ref,
        #     self.ref.start,
        #     call_type='candidates'
        # )

        # # logging.debug("Got haplotype candidates %s" % str(haplotypes))

        # matches = []

        # for ((h0, h1), n1, n2) in self.gt_candidates:
        #     if (h0, h1) in haplotypes:
        #         matches.append((n1, n2, (h0, h1)))

        # if len(matches) == 0:
        #     return False, []

        # best_match = min(matches, key=lambda x: (x[0], x[1]))

        # n1, n2, (h0, h1) = best_match

        # # If there is only one candidate record, and we are mislabeling it,
        # # do not use it
        # if len(candidate_records) == 1 and (n1 + n2 > 0):
        #     return False, []

        # if (h0, h1) in haplotypes:
        #     return True, haplotypes[(h0, h1)]
        # elif (h1, h0) in haplotypes:
        #     return True, haplotypes[(h1, h0)]
        # else:
        #     return False, []
