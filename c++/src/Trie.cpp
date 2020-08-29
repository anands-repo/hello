#include "Trie.h"
#include <iostream>

VariantTrie::VariantTrie(
    const p::list& variant_records,
    const string& segment,
    long left
) : records(list_converter<SiteRecord>(variant_records)),
    ref(make_shared<Reference>(segment, left)),
    num_errors(pair<long, long>(MAX_NUM_ERRORS, MAX_NUM_ERRORS)),
    segment_start(left),
    segment_stop(left + segment.size())
{ }

bool VariantTrie::search_path(
    const string& path,
    vector<AllelicRecord>& results,
    long ref_ptr,
    long record_ptr,
    long path_ptr
) {
    if (this->records.empty()) {
        return (
            get_reference_bases(
                *ref,
                segment_start,
                segment_stop
            ) == path
        );
    }

    if ((ref_ptr == segment_stop) && (path_ptr == path.size())) return true;
    if ((ref_ptr >= segment_stop)) return false;
    if ((path_ptr >= path.size())) return false;

    const SiteRecord* next_site_record = record_ptr < records.size() ? &records[record_ptr]: nullptr;

    if (!next_site_record) {
        while (path_ptr < path.size()) {
            if (ref_ptr >= segment_stop)
                return false;

            if (path[path_ptr++] != (*ref)[ref_ptr++])
                return false;
        }

        if (ref_ptr != segment_stop)
            return false;
    } else {
        while (ref_ptr < next_site_record->start) {
            if (path_ptr >= path.size())
                return false;

            if (path[path_ptr++] != (*ref)[ref_ptr++])
                return false;
        }

        // For each case that works, recurse. If the recursion succeeds, stop
        // If the recursion fails, remove the added allele
        bool successful(false);

        for (auto& allele: next_site_record->alleles) {
            if (path.substr(path_ptr, allele.size()) == allele) {
                AllelicRecord record(allele, next_site_record->start, next_site_record->stop, 50);
                results.emplace_back(move(record));
                bool flag = search_path(
                    path,
                    results,
                    next_site_record->stop,
                    record_ptr + 1,
                    path_ptr + allele.size()
                );

                if (flag) {
                    successful = true;
                    break;
                } else {
                    results.pop_back();
                }
            }
        }

        if (!successful)
            return false;
    }

    return true;
}

void VariantTrie::search_haplotype_pair(const string& h0, const string& h1, long num_missing, long num_extra) {
    vector<AllelicRecord> h0_results;
    vector<AllelicRecord> h1_results;
    pair<long, long> num_errors_for_haplotypes(num_missing, num_extra);

    if (num_errors_for_haplotypes > num_errors) return;

    if (search_path(h0, h0_results, segment_start, 0, 0) && search_path(h1, h1_results, segment_start, 0, 0)) {
        num_errors = num_errors_for_haplotypes;
        min_haplotypes.first = h0_results;
        min_haplotypes.second = h1_results;
    }
}

vector<pair<string, string> > VariantTrie::get_best_matching_variants() {
    vector<pair<string, string> > results;

    if (num_errors.first < MAX_NUM_ERRORS) {
        for (long i = 0; i < min_haplotypes.first.size(); i++) {
            const auto& allele0 = min_haplotypes.first[i].allele;
            const auto& allele1 = min_haplotypes.second[i].allele;
            results.push_back(make_pair(allele0, allele1));
        }
    }

    return results;
}

bool VariantTrie::success() {
    return (num_errors.first < MAX_NUM_ERRORS);
}
