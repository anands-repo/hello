#include "Read.h"

void Read::_get_read_mapping() {
    long ref_ptr = this->reference_start;
    long rd_ptr = 0;
    long cigar_count = 0;

    for (auto& cigar: cigartuples) {
        const auto& operation = cigar.first;
        const auto& length = cigar.second;

        switch(operation) {
            case BAM_CEQUAL:
            case BAM_CDIFF:
            case BAM_CMATCH: {
                for (int i = 0; i < length; i++) {
                    aligned_pairs[ref_ptr + i] = read.substr(rd_ptr + i, 1);
                    aligned_qualities[ref_ptr + i] = quality[rd_ptr + i];
                    last_position = ref_ptr + i;
                }
                rd_ptr += length;
                ref_ptr += length;
                break;
            }

            case BAM_CDEL: {
                for (int i = 0; i < length; i++) {
                    aligned_pairs[ref_ptr + i] = "";
                    aligned_qualities[ref_ptr + i] = 60;
                    last_position = ref_ptr + i;
                }
            }
            case BAM_CREF_SKIP: {
                ref_ptr += length;
                break;
            }

            case BAM_CINS: {
                // Add read bases to the mapping position
                string rd_string(read.substr(rd_ptr, length));
                if (aligned_pairs.find(ref_ptr - 1) == aligned_pairs.end()) {
                    aligned_pairs[ref_ptr - 1] = rd_string;
                    partial_start = true;
                } else {
                    aligned_pairs[ref_ptr - 1] += rd_string;
                    if (cigar_count == cigartuples.size() - 1) {
                        partial_stop = true;
                    }
                }

                // Find aggregate minimum quality for insertion
                long min_quality_added = *std::min_element(
                    quality.begin() + rd_ptr,
                    quality.begin() + rd_ptr + length
                );

                if (aligned_qualities.find(ref_ptr - 1) == aligned_qualities.end()) {
                    aligned_qualities[ref_ptr - 1] = min_quality_added;
                } else {
                    aligned_qualities[ref_ptr - 1] = std::min(
                        {
                            aligned_qualities[ref_ptr - 1],
                            min_quality_added
                        }
                    );
                }
            }
            case BAM_CSOFT_CLIP: {
                rd_ptr += length;
                break;
            }
        }

        cigar_count++;
    }
}

AlignedBases Read::get_aligned_bases(long start, long stop) const {
    AlignedBases result;

    long min_pos = reference_start;
    long max_pos = last_position;

    // If no overlap between the provided intervals
    // and the read intervals on the reference, then this
    // fails to provide any bases
    if ((!((start <= max_pos) && (min_pos < stop))) || (last_position == -1)){
        result.second = Fail;
        return result;
    }

    // Decode cases of partial matches and overlaps
    if (aligned_pairs.find(start) == aligned_pairs.end()) {
        result.second = LeftPartial;
    } else if (aligned_pairs.find(start - 1) == aligned_pairs.end()) {
        result.second = partial_start ? LeftPartial : Success;
    } else if (aligned_pairs.find(stop - 1) == aligned_pairs.end()) {
        result.second = RightPartial;
    } else if (aligned_pairs.find(stop) == aligned_pairs.end()) {
        result.second = partial_stop ? RightPartial : Success;
    } else {
        result.second = Success;
    }

    // Check wether there is a deletion at the start or the end position
    if (aligned_pairs.find(start) != aligned_pairs.end()) {
        if (aligned_pairs.find(start)->second.size() == 0) {
            result.second = Fail;
        }
    }

    if (aligned_pairs.find(stop  - 1) != aligned_pairs.end()) {
        if (aligned_pairs.find(stop - 1)->second.size() == 0) {
            result.second = Fail;
        }
    }

    // Collect base strings
    long min_quality = 10000;
    ostringstream sstr;
    sstr << "";

    for (long i = start; i < stop; i++) {
        if (aligned_pairs.find(i) != aligned_pairs.end()) {
            sstr << aligned_pairs.find(i)->second;
        }
        if (aligned_qualities.find(i) != aligned_qualities.end()) {
            min_quality = min_quality < aligned_qualities.at(i) ? min_quality : aligned_qualities.at(i);
        }
    }

    result.first = sstr.str();
    result.third = min_quality;

    return result;
}

void Read::extract_alleles(const vector<pair<size_t, size_t>>& locations) {
    if (assembled) return;

    for (auto& location: locations) {
        auto result = this->get_aligned_bases(location.first, location.second);

        AllelicRecord record(
            result.first,
            location.first,
            location.second,
            result.third
        );

        switch(result.second) {
            case AlignedBaseStatus::Success: {
                this->alleles.emplace_back(move(record));
                break;
            }
            case AlignedBaseStatus::LeftPartial: {
                this->left_partial = record;
                this->has_left_partial = true;
                break;
            }
            case AlignedBaseStatus::RightPartial: {
                this->right_partial = record;
                this->has_right_partial = true;
                break;
            }
            default: {}
        }
    }

    assembled = true;
}

string Read::get_haplotype_string(const Reference& ref, long start, long stop) {
    long allele_ptr = 0;
    long refptr = start;

    if (this->alleles.empty()) {
        return get_reference_bases(ref, start, stop);
    } else {
        string haplotype("");

        const auto& first_allele = this->alleles.front();

        if (first_allele.start > start) {
            haplotype += get_reference_bases(ref, start, first_allele.start);
            haplotype += first_allele.allele;
        }

        for (size_t i = 1; i < this->alleles.size(); i++) {
            const auto& current_allele = this->alleles[i];
            const auto& prev_allele = this->alleles[i - 1];
            haplotype += get_reference_bases(ref, prev_allele.stop, current_allele.start);
            haplotype += current_allele.allele;
        }

        if (this->alleles.back().stop < stop) {
            haplotype += get_reference_bases(ref, this->alleles.back().stop, stop);
        }

        return haplotype;
    }
}

void Read::update_allelic_records(
    const Reference& ref,
    const unordered_map<string, vector<AllelicRecord>>& repr,
    long start,
    long stop
) {
    // If the read doesn't span the full start to stop
    // segment, then do not continue
    if (start > this->reference_start) return;
    if (this->last_position < stop) return;

    string haplotype = this->get_haplotype_string(ref, start, stop);

    for (auto& r: repr) {
        if (r.first == haplotype) {
            auto new_allele_list = r.second;
            // Remove spurious first element
            if (new_allele_list.front().start == -1) {
                new_allele_list.erase(new_allele_list.begin());
            }
            this->alleles = new_allele_list;
            break;
        }
    }
}

void Read::create_allele_map() {
    for (auto& record: this->alleles) {
        this->allele_map[
            pair<long, long>(record.start, record.stop)
        ] = record.allele;
    }
}

void enumerate_all_haplotypes(
    vector<SiteRecord>& site_records,
    const Reference& ref,
    long start,
    long stop,
    unordered_map<string, vector<AllelicRecord>>& result,
    int call_level
) {
    // Base case ... nothing to do
    if (site_records.empty()) {
        AllelicRecord dummy(
            "",
            -1,
            start,
            60
        );
        vector<AllelicRecord> dummy_;
        dummy_.push_back(dummy);
        result[""] = dummy_;
        return;
    }

    // Obtain the last item
    auto last_record = site_records.back();
    site_records.pop_back();

    // Call enumeration on remaining items
    enumerate_all_haplotypes(
        site_records,
        ref,
        start,
        stop,
        result,
        call_level + 1
    );

    // Purge all previous results with this one
    unordered_map<string, vector<AllelicRecord>> result_partial;

    // Expand items with alleles at the present site
    for (auto& allele_string: last_record.alleles) {
        AllelicRecord record(allele_string, last_record.start, last_record.stop, 60);

        // For each result, append 
        for (auto& item: result) {
            string haplotype = item.first;
            vector<AllelicRecord> allele_record_list = item.second;

            // Collect reference bases from the previous allele position
            // to the present one. If there is no previous allele position,
            // collect bases from the start of the segment
            long previous = allele_record_list.empty() ? start : allele_record_list.back().stop;

            string ref_string = get_reference_bases(
                ref,
                previous,
                record.start
            );

            // Append collected reference bases and allelic bases
            // to the previous version of the haplotype
            haplotype += ref_string;
            haplotype += record.allele;

            // Update the corresponding list of alleles to contain
            // the record being used to create this haplotype candidate
            allele_record_list.push_back(record);

            // If this is the root call, we also need to append the end of
            // the reference segment in the window to the haplotype
            if (call_level == 0) {
                string ref_addendum = get_reference_bases(ref, record.stop, stop);
                haplotype += ref_addendum;
            }

            // Record the new results
            result_partial[haplotype] = allele_record_list;
        }

    }

    // Remove stale results and update new ones
    result.clear();
    result.insert(result_partial.begin(), result_partial.end());
}

string get_reference_bases(const Reference& ref, long start, long stop) {
    ostringstream sstr;

    for (long i = start; i < stop; i++) {
        sstr << ref[i];
    }

    return sstr.str();
}

#ifndef _READ_TEST_
TruthSet get_ground_truth_alleles(
    const p::list& truth_records,
    const string& reference_segment,
    const string& haplotype0,
    const string& haplotype1,
    long left_position
) {
    // Prepare all data for enumerating candidates
    Reference ref(reference_segment, left_position);
    vector<SiteRecord> truth_records_cpp(list_converter<SiteRecord>(truth_records));
    unordered_map<string, vector<AllelicRecord> > candidates;

    // Enumerate all candidate alleles
    enumerate_all_haplotypes(
        truth_records_cpp,
        ref,
        left_position,
        left_position + reference_segment.size(),
        candidates,
        0
    );

    for (auto& item: candidates) {
        cout << "Found candidate haplotype" << item.first << endl;
    }

    TruthSet results;

    // If neither ground-truth haplotype is found, reject the labeling
    if (candidates.find(haplotype0) == candidates.end()) {
        results.valid = false;
        return results;
    }

    if (candidates.find(haplotype1) == candidates.end()) {
        results.valid = false;
        return results;
    }

    // Collect the candidate alleles
    results.valid = true;

    vector<string> candidates0;
    vector<string> candidates1;

    for (auto& truth_allele: candidates.find(haplotype0)->second) {
        candidates0.push_back(truth_allele.allele);
    }

    for (auto& truth_allele: candidates.find(haplotype1)->second) {
        candidates1.push_back(truth_allele.allele);
    }

    for (long i = 0; i < candidates0.size(); i++) {
        const string& a0 = candidates0[i];
        const string& a1 = candidates1[i];
        results.truth_alleles.push_back(
            pair<string, string>(a0, a1)
        );
    }

    return results;
}
#endif

#ifdef _READ_TEST_
int main() {
    /*
    ---- View 1 ----
    Positions:  0             14 17      24
    Reference:  ATACGACTACGACTACGA-TTTTTTTTACGACTTACGACTACGACTGGGACTACGATTAACA
    Read:                     ACGACTTTTTTT-ACGACTTACGACTAC

    ---- View 2 ----
    Positions:  0             14  18
    Reference:  ATACGACTACGACTACGATTTTTTTTACGACTTACGACTACGACTGGGACTACGATTAACA
    Read:                     ACGACTTTTTTTACGACTTACGACTAC

    -----True ------
    Haplotype:  ATACGACTACGACTACGACTTTTTTTACGACTTACGACTACGACTGGGACTACGATTAACA
                ATACGACTACGACTACGACTTTTTTTACGACTTACGACTACGACTGGGACTACGATTAACA
    */

    string ref_string("ATACGACTACGACTACGATTTTTTTTACGACTTACGACTACGACTGGGACTACGATTAACA");
    string read_string("ACGACTTTTTTTACGACTTACGACTAC");
    vector<pair<size_t, size_t>> cigartuples;

    // The following cigartuple is based on view 2
    cigartuples.push_back(
        make_pair<size_t, size_t>(BAM_CMATCH, read_string.size())
    );

    // The following are cigartuples based on view 1
    // cigartuples.push_back(
    //     make_pair<size_t, size_t>(BAM_CMATCH, 4)
    // );
    // cigartuples.push_back(
    //     make_pair<size_t, size_t>(BAM_CINS, 1)
    // );
    // cigartuples.push_back(
    //     make_pair<size_t, size_t>(BAM_CMATCH, 7)
    // );
    // cigartuples.push_back(
    //     make_pair<size_t, size_t>(BAM_CDEL, 1)
    // );
    // cigartuples.push_back(
    //     make_pair<size_t, size_t>(BAM_CMATCH, 15)
    // );

    vector<size_t> quality;
    for (auto& i: read_string) quality.push_back(60);

    long reference_start(14);
    bool pacbio(true);
    int read_id(1);
    long mapq(60);

    Read read(
        read_string,
        quality,
        cigartuples,
        reference_start,
        pacbio,
        read_id,
        mapq
    );

    vector<pair<size_t, size_t>> differing_regions;
    differing_regions.push_back(
        pair<size_t, size_t>(18, 19)
    );
    Reference ref(ref_string, 0);
    read.extract_alleles(differing_regions);

    for (auto& item: read.alleles) {
        cout << "Original read allele " << item.allele << ", " << item.start << ", " << item.stop << endl;
    }

    // Create site-records with view 1
    vector<SiteRecord> sites;
    vector<string> alleles0;
    alleles0.push_back(
        "AT"
    );
    alleles0.push_back(
        "ACT"
    );
    sites.push_back(
        SiteRecord(
            alleles0,
            17,
            19
        )
    );

    vector<string> alleles1;
    alleles1.push_back(
        "TTA"
    );
    alleles1.push_back(
        "TA"
    );
    sites.push_back(
        SiteRecord(
            alleles1,
            24,
            27
        )
    );

    for (auto& site_item: sites) {
        ostringstream site_alleles;

        for (auto& a: site_item.alleles) {
            site_alleles << a << ", ";
        }

        cout << "Site allele: " << site_item.start << ", " << site_item.stop << ", " << site_alleles.str() << endl;
    }

    unordered_map<string, vector<AllelicRecord>> result;
    enumerate_all_haplotypes(sites, ref, 0, ref_string.size(), result);

    for (auto& item: result) {
        cout << "Result :" << item.first << endl;
    }

    read.update_allelic_records(ref, result, 0, ref_string.size());

    for (auto& item: read.alleles) {
        cout << "New allele " << item.allele << ", " << item.start << ", " << item.stop << endl;
    }

    return -1;
}
#endif