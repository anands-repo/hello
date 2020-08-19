#include "AlleleSearcherLiteFiltered.h"
#include "leftAlignCigars.h"
#include "Arrays.h"

const unordered_map<string, size_t> AlleleSearcherLiteFiltered::BASE_TO_INDEX = unordered_map<string, size_t>
                                                                            ({ {"A", 0}, {"C", 1}, {"G", 2}, {"T", 3}, {"N", 4} });

string printCigar(const vector<pair<size_t, size_t> >& cigars)
{
    ostringstream sstr;
    for (auto& cigar : cigars)
    {
        sstr << "(" << cigar.first << ", " << cigar.second << "), ";
    }
    return sstr.str();
}

void AlleleCounts::resolvePartials()
{
    PartialMatchTracker partialMatches;

    // Determine (refAllele, readAllele) key which matches the allele
    auto determineMatchingAlleles = [] (
        PartialMatchTracker& matches,
        const pair<string, string>& altAlleleKey,
        const Counts& counts,
        bool left
    ) {
        const string& altAllele = altAlleleKey.second;

        // Note altAllele.first is simply going to be a
        // single reference base since partial alleles simply apply to
        // insertions

        for (auto& item: counts) {
            const string& fullAltAllele = item.first.second;
            string slice;

            if (fullAltAllele.size() < altAllele.size()) {
                continue;
            }

            if (left) {
                // Left partial implies the left "half" of the allele is not known. Hence, we
                // compare the partial alt allele to the right "half" of the full alt allele
                slice = fullAltAllele.substr(fullAltAllele.size() - altAllele.size(), altAllele.size());
            } else {
                // Right partial implies the right "half" of the allele is not known. Hence, we
                // compare the partial alt allele to the left "half" of the full alt allele
                slice = fullAltAllele.substr(0, altAllele.size());
            }

            if (slice == altAllele) {
                matches[altAlleleKey].insert(item.first);
            }
        }
    };

    // Resolve a partially matching allele
    auto resolvePartialMatches = [] (
        Counts& counts,
        const PartialMatchTracker& matches,
        Counts& source
    ) {
        for (auto& item: matches) {
            const auto& altAlleleKey = item.first;
            auto& altAllele = altAlleleKey.second;
            auto& keys = item.second;
            if (keys.size() == 1) {
                auto key = *keys.begin();
                counts[key] += source[altAlleleKey];
            } else if (keys.size() == 0) {
                counts[altAlleleKey] = source[altAlleleKey];
            }
        }
    };

    for (auto& a : this->leftPartialAltCounts) {
        determineMatchingAlleles(
            partialMatches,
            a.first,
            this->altCounts,
            true
        );
    }
    resolvePartialMatches(this->altCounts, partialMatches, this->leftPartialAltCounts);
    this->leftPartialAltCounts.clear();
    partialMatches.clear();

    for (auto& a : this->rightPartialAltCounts) {
        determineMatchingAlleles(
            partialMatches,
            a.first,
            this->altCounts,
            false
        );
    }
    resolvePartialMatches(this->altCounts, partialMatches, this->rightPartialAltCounts);
    this->rightPartialAltCounts.clear();
}

// Expand a given region so that adjacent indels are absorbed into the region
pair<size_t, size_t> AlleleSearcherLiteFiltered::expandRegion(size_t x, size_t y)
{
    pair<size_t, size_t> modified(x, y);
    // This is a dummy place-holder function. We do not need to use it anymore.
    return modified;
}

// Convert quality in list format to vector format
void AlleleSearcherLiteFiltered::listToQuality(const p::list& data)
{
    for (size_t i = 0; i < (size_t) p::len(data); i++)
    {
        vector<size_t> quality = LIST_TO_SIZET_VECTOR(p::list(data[i]));
        this->qualities.emplace_back(move(quality));
    }
}

void AlleleSearcherLiteFiltered::updateAlleleCounts()
{
    DEBUG << "Starting read count updates";

    for (size_t i = 0; i < this->reads.size(); i++)
    {
        const auto& read = this->reads[i];
        const auto& qual = this->qualities[i];
        const auto& cigartuple = this->cigartuples[i];
        long rfcounter = long(this->referenceStarts[i]) - long(this->windowStart);
        long rdcounter = 0;
        auto& counts = this->pacbio[i] ? this->counts_p: this->counts_i;

        if (this->mapq[i] < this->minMapQ) {
            continue;
        }

        assert(rfcounter > 0);

        // Quality control and allele addition function
        auto addToCount = [this] (
            AlleleCounts& count,
            const string& refAllele,
            const string& altAllele,
            const vector<size_t>& quality,
            long rdcounter,
            long rdlength,
            size_t threshold,
            bool partial = false,
            bool leftPartial = false,
            int increment = 1
        ) {
            pair<string, string> allele(refAllele, altAllele);

            if (
                (rdcounter < 0) || 
                (*min_element(quality.begin() + rdcounter, quality.begin() + rdcounter + rdlength) >= threshold)
            ) {
                auto& altCounts = partial ? 
                    (leftPartial ? count.leftPartialAltCounts : count.rightPartialAltCounts) : count.altCounts;
                if (altCounts.find(allele) != altCounts.end()) {
                    altCounts[allele] += increment;
                } else {
                    altCounts[allele] = increment;
                }

                DEBUG << "Updated altcount at position " << count.pos << " to value " << altCounts[allele] << ", ref, alt = " << refAllele << ", " << altAllele << ", refcount = " << count.refCount << ", total = " << count.total;
            }
        };

        size_t cigarcount = 0;

        for (auto& cigar : cigartuple)
        {
            long operation = cigar.first;
            long length = cigar.second;

            switch(operation)
            {
                case BAM_CEQUAL:
                case BAM_CDIFF:
                case BAM_CMATCH:
                {
                    for (size_t j = 0; j < length; j++)
                    {
                        auto& count = counts[rfcounter + j];

                        if (read[rdcounter + j] != this->reference[rfcounter + j]) {
                            string refAllele = this->reference.substr(rfcounter + j, 1);
                            string readAllele = read.substr(rdcounter + j, 1);
                            addToCount(
                                count, refAllele, readAllele, qual, rdcounter + j, 1, this->qThreshold
                            );
                        } else {
                            count.refCount += 1;
                        }

                        count.total += 1;   // We use a second set of counts to track totals. This is because,
                                            // indel counts for Illumina are 2x that of PacBio. Hence using refCount
                                            // and altCount to determine totals is incorrect, since some altCounts do not reflect
                                            // the actual coverage at a site. Also, the total number of bases aligning to a position
                                            // are determined by the number of match/mismatch bases and it is not dependent on the insertion/deletion
                                            // bases at a site. This is because indels are impinged upon the position to the left of the indel cigar, and
                                            // that position already has a count for the base since that position has a cigar value.
                                            // The only exception to this rule is when there is a left partial indel at a position, in which case also
                                            // we increment the count. Note that left partials can occur due to indel realignment.
                    }

                    rdcounter += length;
                    rfcounter += length;
                    break;
                }

                case BAM_CDEL:
                {
                    // Plant this on the location just before the deletion
                    auto& count = counts[rfcounter - 1];
                    string refAllele = this->reference.substr(rfcounter - 1, length + 1);
                    string readAllele = rdcounter > 0 ? read.substr(rdcounter - 1, 1) : this->reference.substr(rfcounter - 1, 1);
                    addToCount(
                        count,
                        refAllele,
                        readAllele,
                        qual,
                        rdcounter - 1,
                        1,
                        this->qThreshold,
                        false,
                        false,
                        this->pacbio[i] ? 1: 2
                    );

                    DEBUG << "Updating deletion allele at " << count.pos;
                }
                case BAM_CREF_SKIP:
                {
                    rfcounter += length;
                    break;
                }

                case BAM_CINS:
                {
                    auto& count = counts[rfcounter - 1];
                    string refAllele = this->reference.substr(rfcounter - 1, 1);

                    if (cigarcount == 0) {
                        string readAllele = read.substr(rdcounter, length);
                        addToCount(
                            count,
                            refAllele,
                            readAllele,
                            qual,
                            rdcounter,
                            length,
                            this->qThreshold,
                            true,
                            true,
                            this->pacbio[i] ? 1: 2
                        );

                        // Left partial indel: we must increase count.total for this case
                        // See the match/mismatch case statement for more explanations
                        count.total += 1;
                    } else if ((cigarcount == cigartuple.size() - 1) && (rdcounter > 0)) {
                        string readAllele = read.substr(rdcounter - 1, length + 1);
                        addToCount(
                            count,
                            refAllele,
                            readAllele,
                            qual,
                            rdcounter - 1,
                            length + 1,
                            this->qThreshold,
                            true,
                            false,
                            this->pacbio[i] ? 1: 2
                        );
                    } else {
                        string readAllele;
                        long rdc = rdcounter > 0 ? rdcounter - 1 : rdcounter;
                        long len = rdcounter > 0 ? length + 1 : length;
                        if (rdcounter > 0) {
                            readAllele = read.substr(rdcounter - 1, length + 1);
                        } else {
                            readAllele = refAllele + read.substr(rdcounter, length);
                        }
                        addToCount(
                            count,
                            refAllele,
                            readAllele,
                            qual,
                            rdc,
                            len,
                            this->qThreshold,
                            false,
                            false,
                            this->pacbio[i] ? 1: 2
                        );
                    }
                }
                case BAM_CSOFT_CLIP:
                {
                    rdcounter += length;
                    break;
                }
            }

            cigarcount++;
        }
    }

    DEBUG << "Completed updating allele counts";
}

void AlleleSearcherLiteFiltered::listToCigar(const p::list& cigartuples)
{
    for (size_t i = 0; i < len(cigartuples); i++)
    {
        Cigar cigartuple;
        const auto& cigar = p::list(cigartuples[i]);

        for (size_t j = 0; j < len(cigar); j++)
        {
            size_t operation = p::extract<int>(p::list(cigar[j])[0]);
            size_t length = p::extract<int>(p::list(cigar[j])[1]);
            cigartuple.push_back(pair<size_t, size_t>(operation, length));
        }

        this->cigartuples.emplace_back(move(cigartuple));
    }
}

AlleleSearcherLiteFiltered::AlleleSearcherLiteFiltered(
    const p::list& reads,
    const p::list& names,
    const p::list& qualities,
    const p::list& cigartuples,
    const p::list& referenceStarts,
    const p::list& mapq,
    const p::list& orientation,
    const p::list& pacbio,
    const string& reference,
    size_t windowStart,
    size_t start,
    size_t stop,
    bool hybrid_hotspot
) :
    reads(strListToVector(reads)),
    pacbio(LIST_TO_BOOL_VECTOR(pacbio)),
    names(strListToVector(names)),
    reference(reference),
    referenceStarts(LIST_TO_SIZET_VECTOR(referenceStarts)),
    mapq(LIST_TO_SIZET_VECTOR(mapq)),
    orientation(LIST_TO_INT_VECTOR(orientation)),
    mismatchScore(1),
    insertScore(4),
    deleteScore(4),
    windowStart(windowStart),
    qThreshold(10),
    useMapq(true),
    useOrientation(true),
    useQEncoding(true),
    base_color_offset_a_and_g(40),
    base_color_offset_t_and_c(30),
    base_color_stride(70),
    base_quality_cap(40),
    mapping_quality_cap(60),
    positive_strand_color(70),
    negative_strand_color(240),
    allele_position_color(240),
    background_position_color(70),
    READ_BASE_TRACK(0),
    REF_BASE_TRACK(1),
    READ_QUAL_TRACK(2),
    READ_MAPQ_TRACK(3),
    READ_ORIENTATION_TRACK(4),
    POSITION_MARKER_TRACK(5),
    snvThreshold(0.12),
    indelThreshold(0.12),
    minCount(2),
    minMapQ(10),
    maxAlleleSize(100),
    num_pacbio_reads(0),
    num_illumina_reads(0),
    min_depth_for_pacbio_realignment(20),
    band_margin(6),
    region_start(start),
    region_stop(stop),
    max_reassembly_region_size(10),
    hybrid_hotspot(hybrid_hotspot)
{
    bool leftAlign = false;

    DEBUG << "Creating searcher C++ object with " << this->reads.size() << " reads";

    // Quality to probability for fast computation
    for (size_t q = 0; q < 100; q++)
    {
        this->qualityToProbability[q] = 1 - pow(10, -float(q) / 10.0);
    }

    // Create quality values and reads
    this->listToQuality(qualities);

    // Create cigartuples
    this->listToCigar(cigartuples);

    // Left align cigars if required
    if (leftAlign)
    {
        Reference refObj(this->reference, this->windowStart);

        for (size_t i = 0; i < this->reads.size(); i++)
        {
            leftAlignCigars(
                this->reads[i],
                this->referenceStarts[i],
                this->cigartuples[i],
                refObj,
                true        // Set indelRealigned as "true"
            );

            if (this->pacbio[i]) num_pacbio_reads++;
            if (!this->pacbio[i]) num_illumina_reads++;
        }
    }

    // Initialize allele count object
    for (size_t i = 0; i < this->reference.size(); i++)
    {
        AlleleCounts count_i;
        AlleleCounts count_p;
        count_i.refCount = 0;
        count_i.pos = i + this->windowStart;
        count_i.total = 0;
        count_p.refCount = 0;
        count_p.pos = i + this->windowStart;
        count_p.total = 0;
        this->counts_i.emplace_back(move(count_i));
        this->counts_p.emplace_back(move(count_p));
    }

    // Update all allele counts
    this->updateAlleleCounts();

    // Resolve partial counts
    for (auto& count : this->counts_i) {
        count.resolvePartials();
    }

    // Resolve partial counts
    for (auto& count : this->counts_p) {
        count.resolvePartials();
    }
}

void AlleleSearcherLiteFiltered::prepReadObjs() {
    if (!this->read_objs.empty()) return;

    for (int i = 0; i < this->reads.size(); i++) {
        const auto& read = this->reads[i];
        const auto& quality = this->qualities[i];
        const auto& cigar = this->cigartuples[i];
        const auto& ref_start = this->referenceStarts[i];
        const auto& pacbio = this->pacbio[i];
        const auto& mapq = this->mapq[i];
        const auto& name = this->names[i];
        Read readobj(
            read,
            name,
            quality,
            cigar,
            ref_start,
            pacbio,
            i,
            mapq
        );
        this->read_objs.emplace_back(std::move(readobj));
    }
}

// Push a cluster of alleles into regions, but make sure there are no empty alleles at the start or stop
// position of the cluster. If there is, shift the cluster end-points
void AlleleSearcherLiteFiltered::pushRegions(
    vector<size_t>& cluster, vector<pair<size_t,size_t> >& regions, bool strict
) {
    pair<size_t,size_t> region;
    region.first  = cluster.front();
    region.second = cluster.back() + 1;

    DEBUG << "Received region " << region.first << ", " << region.second;

    // Strict requires only those regions to be pushed which are
    // completely within the searcher regions
    if (strict) {
        if ((region.first < this->region_start) || (region.second > this->region_stop)) {
            DEBUG << "Discarding region";
            cluster.clear();
            return;
        }
    }

    regions.push_back(region);
    cluster.clear();
    DEBUG << "Pushing region " << regions.back().first << ", " << regions.back().second;
}

void AlleleSearcherLiteFiltered::cluster_differing_regions_helper(
    const set<long>& differing_locations,
    vector<pair<size_t, size_t>>& differing_regions,
    bool strict
) {
    // Collect consecutive locations that are affected into a cluster
    vector<size_t> cluster;

    for (auto& location : differing_locations) {
        if (cluster.empty()) {
            cluster.push_back(location);
        }
        else {
            if (cluster.back() == location - 1)
            {
                cluster.push_back(location);
            }
            else
            {
                this->pushRegions(cluster, differing_regions, strict);
                cluster.push_back(location);
            }
        }
    }

    if (!cluster.empty()) {
        this->pushRegions(cluster, differing_regions, strict);
    }
}

// Perform hybrid (combined) differing regions calculations
void AlleleSearcherLiteFiltered::determine_differing_regions_hybrid_helper(
    set<long>& differing_locations
) {
    for (long i = 0; i < this->counts_i.size(); ++i) {
        const auto& count_i = this->counts_i[i];
        const auto& count_p = this->counts_p[i];
        set<pair<string, string> > alt_keys;

        // Collect all allelic keys from both pacbio and Illumina reads
        for (auto& alt: count_i.altCounts) {
            alt_keys.insert(alt.first);
        }

        for (auto& alt: count_i.altCounts) {
            alt_keys.insert(alt.first);
        }

        for (auto& alt: alt_keys) {
            float value_i = 0;
            float value_p = 0;
            float total = count_i.total + count_p.total;
            float ref_count_i = count_i.refCount;
            float ref_count_p = count_p.refCount;
            float alt_count;
            const string& ref_base = alt.first;
            const string& alt_base = alt.second;

            if (total == 0) continue;

            if (count_i.altCounts.find(alt) != count_i.altCounts.end()) {
                value_i = count_i.altCounts.find(alt)->second;
            }

            if (count_p.altCounts.find(alt) != count_p.altCounts.end()) {
                value_p = count_p.altCounts.find(alt)->second;
            }

            if ((ref_base.size() == 1) && (alt_base.size() == 1)) {
                alt_count = value_i + value_p;

                if (
                    ((value_i + value_p) / total >= this->snvThreshold) &&
                    (alt_count >= this->minCount)
                ) {
                    differing_locations.insert(count_i.pos);
                }
            } else {
                alt_count = value_i / 2 + value_p;
                if (
                    ((value_i + value_p) / total >= this->indelThreshold) &&
                    (alt_count >= this->minCount)
                ) {
                    for (long i = count_i.pos; i < count_i.pos + ref_base.size(); i++) {
                        differing_locations.insert(i);
                    }
                }
            } // else
        } // for
    } // for
}

void AlleleSearcherLiteFiltered::determineDifferingRegions(bool strict) {
    CLEAR(this->differingRegions)
    CLEAR(this->differing_regions_i)
    CLEAR(this->differing_regions_p)
    set<long> differing_locations_i;
    set<long> differing_locations_p;
    set<long> differing_locations;

    if (!this->hybrid_hotspot) {
        if ((num_illumina_reads > 0) && (num_pacbio_reads == 0)) {
            this->determine_differing_regions_helper(this->counts_i, differing_locations_i, this->minCount, 2 * this->minCount);
            cluster_differing_regions_helper(differing_locations_i, differing_regions_i, strict);
        } else if ((num_pacbio_reads > 0) && (num_illumina_reads == 0)) {
            this->determine_differing_regions_helper(this->counts_p, differing_locations_p, this->minCount, this->minCount);
            cluster_differing_regions_helper(differing_locations_p, differing_regions_p, strict);
        } else {
            this->determine_differing_regions_helper(this->counts_i, differing_locations_i, this->minCount, 2 * this->minCount);
            this->determine_differing_regions_helper(this->counts_p, differing_locations_p, this->minCount, this->minCount);
            cluster_differing_regions_helper(differing_locations_i, differing_regions_i, strict);
            cluster_differing_regions_helper(differing_locations_p, differing_regions_p, strict);
            std::set_union(
                differing_locations_i.begin(), differing_locations_i.end(),
                differing_locations_p.begin(), differing_locations_p.end(),
                std::inserter(differing_locations, differing_locations.begin())
            );
            cluster_differing_regions_helper(differing_locations, this->differingRegions, strict);
        }
    } else {
        this->determine_differing_regions_hybrid_helper(differing_locations);
        cluster_differing_regions_helper(differing_locations, this->differingRegions, strict);
    }

    for (auto& item: differingRegions) {
        DEBUG << "Found differing region " << item.first << ", " << item.second;
    }
}

void AlleleSearcherLiteFiltered::get_alleles_from_reads(
    map<pair<long, long>, unordered_set<string>>& alleles,
    vector<Read*>& read_objects,
    const vector<pair<size_t, size_t> >& differing_regions
) {
    for (auto& read_object: read_objects) {
        for (auto& record: read_object->alleles) {
            if (
                (record.min_q >= this->qThreshold) &&
                (read_object->mapq >= this->minMapQ) &&
                (record.allele.find("N") == string::npos)
            ) {
                alleles[
                    pair<long, long>(record.start, record.stop)
                ].insert(record.allele);
            }
        }
    }
}

void AlleleSearcherLiteFiltered::assemble_alleles_from_reads(bool reassemble) {
    this->prepReadObjs();

    DEBUG << "Prepared read objects, beginning assembly for region";

    // Create reference object
    Reference ref(this->reference, this->windowStart);

    DEBUG << "Number of differing regions = " << this->differingRegions.size();

    if (this->differingRegions.empty()) return;

    // Find the minimum location and maximum location from
    // differing regions
    long start = this->differingRegions.front().first - this->band_margin;
    long stop = this->differingRegions.back().second + this->band_margin;

    DEBUG << "Assembly regions will be between " << start << " and " << stop;

    vector<Read*> read_objects;

    for (auto& read_obj: this->read_objs) {
        read_obj.extract_alleles(differingRegions);
    }

    DEBUG << "Prepared supporting alleles from every read";

    if ((reassemble) && (this->differingRegions.size() < max_reassembly_region_size)) {
        DEBUG << "Performing reassembly of Pacbio reads with " << differingRegions.size() << " locations";

        map<pair<long, long>, unordered_set<string>> i_alleles;

        for (auto& read_obj: this->read_objs) {
            if (!read_obj.pacbio) read_objects.push_back(&read_obj);
        }

        get_alleles_from_reads(
            i_alleles,
            read_objects,
            this->differingRegions
        );

        // Build these Illumina alleles into a vector of Site records
        vector<SiteRecord> sites;

        for (auto& item: i_alleles) {
            vector<string> allele_strings(
                item.second.begin(),
                item.second.end()
            );
            SiteRecord s(
                allele_strings,
                item.first.first,
                item.first.second
            );
            sites.emplace_back(move(s));
        }

        // Obtain all potential haplotype combinations between start and stop
        // and corresponding Illumina allelic records
        unordered_map<string, vector<AllelicRecord>> result;
        enumerate_all_haplotypes(sites, ref, start, stop, result);

        for (auto& read_obj: this->read_objs) {
            if (read_obj.pacbio) {
                read_obj.update_allelic_records(ref, result, start, stop);
            }
        }

        DEBUG << "Completed pacbio reassembly";
    }

    DEBUG << "Computing supported alleles in differing regions";

    // Recompute alleles from all reads
    read_objects.clear();
    alleles_in_regions.clear();
    for (auto& obj: this->read_objs) {
        read_objects.push_back(&obj);
    }
    get_alleles_from_reads(
        alleles_in_regions,
        read_objects,
        this->differingRegions
    );

    // // Create an allele map in each read
    // for (auto& read_obj: this->read_objs) {
    //     read_obj.create_allele_map();
    // }

    DEBUG << "Final stage of assembly commencing";

    // Find reads which map to each allele
    // in each differing region, and record
    // their support
    for (auto& read_obj: this->read_objs) {
        for (auto& record: read_obj.alleles) {
            if ((read_obj.mapq >= this->minMapQ) && (record.min_q >= this->qThreshold)) {
                supports_in_region[
                    pair<long, long>(record.start, record.stop)
                ][record.allele].insert(read_obj.read_id);
                DEBUG << "For location " << record.start << ", " << record.stop << " found allele " << record.allele << " from read " << read_obj.name;
            }
        }
    }

    // Helper functions for determining partial supports
    auto check_left_partial = [](const string& partial, const string& full) -> bool {
        if (full.size() < partial.size()) return false;
        return (full.substr(full.size() - partial.size(), partial.size()) == partial);
    };

    auto check_right_partial = [](const string& partial, const string& full) -> bool {
        if (full.size() < partial.size()) return false;
        return (full.substr(0, partial.size()) == partial);
    };

    auto get_matching_alleles = [check_left_partial, check_right_partial](
        const string& partial,
        const pair<long, long>& location,
        map<pair<long, long>, unordered_map<string, unordered_set<size_t>>>& support_items,
        bool left
    ) {
        unordered_set<string> match_set;

        if (support_items.find(location) != support_items.end()) {
            const auto& supported_alleles = support_items[location];

            for (auto& full_allele_set: supported_alleles) {
                bool flag;

                if (left) flag = check_left_partial(partial, full_allele_set.first);
                else flag = check_right_partial(partial, full_allele_set.first);

                if (flag) match_set.insert(full_allele_set.first);
            }
        }

        return match_set;
    };

    // Determine partial supports
    for (auto& read_obj: this->read_objs) {
        if (read_obj.has_left_partial) {
            const auto& lallele = read_obj.left_partial;
            pair<long, long> key(lallele.start, lallele.stop);
            const string& partial = lallele.allele;
            auto match_set = get_matching_alleles(partial, key, this->supports_in_region, true);
            if (match_set.size() == 1) {
                supports_in_region[key][*match_set.begin()].insert(read_obj.read_id);
            }
        } else if (read_obj.has_right_partial) {
            const auto& rallele = read_obj.right_partial;
            pair<long, long> key(rallele.start, rallele.stop);
            const string& partial = rallele.allele;
            auto match_set = get_matching_alleles(partial, key, this->supports_in_region, false);
            if (match_set.size() == 1) {
                supports_in_region[key][*match_set.begin()].insert(read_obj.read_id);
            }
        }
    }
}

void AlleleSearcherLiteFiltered::determine_differing_regions_helper(
    const vector<AlleleCounts>& counts,
    set<long>& differingLocations,
    long min_count_snv,
    long min_count_indel
) {
    // set<long> differingLocations;

    DEBUG << "Determining differing regions with snv threshold = " << this->snvThreshold << " , indel threshold = " << this->indelThreshold;

    // Go through allele counts and push each marked position
    for (auto& item: counts) {
        // There may be sites that are completely
        // spanned by homozygous deletions. Skip them.
        if (item.total == 0) {
            continue;
            DEBUG << "Location " << item.pos << " has no reads";
        }

        for (auto& count: item.altCounts) {
            float value = count.second;
            auto& key = count.first;
            auto& refBase = key.first;
            auto& altBase = key.second;

            if ((refBase.size() == 1) && (altBase.size() == 1)) {
                // SNV threshold
                if ((value / item.total >= this->snvThreshold) && (value >= min_count_snv)) {
                    differingLocations.insert(item.pos);
                }
            } else {
                // Only use alleles of small size
                if (max(refBase.size(), altBase.size()) > this->maxAlleleSize) continue;

                DEBUG << "Position = " << item.pos << ", altcount, total = " << value << ", " << item.total;

                if ((value / item.total >= this->indelThreshold) && (value >= min_count_indel)) {
                    DEBUG << "yes";

                    // Add all bases from the left-flanking base to the right
                    // flanking base of the indel
                    long start = item.pos;
                    long stop = start + refBase.size() + 1;

                    for (long i = start; i < stop; i ++) {
                        differingLocations.insert(i);
                        DEBUG << "here";
                    }
                } else {
                    DEBUG << "no";
                }
            }
        }
    }
}

// A utility function to obtain all alleles at a site
vector<string> AlleleSearcherLiteFiltered::determineAllelesAtSite(size_t start_, size_t stop_)
{
    vector<string> allelesInRegion;

    pair<long, long> site(start_, stop_);

    if (alleles_in_regions.find(site) != alleles_in_regions.end()) {
        allelesInRegion.insert(
            allelesInRegion.end(),
            alleles_in_regions[site].begin(),
            alleles_in_regions[site].end()
        );
    }

    return allelesInRegion;
}

void AlleleSearcherLiteFiltered::addAlleleForAssembly(const string& allele)
{
    this->allelesForAssembly.insert(allele);
}

void AlleleSearcherLiteFiltered::clearAllelesForAssembly()
{
    CLEAR(this->allelesForAssembly)
}

/* Determine supporting reads */
void AlleleSearcherLiteFiltered::assemble(size_t start_, size_t stop_)
{
    this->supports.clear();

    pair<long, long> site(start_, stop_);

    if (supports_in_region.find(site) != supports_in_region.end()) {
        const auto& support_map = supports_in_region[site];

        for (auto& item: support_map) {
            vector<size_t> supporting_reads;
            supporting_reads.insert(
                supporting_reads.end(),
                item.second.begin(),
                item.second.end()
            );
            this->supports[item.first] = supporting_reads;
        }
    }

    for (auto& item: this->supports) {
        this->allelesAtSite.push_back(item.first);
    }

    this->assemblyStart = start_;
    this->assemblyStop = stop_;

    DEBUG << "Completed assembly";
}

size_t AlleleSearcherLiteFiltered::numReadsSupportingAllele(const string& allele)
{
    // Note: this call is deprecated
    size_t num = 0;
    return num;
}

size_t AlleleSearcherLiteFiltered::numReadsSupportingAlleleStrict(const string& allele, bool pacbio_)
{
    size_t num = 0;
    if (this->supports.find(allele) != this->supports.end()) {
        for (auto& item: this->supports[allele]) {
            size_t readId = item; // .first;
            if (!(pacbio_ ^ this->pacbio[readId])) num ++;
        }
    }
    return num;
}

// Modified from DeepVariant - for read and reference tracks
int AlleleSearcherLiteFiltered::BaseColor(char base)
{
    switch (base)
    {
        case 'A': return (this->base_color_offset_a_and_g + 3 * this->base_color_stride);
        case 'G': return (this->base_color_offset_a_and_g + 2 * this->base_color_stride);
        case 'T': return (this->base_color_offset_t_and_c + 1 * this->base_color_stride);
        case 'C': return (this->base_color_offset_t_and_c + 0 * this->base_color_stride);
        default: return 0;
            // Note: since the read cannot have 'N's in their aligned segments,
            // we do not need a special color for 'N'. So default colour is simply
            // the color for gaps
    }
}

// From DeepVariant - for read track
int AlleleSearcherLiteFiltered::BaseQualityColor(int qual)
{
    float capped = min(qual, this->base_quality_cap);
    return int(254 * (1.0 * capped / this->base_quality_cap));
}

// From DeepVariant - for mapping quality track
int AlleleSearcherLiteFiltered::MappingQualityColor(int qual)
{
    float capped = min(qual, this->mapping_quality_cap);
    return int(254 * (1.0 * capped / this->mapping_quality_cap));
}

// From DeepVariant - for orientation track
int AlleleSearcherLiteFiltered::StrandColor(int value)
{
    return ((value > 0) ? this->positive_strand_color : this->negative_strand_color);
}

// For allele position marker track
int AlleleSearcherLiteFiltered::PositionColor(size_t position)
{
    if ((this->assemblyStart - this->windowStart <= position) && (position < this->assemblyStop - this->windowStart))
    {
        return this->allele_position_color;
    }

    return this->background_position_color;
}

// Computes colored feature-maps with indels colored in at a single position
// This follows the DeepVariant method
np::ndarray AlleleSearcherLiteFiltered::computeFeaturesColoredSimple(const string& allele, size_t featureLength, bool pacbio_) {
    size_t numSupports = this->numReadsSupportingAlleleStrict(allele, pacbio_);
    size_t numChannels = 6;

    if (numSupports == 0) {
        // Return dummy array if there is no support
        p::tuple shape = p::make_tuple(1, featureLength, numChannels);
        np::dtype dtype = np::dtype::get_builtin<uint8_t>();
        np::ndarray array = np::zeros(shape, dtype);
        return array;
    }

    Array3D<uint8_t> array(numSupports, featureLength, numChannels);

    // (assemblyStart + assemblyStop) / 2 is the middle of the feature map
    long midReferencePoint = (this->assemblyStart + this->assemblyStop) / 2;
    long startReferencePoint = midReferencePoint - (long(featureLength) / 2);
    long endReferencePoint = startReferencePoint + featureLength;
    const auto& supportItems = this->supports[allele];
    size_t readCounter = 0;

    auto between = [] (long x, long y, long z) { return ((x <= y) && (y < z)); };

    for (auto& item: supportItems) {
        // size_t readId = item.first;
        size_t readId = item;

        if (pacbio_ ^ this->pacbio[readId]) continue;

        const auto& read = this->reads[readId];
        const auto& qual = this->qualities[readId];
        const auto& cigartuple = this->cigartuples[readId];
        long rfcounter = this->referenceStarts[readId];
        long rdcounter = 0;
        int mapQColor = this->MappingQualityColor(this->mapq[readId]);
        int strandColor = this->StrandColor(this->orientation[readId]);

        for (auto& cigar: cigartuple) {
            long operation = cigar.first;
            long length = cigar.second;

            switch(operation) {
                case BAM_CEQUAL:
                case BAM_CDIFF:
                case BAM_CMATCH: {
                    for (long j = 0; j < length; j++) {
                        if (between(startReferencePoint, rfcounter + j, endReferencePoint)) {
                            long fmapindex = rfcounter + j - startReferencePoint;
                            int refColor = this->BaseColor(this->reference[rfcounter + j - this->windowStart]);
                            int readColor = this->BaseColor(read[rdcounter + j]);
                            int qualColor = this->BaseQualityColor(qual[rdcounter + j]);
                            int positionColor = this->PositionColor(rfcounter + j - this->windowStart);
                            accessArray(array, readCounter, fmapindex, READ_BASE_TRACK) = readColor;
                            accessArray(array, readCounter, fmapindex, REF_BASE_TRACK) = refColor;
                            accessArray(array, readCounter, fmapindex, READ_QUAL_TRACK) = qualColor;
                            accessArray(array, readCounter, fmapindex, READ_MAPQ_TRACK) = mapQColor;
                            accessArray(array, readCounter, fmapindex, READ_ORIENTATION_TRACK) = strandColor;
                            accessArray(array, readCounter, fmapindex, POSITION_MARKER_TRACK) = positionColor;
                        }
                    }
                    rfcounter += length;
                    rdcounter += length;
                    break;
                }

                case BAM_CDEL: {
                    if (between(startReferencePoint, rfcounter - 1, endReferencePoint)) {
                        // Fill reference bases and position marker at all positions in the deletion
                        for (long i = rfcounter - 1; i < rfcounter + length; i++) {
                            if (!between(startReferencePoint, i, endReferencePoint)) continue;
                            long fmapindex = i - startReferencePoint;
                            int refColor = this->BaseColor(this->reference[i - this->windowStart]);
                            int positionColor = this->PositionColor(i - this->windowStart);
                            accessArray(array, readCounter, fmapindex, REF_BASE_TRACK) = refColor;
                            accessArray(array, readCounter, fmapindex, READ_MAPQ_TRACK) = mapQColor;
                            accessArray(array, readCounter, fmapindex, READ_ORIENTATION_TRACK) = strandColor;
                            accessArray(array, readCounter, fmapindex, POSITION_MARKER_TRACK) = positionColor;
                        }

                        // Fill read base color at the first position
                        char readBase = '*';
                        int rdcolor = this->BaseColor(readBase);
                        long fmapindex = rfcounter - 1 - startReferencePoint;
                        int qualColor = rdcounter > 0 ? this->BaseQualityColor(qual[rdcounter - 1]) : 0;
                        accessArray(array, readCounter, fmapindex, READ_BASE_TRACK) = rdcolor;
                        accessArray(array, readCounter, fmapindex, READ_QUAL_TRACK) = qualColor;
                    }
                }
                case BAM_CREF_SKIP: {
                    rfcounter += length;
                    break;
                }

                case BAM_CINS: {
                    if (between(startReferencePoint, rfcounter - 1, endReferencePoint)) {
                        char readBase = '*';
                        int readColor = this->BaseColor(readBase);
                        int refColor = this->BaseColor(this->reference[rfcounter - 1 - this->windowStart]);
                        int qualColor;
                        if (rdcounter > 0) {
                            qualColor = this->BaseQualityColor(
                                *min_element(qual.begin() + rdcounter - 1, qual.begin() + rdcounter + length)
                            );
                        } else {
                            qualColor = this->BaseQualityColor(
                                *min_element(qual.begin() + rdcounter, qual.begin() + rdcounter + length)
                            );
                        }
                        int positionColor = this->PositionColor(rfcounter - 1 - this->windowStart);
                        long fmapindex = rfcounter - 1 - startReferencePoint;
                        accessArray(array, readCounter, fmapindex, READ_BASE_TRACK) = readColor;
                        accessArray(array, readCounter, fmapindex, REF_BASE_TRACK) = refColor;
                        accessArray(array, readCounter, fmapindex, READ_QUAL_TRACK) = qualColor;
                        accessArray(array, readCounter, fmapindex, READ_MAPQ_TRACK) = mapQColor;
                        accessArray(array, readCounter, fmapindex, READ_ORIENTATION_TRACK) = strandColor;
                        accessArray(array, readCounter, fmapindex, POSITION_MARKER_TRACK) = positionColor;
                    }
                }
                case BAM_CSOFT_CLIP: {
                    rdcounter += length;
                    break;
                }
            }
        }

        readCounter += 1;
    }

    // Copy over to numpy array
    p::tuple shape = p::make_tuple(numSupports, featureLength, numChannels);
    np::dtype dtype = np::dtype::get_builtin<uint8_t>();
    np::ndarray featureMapNp = np::zeros(shape, dtype);

    // Copy over into numpy array
    std::copy(array.begin(), array.end(), reinterpret_cast<uint8_t*>(featureMapNp.get_data()));

    return featureMapNp;
}

AlleleSearcherLiteFiltered::~AlleleSearcherLiteFiltered()
{ }
