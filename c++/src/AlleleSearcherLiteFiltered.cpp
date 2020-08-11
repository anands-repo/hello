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

PositionLite::PositionLite(const string& ref, const size_t numTracks, const size_t pos)
: ref(ref), pos(pos), score(0), coverage(0), marked(false)
{
    for (size_t j = 0; j < numTracks; j++)
    {
        string alt_;
        vector<size_t> qual_;
        this->alt.push_back(alt_);
        this->qual.push_back(qual_);
    }
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

// Add a single base and single quality to a specific track
void PositionLite::addTrackElement(const size_t trackId, const string& base, const size_t qual, long readPos)
{
    auto& altTrack  = this->alt[trackId];
    auto& qualTrack = this->qual[trackId];
    altTrack = altTrack + base;
    qualTrack.push_back(qual);
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

        if (this->mapq[i] < this->minMapQ) {
            continue;
        }

        assert(rfcounter > 0);

        // Quality control and allele addition function
        auto addToCount = [] (
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
                        auto& count = this->counts[rfcounter + j];

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
                    auto& count = this->counts[rfcounter - 1];
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
                }
                case BAM_CREF_SKIP:
                {
                    rfcounter += length;
                    break;
                }

                case BAM_CINS:
                {
                    auto& count = this->counts[rfcounter - 1];
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

// Determine aligned pairs for all reads
void AlleleSearcherLiteFiltered::getAlignedPairs()
{
    size_t i = 0;

    for (auto& cigartuple : this->cigartuples)
    {
        size_t rdcounter = 0;
        size_t rfcounter = this->referenceStarts[i];
        AlignedPair alignedPair;

        for (auto& cigar : cigartuple)
        {
            size_t operation = cigar.first;
            size_t length    = cigar.second;

            switch(operation)
            {
                case BAM_CSOFT_CLIP:
                {
                    rdcounter += length;
                    break;
                }
                case BAM_CEQUAL:
                case BAM_CDIFF:
                case BAM_CMATCH:
                {
                    for (size_t j = 0; j < length; j++)
                    {
                        pair<long,long> entry(rdcounter+j,rfcounter+j);
                        alignedPair.emplace_back(move(entry));
                    }
                    rdcounter += length;
                    rfcounter += length;
                    break;
                }
                case BAM_CDEL:
                case BAM_CREF_SKIP:
                {
                    for (size_t j = 0; j < length; j++)
                    {
                        pair<long,long> entry(-1,rfcounter+j);
                        alignedPair.emplace_back(move(entry));
                    }
                    rfcounter += length;
                    break;
                }
                case BAM_CINS:
                {
                    for (size_t j = 0; j < length; j++)
                    {
                        pair<long,long> entry(rdcounter+j,-1);
                        alignedPair.emplace_back(move(entry));
                    }
                    rdcounter += length;
                    break;
                }
                default: break;
            } //switch
        } // for

        this->alignedPairs.emplace_back(move(alignedPair));
        i++;
    }
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
    size_t qThreshold,
    bool leftAlign
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
    qThreshold(qThreshold),
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
    maxAlleleSize(100)
{
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
        }
    }

    // Initialize allele count object
    for (size_t i = 0; i < this->reference.size(); i++)
    {
        AlleleCounts count;
        count.refCount = 0;
        count.pos = i + this->windowStart;
        count.total = 0;
        this->counts.emplace_back(move(count));
    }

    // Update all allele counts
    this->updateAlleleCounts();

    // Resolve partial counts
    for (auto& count : this->counts) {
        count.resolvePartials();
    }
}

void AlleleSearcherLiteFiltered::prepMatrix() {
    if (!this->matrix.empty()) {
        return;
    }

    // Create aligned pairs
    this->getAlignedPairs();

    // Create a Position item for each reference position
    for (size_t i = 0; i < this->reference.size(); i++)
    {
        string refItem(1, this->reference[i]);
        PositionLite p(refItem, this->reads.size(), windowStart + i);
        this->matrix.emplace_back(move(p));
    }

    // Add reads to Position matrix
    for (size_t index = 0; index < this->reads.size(); index++)
    {
        this->addRead(
            index,
            this->referenceStarts[index],
            this->alignedPairs[index]
        );
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
        Read readobj(
            read,
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

// Add a read to the Position matrix
void AlleleSearcherLiteFiltered::addRead(
    const size_t index,
    const size_t referenceStart,
    const AlignedPair& alignedPairs
)
{
    const string& read = this->reads[index];
    const vector<size_t>& quality = this->qualities[index];
    size_t refCounter = referenceStart - this->windowStart;

    for (auto& alignedPair : alignedPairs)
    {
        if ((alignedPair.first >= 0) || (alignedPair.second >= 0))
        {
            auto& position = this->matrix[refCounter];
            string base;
            size_t qual;

            if (alignedPair.first >= 0)
            {
                // Insertion or Match
                base = read[alignedPair.first];
                qual = quality[alignedPair.first];
            }
            else
            {
                // Deletion (refCounter increments)
                base = '-';
                qual = 30;
            }

            position.addTrackElement(index, base, qual, alignedPair.first);
        }

        refCounter = alignedPair.second >= 0 ? refCounter + 1 : refCounter;

        assert(refCounter < this->matrix.size());
    }
}

// Does allele in trackId between start (inclusive) and stop (exclusive) meet quality requirements?
bool AlleleSearcherLiteFiltered::doesAlleleMeetQuality(size_t trackId, size_t start, size_t stop)
{
    assert(stop >= start);

    if (stop == start) return false;

    for (size_t i = start; i < stop; i++)
    {
        for (auto& q : this->matrix[i].qual[trackId])
        {
            if (q < this->qThreshold)
            {
                return false;
            }
        }
    }

    return true;
}

// Get bases in a track between start and stop
// Note: this function supports right-to-left indexing as well, for legacy reasons
string AlleleSearcherLiteFiltered::getBasesInTrack(const size_t trackId, const int start, const int stop)
{
    bool rightWards = (start >= 0) || ((start < 0) && (stop <= 0));
    string basesInTrack;

    if ((start < 0) && (stop <= 0))
        return this->reads[trackId]; // All bases in a specific track

    int startPoint = rightWards ? start : stop - 1;
    int increment  = rightWards ? 1 : -1;

    for (int i = startPoint; 
        rightWards ? (stop > 0 ? i < stop : i < this->matrix.size()) : i >= max_(start,0);
        i += increment 
    )
    {
        const string& bases = this->matrix[i].alt[trackId];

        if (bases.empty()) break;

        if (basesInTrack.empty())
        {
            basesInTrack = bases;
        }
        else
        {
            basesInTrack = rightWards ? basesInTrack + bases : bases + basesInTrack;
        }
    }

    boost::erase_all(basesInTrack, "-");

    return basesInTrack;
}

// Get a set of quality values from start (inclusive) to stop (exclusive). Unlike
// getBasesInTrack, this expects stop > start
vector<size_t> AlleleSearcherLiteFiltered::getQualitiesInTrack(size_t trackId, int start, int stop)
{
    vector<size_t> qualities;

    for (size_t i = start; i < stop; i++)
    {
        const string& bases = this->matrix[i].alt[trackId];
        const vector<size_t>& quals = this->matrix[i].qual[trackId];

        if (bases.empty()) break;

        if (bases.size() == 1)
        {
            if (bases != "-")
            {
                qualities.insert(qualities.end(), quals.begin(), quals.end());
            }
        }
        else
        {
            qualities.insert(qualities.end(), quals.begin(), quals.end());
        }
    }

    return qualities;
}

// Is a read track empty between start and stop
bool AlleleSearcherLiteFiltered::isTrackEmpty(const size_t trackId, const size_t start, const size_t stop)
{
    if (stop > start)
    {
        for (size_t i = start; i < stop; i++)
        {
            if (this->matrix[i].alt[trackId].empty()) return true;
        }

        return false;
    }

    return true;
}

// Is there an indel in a cluster of positions?
bool AlleleSearcherLiteFiltered::indelInCluster(const vector<size_t>& cluster)
{
    for (auto& l_ : cluster)
    {
        auto l = l_ - this->windowStart;

        if (this->matrix[l].vtypes.find("I") != this->matrix[l].vtypes.end())
        {
            return true;
        }

        if (this->matrix[l].vtypes.find("D") != this->matrix[l].vtypes.end())
        {
            return true;
        }
    }

    return false;
}

// Is there an empty allele in cluster between start and stop
// Empty allele refers to deletion, not whether the track is empty
bool AlleleSearcherLiteFiltered::isEmptyAlleleInCluster(size_t first, size_t last)
{

    for (size_t i = 0; i < this->reads.size(); i++)
    {
        if (!this->isTrackEmpty(i, first, last))
        {
            string allele = this->getBasesInTrack(i, first, last);
            if (allele.size() == 0)
            {
                return true;
            }
        }
    }

    return false;
}

// Push a cluster of alleles into regions, but make sure there are no empty alleles at the start or stop
// position of the cluster. If there is, shift the cluster end-points
void AlleleSearcherLiteFiltered::pushRegions(vector<size_t>& cluster, vector<pair<size_t,size_t> >& regions)
{
    // if (this->indelInCluster(cluster))
    {
        pair<size_t,size_t> region;
        // region.first  = this->windowStart + cluster.front();
        // region.second = this->windowStart + cluster.back() + 1;
        region.first  = cluster.front();
        region.second = cluster.back() + 1;
        bool flag = true;

        regions.push_back(region);
    }

    cluster.clear();
}

void AlleleSearcherLiteFiltered::determineDifferingRegions()
{
    CLEAR(this->differingRegions)

    set<long> differingLocations;

    DEBUG << "Determining differing regions with snv threshold = " << this->snvThreshold << " , indel threshold = " << this->indelThreshold;

    // Go through allele counts and push each marked position
    for (auto& item: this->counts) {
        // There may be sites that are completely
        // spanned by homozygous deletions. Skip them.
        if (item.total == 0) {
            continue;
        }

        for (auto& count: item.altCounts) {
            float value = count.second;
            auto& key = count.first;
            auto& refBase = key.first;
            auto& altBase = key.second;

            if ((refBase.size() == 1) && (altBase.size() == 1)) {
                // SNV threshold
                if ((value / item.total >= this->snvThreshold) && (value >= this->minCount)) {
                    differingLocations.insert(item.pos);
                }
            } else {
                // Only use alleles of small size
                if (max(refBase.size(), altBase.size()) > this->maxAlleleSize) continue;

                if ((value / item.total >= this->indelThreshold) && (value >= this->minCount)) {
                    // Add all bases from the left-flanking base to the right
                    // flanking base of the indel
                    long start = item.pos;
                    long stop = start + refBase.size() + 1;

                    for (long i = start; i < stop; i ++) {
                        differingLocations.insert(i);
                    }
                }
            }
        }
    }

    // Collect consecutive locations that are affected into a cluster
    vector<size_t> cluster;

    for (auto& location : differingLocations) {
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
                this->pushRegions(cluster, this->differingRegions);
                cluster.push_back(location);
            }
        }
    }

    if (!cluster.empty()) {
        this->pushRegions(cluster, this->differingRegions);
    }
}

// A utility function to obtain all alleles at a site
vector<string> AlleleSearcherLiteFiltered::determineAllelesAtSite(size_t start_, size_t stop_)
{
    this->prepReadObjs();

    unordered_set<string> alleleSet;

    for (const auto& read_obj: this->read_objs) {
        auto result = read_obj.get_aligned_bases(start_, stop_);
        if (result.second != AlignedBaseStatus::Success) continue;
        if (result.third < this->qThreshold) continue;
        if (result.first.find("N") != string::npos) continue;
        if (read_obj.mapq < this->minMapQ) continue;
        alleleSet.insert(result.first);
    }

    vector<string> allelesInRegion(alleleSet.begin(), alleleSet.end());

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
void AlleleSearcherLiteFiltered::assemble(size_t start_, size_t stop_, bool noFilter)
{
    this->prepReadObjs();

    DEBUG << "Assembling between " << start_ << " (inclusive) and " << stop_;

    CLEAR(this->allelesAtSite)
    CLEAR(this->refAllele)
    CLEAR(this->supports)

    unordered_map<string, vector<size_t>> partial_supports_left;
    unordered_map<string, vector<size_t>> partial_supports_right;

    this->refAllele = this->reference.substr(start_ - this->windowStart, stop_ - start_);

    if (this->allelesForAssembly.empty()) {
        DEBUG << "Alleles for assembly not provided, determining alleles for assembly";
        auto alleles = this->determineAllelesAtSite(start_, stop_);
        this->allelesAtSite.insert(
            this->allelesAtSite.end(), alleles.begin(), alleles.end()
        );
    } else {
        DEBUG << "Using the provided alleles for assembly";
        this->allelesAtSite.insert(
            this->allelesAtSite.end(), this->allelesForAssembly.begin(), this->allelesForAssembly.end()
        );
    }

    unordered_set<string> allelesForAssembly(this->allelesAtSite.begin(), this->allelesAtSite.end());

    // Bucketize reads
    for (auto& read_obj: this->read_objs) {
        if (read_obj.mapq < this->minMapQ) continue;

        auto result = read_obj.get_aligned_bases(start_, stop_);

        switch(result.second) {
            case AlignedBaseStatus::Success: {
                if (allelesForAssembly.find(result.first) != allelesForAssembly.end()) {
                    this->supports[result.first].push_back(read_obj.read_id);
                }
                break;
            }
            case AlignedBaseStatus::LeftPartial: {
                partial_supports_left[result.first].push_back(read_obj.read_id);
                break;
            }
            case AlignedBaseStatus::RightPartial: {
                partial_supports_right[result.first].push_back(read_obj.read_id);
                break;
            }
            default: continue;
        }
    }

    // Resolve partials
    auto get_partial_to_full_allele_maps = [](
        const unordered_map<string, vector<size_t>>& partial_map,
        const unordered_map<string, vector<size_t>>& full_map,
        unordered_map<string, string>& resolution_map,
        bool left
    ) {
        unordered_map<string, unordered_set<string>> resolution_map_preliminary;

        // Get all full alleles to which a partial allele is a (partial) match
        for (auto& partial_item: partial_map) {
            const string& partial_allele = partial_item.first;

            for (auto& full_support_item: full_map) {
                const string& full_allele = full_support_item.first;

                if (full_allele.size() < partial_allele.size()) continue;

                string substr = left ? full_allele.substr(
                    full_allele.size() - partial_allele.size(),
                    partial_allele.size()
                ) : full_allele.substr(0, partial_allele.size());

                if (substr == partial_allele) {
                    resolution_map_preliminary[partial_allele].insert(full_allele);
                }
            }
        }

        // Only maintain those partial alleles which have a single match
        for (auto& preliminary_resolution: resolution_map_preliminary) {
            if (preliminary_resolution.second.size() == 1) {
                resolution_map[preliminary_resolution.first] = *preliminary_resolution.second.begin();
            }
        }
    };

    auto push_partial_to_full = [](
        unordered_map<string, string>& partial_to_full_resolution,
        unordered_map<string, vector<size_t>>& full_map,
        unordered_map<string, vector<size_t>>& partial_map
    ) {
        for (auto& partial_map_item: partial_to_full_resolution) {
            const string& partial_allele = partial_map_item.first;
            const string& full_allele = partial_map_item.second;
            full_map[full_allele].insert(
                full_map[full_allele].end(),
                partial_map[partial_allele].begin(),
                partial_map[partial_allele].end()
            );
        }
    };

    // Resolve left partials
    unordered_map<string, string> partial_to_full_allele_maps;
    get_partial_to_full_allele_maps(partial_supports_left, this->supports, partial_to_full_allele_maps, true);
    push_partial_to_full(partial_to_full_allele_maps, this->supports, partial_supports_left);

    // Resolve right partials
    partial_to_full_allele_maps.clear();
    get_partial_to_full_allele_maps(partial_supports_right, this->supports, partial_to_full_allele_maps, false);
    push_partial_to_full(partial_to_full_allele_maps, this->supports, partial_supports_right);

    for (auto& item: supports) {
        for (auto& read: item.second) {
            DEBUG << "Read " << read << " supports allele " << item.first;
        }
    }

    this->assemblyStart = start_;
    this->assemblyStop = stop_;

    DEBUG << "Completed assembly";
}

// Get bases from position till the end of a read's alignment
string AlleleSearcherLiteFiltered::getLeftBases(size_t trackId, size_t pos)
{
    string bases;

    for (long i = pos; i >= 0; i--)
    {
        const auto& position = this->matrix[i];

        if (position.alt[trackId].empty())
            break;

        const string& altString = position.alt[trackId];

        if (altString != "-")
        {
            bases = altString + bases;  // Note: append bases to the left
        }
    }

    return bases;
}

// Get bases to the right of a position
string AlleleSearcherLiteFiltered::getRightBases(size_t trackId, size_t pos)
{
    string bases;

    for (long i = pos; i < this->matrix.size(); i++)
    {
        const auto& position = this->matrix[i];

        if (position.alt[trackId].empty()) break;

        const string& altString = position.alt[trackId];

        if (altString != "-") bases += altString;
    }

    return bases;
}

// Count the number of bases to the left of a given position in a given track
size_t AlleleSearcherLiteFiltered::countLeftBases(size_t trackId, size_t pos)
{
    size_t numBases = 0;

    for (long i = pos; i >= 0; i--)
    {
        const auto& position = this->matrix[i];

        if (position.alt[trackId].empty())
            break;

        const string& altString = position.alt[trackId];

        if (altString != "-")
            numBases += altString.size();
    }

    return numBases;
}

size_t AlleleSearcherLiteFiltered::countRightBases(size_t trackId, size_t pos)
{
    size_t numBases = 0;

    for (long i = pos; i < this->matrix.size(); i++)
    {
        const auto& position = this->matrix[i];

        if (position.alt[trackId].empty()) break;

        const string& altString = position.alt[trackId];

        if (altString != "-") numBases += altString.size();
    }

    return numBases;
}

size_t AlleleSearcherLiteFiltered::numReadsSupportingAllele(const string& allele)
{
    size_t num = 0;
    if (this->supports.find(allele) != this->supports.end()) num += this->supports.find(allele)->second.size();
    // if (this->nonUniqueSupports.find(allele) != this->nonUniqueSupports.end()) num += this->nonUniqueSupports.find(allele)->second.size();
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

size_t AlleleSearcherLiteFiltered::coverage(size_t position_)
{
    size_t position = position_ - this->windowStart;
    return this->matrix[position].coverage;
}

AlleleSearcherLiteFiltered::~AlleleSearcherLiteFiltered()
{ }
