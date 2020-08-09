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

    /// TEST THIS!!! Meaning, I have commented this, and we need to test
    /// whether commenting this has an effect or not
    /// Because intervaltrees are used in python, the result of the above code
    /// absorbs the results of the code below
    // // else
    // // {
    //     for (auto& l : cluster)
    //     {
    //         pair<size_t,size_t> region(l, l + 1);
    //         regions.push_back(region);
    //     }
    // // }
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

// Function for scoring each location as to its activity
void AlleleSearcherLiteFiltered::scoreLocations()
{
    size_t positionStart = 0;
    size_t positionEnd   = this->matrix.size();

    for (size_t positionCounter = positionStart;
        positionCounter < positionEnd;
        positionCounter++
    )
    {
        auto& position = this->matrix[positionCounter];
        for (size_t i = 0; i < position.alt.size(); i++)
        {
            if (position.alt[i].empty()) continue;
            string readTrack(position.alt[i]);
            const vector<size_t>& qualityTrack = position.qual[i];
            const string& ref = position.ref;
            boost::erase_all(readTrack, "-");

            if (readTrack.size() <= ref.size())
            {
                if (readTrack != ref)
                {
                    // Mismatch
                    if ((readTrack.size() == ref.size()) && (readTrack.size() == 1))
                    {
                        if (qualityTrack.front() >= this->qThreshold)
                        {
                            position.score += this->mismatchScore;
                            string vtype("M");
                            position.vtypes.insert(vtype);
                            // DEBUG << "Found mismatch at position " << positionCounter + this->windowStart << " reference = " << position.ref << ", alt = " << position.alt[i] << " read name = " << this->names[i] << " reference start = " << this->referenceStarts[i] << " cigars = " << printCigar(this->cigartuples[i]);
                        }
                    }
                    // Deletion or MNP
                    else
                    {
                        position.score += (readTrack.size() != ref.size()) ? this->deleteScore : this->mismatchScore;
                        string vtype(readTrack.size() != ref.size() ? "D" : "M");
                        position.vtypes.insert(vtype);
                        // DEBUG << "Found deletion at position " << positionCounter + this->windowStart << " reference = " << position.ref << ", alt = " << position.alt[i] << " read name = " << this->names[i] << " reference start = " << this->referenceStarts[i] << " cigars = " << printCigar(this->cigartuples[i]);
                    }
                }
            }
            // Insertion
            else
            {
                string vtype("I");

                if (positionCounter >= 1)
                {
                    this->matrix[positionCounter-1].score += this->insertScore;
                    this->matrix[positionCounter-1].vtypes.insert(vtype);
                }
                position.score += this->insertScore;
                position.vtypes.insert(vtype);
                // DEBUG << "Found insertion at position " << positionCounter + this->windowStart << " reference = " << position.ref << ", alt = " << position.alt[i] << " read name = " << this->names[i] << " reference start = " << this->referenceStarts[i] << " cigars = " << printCigar(this->cigartuples[i]);
            }

            position.coverage++;
        }
    }
}

// A utility function to obtain all alleles at a site
vector<string> AlleleSearcherLiteFiltered::determineAllelesAtSite(size_t start_, size_t stop_)
{
    this->prepMatrix();

    vector<string> allelesInRegion;
    unordered_set<string> allelesInRegion_;
    size_t start = start_ - this->windowStart;
    size_t stop = stop_ - this->windowStart;

    DEBUG << "stop = " << stop << ", start = " << start;

    for (size_t i = 0; i < this->reads.size(); i++)
    {
        // When collecting alleles at a site, ensure that they are collected from reads
        // that do not have a deletion at start/stop
        if (
            (!this->isTrackEmpty(i, start, stop)) &&
            (this->getBasesInTrack(i, start, start + 1).size() != 0) &&
            (this->getBasesInTrack(i, stop - 1, stop).size() != 0) && 
            (this->getBasesInTrack(i, start, start + 1).size() <= 1)
        ) {
            string allele = this->getBasesInTrack(i, start, stop);
            allelesInRegion_.insert(allele);
        }
    }

    for (auto& item : allelesInRegion_)
    {
        allelesInRegion.push_back(item);
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
void AlleleSearcherLiteFiltered::assemble(size_t start_, size_t stop_, bool noFilter)
{
    this->prepMatrix();

    DEBUG << "Assembling between " << start_ << " (inclusive) and " << stop_;

    this->assemblyStart = start_;
    this->assemblyStop = stop_;

    size_t start = start_ - this->windowStart;
    size_t stop = stop_ - this->windowStart;

    CLEAR(this->allelesAtSite)
    CLEAR(this->refAllele)
    CLEAR(this->numItemsPerMatrixPosition)
    CLEAR(this->ndarrayPositionToMatrixPosition)
    CLEAR(this->matrixPositionToNdarrayStartPosition)
    CLEAR(this->supports)
    CLEAR(this->nonUniqueSupports)

    if ((start_ < this->windowStart) || (stop > this->matrix.size()))
    {
        return;
    }

    // Determine reference allele
    for (size_t i = start; i < stop; i++)
    {
        this->refAllele += this->matrix[i].ref;
    }

    DEBUG << "Found ref allele " << this->refAllele ;

    auto notFlankedByDeletion = [this] (long i, long start, long stop) -> bool {
        return (
            (this->getBasesInTrack(i, start, start + 1).size() != 0) &&
            (this->getBasesInTrack(i, stop - 1, stop).size() != 0)
        );
    };

    auto notFlankedByInsertion = [this] (long i, long start, long stop) -> bool {
        /*
        Based on alignedPair and addRead, insertions are added NOT to the "launching" position
        in the reference, but to the "landing" position on the reference; unlike the VCF standard.
        During feature-map construction, it is reversed, though, and follows the VCF standard.
        Hence, we only need to check whether an insertion is there at the first reference position. Any
        insertion at the last base position is technically between start and stop.
        */
       return (this->getBasesInTrack(i, start, start + 1).size() <= 1);
    };

    // Create support containers for each allele
    // either by looking at each read OR ...
    if (this->allelesForAssembly.empty())
    {
        for (size_t i = 0; i < this->reads.size(); i++)
        {
            if ((!this->isTrackEmpty(i, start, stop)) && (noFilter | this->doesAlleleMeetQuality(i, start, stop))) 
            {
                // Ensure that read doesn't have a deletion at start and stop positions
                if (
                    // (this->getBasesInTrack(i, start, start + 1).size() != 0) &&
                    // (this->getBasesInTrack(i, stop - 1, stop).size() != 0)
                    notFlankedByDeletion(i, start, stop) && notFlankedByInsertion(i, start, stop)
                ) {
                    string allele = this->getBasesInTrack(i, start, stop);

                    if (this->supports.find(allele) == this->supports.end())
                    {
                        Mapping emptyMapping;
                        this->supports[allele] = emptyMapping;
                        this->allelesAtSite.push_back(allele);
                    }
                }
            }
        }
    }
    // ... using a predetermined set of alleles
    else
    {
        for (auto& allele : this->allelesForAssembly)
        {
            Mapping emptyMapping;
            this->supports[allele] = emptyMapping;
            this->allelesAtSite.push_back(allele);
        }
    }

    // Remove alleles with Ns in them, as well as alleles that are too long
    //
    // Note: Just by controlling determineDifferingRegions function, we have avoided long deletions
    // Now, we need to avoid long insertions; that can only be accomplished through filtering
    // individual alleles based on their length
    //
    // Note: We do not have to modify supports for this
    // because features are composed only for alleles found in
    // allelesAtSite
    vector<string> goodAlleles;

    for (auto& allele : this->allelesAtSite) {
        if ((allele.find("N") == std::string::npos) && (allele.size() <= this->maxAlleleSize)) {
            goodAlleles.push_back(allele);
        }
    }

    this->allelesAtSite = goodAlleles;

    auto minBaseQual = [this, noFilter] (size_t readId, int position, int length) -> size_t {
        if (noFilter) return this->qThreshold + 10;  // If no filtering is desired, don't allow it
        const auto& quality = this->qualities[readId];
        if (position + length > quality.size()) {
            return *min_element(quality.begin() + position, quality.end());
        } else {
            return *min_element(quality.begin() + position, quality.begin() + position + length);
        }
    };

    // Add reads supporting the assembly of each allele
    for (size_t i = 0; i < this->reads.size(); i++)
    {
        // Check read mapping quality
        if (this->mapq[i] < this->minMapQ) {
            continue;
        }

        // If the track is non-empty add the read to an appropriate allele
        if (
            (!this->isTrackEmpty(i, start, stop)) // &&
            // (this->getBasesInTrack(i, start, start + 1).size() != 0) &&
            // ((this->getBasesInTrack(i, stop - 1, stop).size() != 0)
        ) {
            if (
                notFlankedByDeletion(i, start, stop) & notFlankedByInsertion(i, start, stop)
            ) {
                string alleleInRead = this->getBasesInTrack(i, start, stop);
                if (this->supports.find(alleleInRead) == this->supports.end()) continue;
                size_t numAlignedBasesLeft = this->countLeftBases(i, start);
                size_t numLeftSoftClippedBases = (this->cigartuples[i].front().first == BAM_CSOFT_CLIP) ? this->cigartuples[i].front().second : 0;
                long startPositionInRead = numAlignedBasesLeft + numLeftSoftClippedBases - 1;
                long stopPositionInRead = startPositionInRead + stop - start;

                // Quality score cutoff for allelic bases
                if (minBaseQual(i, startPositionInRead, alleleInRead.size()) >= this->qThreshold) {
                    this->supports[alleleInRead][i] = pair<long, long>(startPositionInRead, stopPositionInRead);
                    DEBUG << "Adding read " << this->names[i] << " in support of allele " << alleleInRead;
                } else {
                    DEBUG << "Read " << this->names[i] << " has low quality bases at variant site, discarding";
                }
            }
        }
        else
        {
            // In cases where a read only covers the starting or the ending position
            // of an allele (but not both), we extract the overlapping bases and check
            // the bases for partial match to any allele
            bool leftIsNotEmpty = !this->isTrackEmpty(i, start, start + 1);
            bool rightIsNotEmpty = !this->isTrackEmpty(i, stop - 1, stop);

            // If the non-empty side ends in a deletion, discontinue
            // Note that for the left-most position, we do not want either insertion or deletion
            // Only way we do not have an insertion at the left position is to have a match or substitution
            // both resulting in alt bases with length of 1
            if (leftIsNotEmpty && (this->getBasesInTrack(i, start, start + 1).size() != 1)) continue;
            if (rightIsNotEmpty && (this->getBasesInTrack(i, stop - 1, stop).size() == 0)) continue;

            if ((leftIsNotEmpty || rightIsNotEmpty))
            {
                DEBUG << "Checking read " << this->names[i] << " for partial support of alleles";

                string alleleInRead;

                if (leftIsNotEmpty)
                {
                    alleleInRead = this->getRightBases(i, start);
                    DEBUG << "Found left allele " << alleleInRead << " in read " << this->names[i];
                }
                else
                {
                    alleleInRead = this->getLeftBases(i, stop - 1);
                    DEBUG << "Found right allele " << alleleInRead << " in read " << this->names[i];
                }

                // Run a check for partial matches
                unordered_set<string> supportedAlleles;

                for (auto& supportItems : this->supports)
                {
                    const string& allele = supportItems.first;

                    if (allele.size() < alleleInRead.size()) continue;

                    string substringToCompare;

                    if (leftIsNotEmpty)
                    {
                        substringToCompare = allele.substr(0, alleleInRead.size());
                    }
                    else
                    {
                        substringToCompare = allele.substr(allele.size() - alleleInRead.size(), alleleInRead.size());
                    }
                    

                    if (substringToCompare == alleleInRead)
                    {
                        supportedAlleles.insert(allele);
                    }
                }

                if (supportedAlleles.empty()) continue;

                long startPositionInRead;
                long stopPositionInRead;

                if (leftIsNotEmpty)
                {
                    size_t numAlignedBasesLeft = this->countLeftBases(i, start);
                    size_t numLeftSoftClippedBases = (this->cigartuples[i].front().first == BAM_CSOFT_CLIP) ? this->cigartuples[i].front().second : 0;
                    startPositionInRead = numAlignedBasesLeft + numLeftSoftClippedBases - 1;
                    stopPositionInRead = -1;
                }
                else
                {
                    size_t numAlignedBasesLeft = this->countLeftBases(i, stop - 1);
                    size_t numLeftSoftClippedBases = (this->cigartuples[i].front().first == BAM_CSOFT_CLIP) ? this->cigartuples[i].front().second : 0;
                    startPositionInRead = -1;
                    stopPositionInRead = numAlignedBasesLeft + numLeftSoftClippedBases; 
                }
                
                if (supportedAlleles.size() == 1)
                {
                    const auto& supportedAllele = *supportedAlleles.begin();

                    DEBUG << "Read " << this->names[i] << " supports allele " << supportedAllele << " through partial overlap";

                    if (minBaseQual(i, startPositionInRead, supportedAllele.size()) >= this->qThreshold) {
                        this->supports[supportedAllele][i] = pair<long, long>(startPositionInRead, stopPositionInRead);
                    } else {
                        DEBUG << "Read " << this->names[i] << " fails quality check at variant site";
                    }
                }
                else
                {
                    DEBUG << "Read " << this->names[i] << " supports multiple alleles through partial overlaps";
                    for (auto& supportedAllele : supportedAlleles)
                    {
                        this->nonUniqueSupports[supportedAllele][i] = pair<long, long>(startPositionInRead, stopPositionInRead);
                    }
                }
            }
        }
    }

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
    if (this->nonUniqueSupports.find(allele) != this->nonUniqueSupports.end()) num += this->nonUniqueSupports.find(allele)->second.size();
    return num;
}

size_t AlleleSearcherLiteFiltered::numReadsSupportingAlleleStrict(const string& allele, bool pacbio_)
{
    size_t num = 0;
    if (this->supports.find(allele) != this->supports.end()) {
        for (auto& item: this->supports[allele]) {
            size_t readId = item.first;
            if (!(pacbio_ ^ this->pacbio[readId])) num ++;
        }
    }
    return num;
}

// Prep for advanced feature computation
void AlleleSearcherLiteFiltered::computePositionMatrixParameters()
{
    if (!this->numItemsPerMatrixPosition.empty()) return;

    // Count the number of entries per reference position etc
    size_t index = 0;
    size_t totalNumItems = 0;

    for (auto& position : this->matrix)
    {
        size_t numItems;

        // At each position we fill in the longest alt allele string
        if (!position.alt.empty())
        {
            numItems = max_element(
                position.alt.begin(),
                position.alt.end(),
                [](string a, string b) {return a.size() < b.size();}
            )->size();
        }

        numItems = max_(1, numItems);

        this->numItemsPerMatrixPosition[index] = numItems;
        this->ndarrayPositionToMatrixPosition[totalNumItems] = index;
        this->matrixPositionToNdarrayStartPosition[index] = totalNumItems;

        totalNumItems += numItems;
        index++;
    }

    this->numTotalFeatureItems = totalNumItems;
}

// Computes advanced features 
np::ndarray AlleleSearcherLiteFiltered::computeFeaturesAdvanced(const string& allele, size_t featureLength)
{
    this->computePositionMatrixParameters();

    size_t numSupports = 0;

    if (this->supports.find(allele) != this->supports.end())
    {
        numSupports += this->supports.find(allele)->second.size();
    }

    size_t numSymbols = AlleleSearcherLiteFiltered::BASE_TO_INDEX.size() + 1;
    size_t numChannels = 3 * numSymbols;
    size_t readTrackSize = 2 * numSymbols;

    if (numSupports == 0)
    {
        // Return dummy array, if there is no support
        p::tuple shape = p::make_tuple(1, featureLength, numChannels);
        np::dtype dtype = np::dtype::get_builtin<float>();
        np::ndarray array = np::zeros(shape, dtype);
        return array;
    }

    p::tuple shape = p::make_tuple(numSupports, this->numTotalFeatureItems, numChannels);
    np::dtype dtype = np::dtype::get_builtin<float>();
    np::ndarray array = np::zeros(shape, dtype);

    size_t readCounter = 0;
    const Mapping& supportItems = this->supports[allele];

    // Start filling in reads
    for (auto& supportItem : supportItems)
    {
        size_t readId = supportItem.first;
        size_t referenceStart = this->referenceStarts[readId];
        size_t matrixPosition = referenceStart - this->windowStart;
        size_t ndarrayPosition = this->matrixPositionToNdarrayStartPosition[matrixPosition];
        int mapq = this->mapq[readId];
        float p1 = 1 - this->qualityToProbability[mapq];
        int orientation = this->orientation[readId];

        while(!this->isTrackEmpty(readId, matrixPosition, matrixPosition + 1))
        {
            size_t numItemsToFill = this->numItemsPerMatrixPosition[matrixPosition];
            string stringFromReadToFill = this->getBasesInTrack(readId, matrixPosition, matrixPosition + 1);
            vector<size_t> qualsFromReadToFill = this->getQualitiesInTrack(readId, matrixPosition, matrixPosition + 1);

            // In case of indels make sure everything is right-justified
            for (size_t i = 0; i < numItemsToFill; i++)
            {
                // Add read track
                size_t offset = 0;
                float value = 0;
                float p0;
                if ((this->assemblyStart - this->windowStart <= matrixPosition) && (matrixPosition < this->assemblyStop - this->windowStart)) offset += numSymbols;

                // This is the index to use to access read bases
                int index = int(i) - int(numItemsToFill - stringFromReadToFill.size());

                // if (i < stringFromReadToFill.size())
                if (index >= 0)
                {
                    string base = stringFromReadToFill.substr(index, 1);
                    size_t qualValue = qualsFromReadToFill[index];
                    offset += this->BASE_TO_INDEX.find(base)->second;
                    p0 = 1 - this->qualityToProbability[qualValue];

                    if (this->useMapq)
                    {
                        p0 = p0 + p1 - p0 * p1;
                    }

                    // If we want quality to be encoded directly ... 
                    if (this->useQEncoding)
                    {
                        value = qualValue;
                    }
                    else
                    {
                        value = 1 - p0;
                    }
                }
                else
                {
                    offset += numSymbols - 1;    // indicates a gap
                    value = 1;
                }

                // If we want to use the orientation of the read
                // modify the value based on the orientation of the read
                if (this->useOrientation)
                {
                    value = value * orientation;
                }

                array[readCounter][ndarrayPosition + i][offset] = value;
                float refValue = (this->useMapq & this->useQEncoding) ? mapq : 1;

                // Add reference track (only the last position gets the reference base)
                if (i < numItemsToFill - 1)
                {
                    array[readCounter][ndarrayPosition + i][readTrackSize + numSymbols - 1] = refValue;
                }
                else
                {
                    // If we want to use MAPQ and if qualities should be directly encoded instead
                    // of being converted to probability scores, then directly put mapq here
                    // instead of using 1/0
                    string refBase = this->matrix[matrixPosition].ref;
                    offset = this->BASE_TO_INDEX.find(refBase)->second;
                    array[readCounter][ndarrayPosition + i][readTrackSize + offset] = refValue;
                }
            }

            matrixPosition++;
            ndarrayPosition = this->matrixPositionToNdarrayStartPosition[matrixPosition];
        }

        readCounter++;
    }

    // Copy over the matrix appropriately to truncated feature map, such that the feature map has
    // the allele positions centered
    shape = p::make_tuple(numSupports, featureLength, numChannels);
    np::ndarray featureMap = np::zeros(shape, dtype);

    // Positioning the allele site at the center of the feature map
    // the positions start, stop correspond to this->matrixPositionToNdarrayStartPosition[start - windowStart],
    // this->matrixPositionToNdarrayStartPosition[stop - windowStart]. This length of alleles must be centered.
    size_t alleleSizeInFeatureMap =  this->matrixPositionToNdarrayStartPosition[this->assemblyStop - this->windowStart] -
                                        this->matrixPositionToNdarrayStartPosition[this->assemblyStart - this->windowStart];
    // fixedPoint here corresponds to the start position of assemblyStart within the featureMap np::ndarray object
    size_t fixedPoint = (featureLength - alleleSizeInFeatureMap) / 2;
    // Left flank is the number of left-flanking positions we can fill into the feature map
    // It is limited by the number of left positions available in the full feature map, and the truncated feature map
    size_t leftFlankLength = min_(fixedPoint, this->assemblyStart - this->windowStart);
    // We use this to determine the start positions within the copyer and the copyee
    size_t startPositionInFullFeatureMap = (this->assemblyStart - this->windowStart) - leftFlankLength;
    size_t startPositionInTruncatedFeatureMap = fixedPoint - leftFlankLength;

    // Actually filling in the truncated feature map
    for (
        size_t i = startPositionInTruncatedFeatureMap, j = startPositionInFullFeatureMap;
        ((i < featureLength) && (j < this->numTotalFeatureItems));
        i++, j++
    ) {
        for (size_t r = 0; r < readCounter; r++)
        {
            for (size_t c = 0; c < numChannels; c++)
            {
                featureMap[r][i][c] = array[r][j][c];
            }
        }
    }

    return featureMap;
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
        size_t readId = item.first;

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

// Computes advanced features with coloring and not dimensions. This is similar to Google's DeepVariant scheme.
// The dataset obtained from this should be much more space efficient than before.
np::ndarray AlleleSearcherLiteFiltered::computeFeaturesColored(const string& allele, size_t featureLength)
{
    this->computePositionMatrixParameters();

    size_t numSupports = 0;

    if (this->supports.find(allele) != this->supports.end())
    {
        numSupports += this->supports.find(allele)->second.size();
    }

    size_t numChannels = 6;

    if (numSupports == 0)
    {
        // Return dummy array, if there is no support
        p::tuple shape = p::make_tuple(1, featureLength, numChannels);
        np::dtype dtype = np::dtype::get_builtin<uint8_t>();
        np::ndarray array = np::zeros(shape, dtype);
        return array;
    }

    DEBUG << "Constructing feature map";

    // We envisage a hypothetical full feature map, which covers the entire range of reads in the region
    // We do not fill the full feature map, but only use parts of it relevant for our purpose (featureLength-sized slice)
    // This featureLength-sized slice is initialized below (called truncated feature map)
    Array3D<uint8_t> array(numSupports, featureLength, numChannels);

    // Positioning the allele site at the center of the truncated feature map
    // the positions start, stop correspond to this->matrixPositionToNdarrayStartPosition[start - windowStart],
    // this->matrixPositionToNdarrayStartPosition[stop - windowStart]. This length of alleles must be centered.
    int alleleSizeInFeatureMap =  int(this->matrixPositionToNdarrayStartPosition[this->assemblyStop - this->windowStart]) -
                                        int(this->matrixPositionToNdarrayStartPosition[this->assemblyStart - this->windowStart]);
    // fixedPoint here corresponds to the start position of assemblyStart within the truncated featureMap np::ndarray object
    int fixedPoint = int(featureLength - alleleSizeInFeatureMap) / 2;

    // Left flank is the number of left-flanking positions we can fill into the feature map
    // It is limited by the number of left positions available in the full feature map, and the truncated feature map
    int leftFlankLength = min_(fixedPoint, int(this->assemblyStart) - int(this->windowStart));

    // We use this to determine the start positions within the full and truncated feature maps
    int startPositionInFullFeatureMap = int(this->matrixPositionToNdarrayStartPosition[(this->assemblyStart - this->windowStart)]) - int(leftFlankLength);
    size_t startPositionInTruncatedFeatureMap = int(fixedPoint) - int(leftFlankLength);

    // Filling in the truncated feature map below
    size_t readCounter = 0;
    const Mapping& supportItems = this->supports[allele];

    // Start filling in reads
    for (auto& supportItem : supportItems)
    {
        size_t readId = supportItem.first;
        int mapq = this->mapq[readId];
        int orientation = this->orientation[readId];
        size_t referenceStart = this->referenceStarts[readId];
        size_t matrixPosition = referenceStart - this->windowStart;
        size_t ndarrayPosition = this->matrixPositionToNdarrayStartPosition[matrixPosition];
        size_t ndarrayPositionNext = this->matrixPositionToNdarrayStartPosition[matrixPosition + 1];

        // if ndarrayPosition is beyond the start position, never mind
        if (!(ndarrayPosition > startPositionInFullFeatureMap))
        {
            // If not ...
            // Find the specific matrixPosition such that it sandwiches the start position within the full feature map
            while (!((ndarrayPosition <= startPositionInFullFeatureMap) && (ndarrayPositionNext > startPositionInFullFeatureMap)))
            {
                matrixPosition++;
                ndarrayPosition = this->matrixPositionToNdarrayStartPosition[matrixPosition];
                ndarrayPositionNext = this->matrixPositionToNdarrayStartPosition[matrixPosition + 1];
            }
        }

        bool fillCompleted = false;

        while ((!this->isTrackEmpty(readId, matrixPosition, matrixPosition + 1)) && (!fillCompleted))
        {
            size_t numItemsToFill = this->numItemsPerMatrixPosition[matrixPosition];
            string stringFromReadToFill = this->getBasesInTrack(readId, matrixPosition, matrixPosition + 1);
            vector<size_t> qualsFromReadToFill = this->getQualitiesInTrack(readId, matrixPosition, matrixPosition + 1);

            // In case of insertions make sure everything is right-justified
            for (size_t i = 0; i < numItemsToFill; i++)
            {
                int index = int(i) - int(numItemsToFill - stringFromReadToFill.size());

                int fmapindex = int(ndarrayPosition + i) - int(startPositionInFullFeatureMap) + int(startPositionInTruncatedFeatureMap);

                if ((ndarrayPosition + i) >= (startPositionInFullFeatureMap + featureLength))
                {
                    fillCompleted = true;
                    break;
                }

                // This can happen if filling doesn't start at the first position
                // (an incomplete feature from a read that is left-justified)
                if (fmapindex >= (int) featureLength)
                {
                    fillCompleted = true;
                    break;
                }

                if (ndarrayPosition + i < startPositionInFullFeatureMap)
                {
                    continue;
                }

                // index >= 0 => we need to fill something here for reads with right-justification rule
                if (index >= 0)
                {
                    char base = stringFromReadToFill[index];
                    int qual = qualsFromReadToFill[index];
                    accessArray(array, readCounter, fmapindex, READ_BASE_TRACK) = this->BaseColor(base);
                    accessArray(array, readCounter, fmapindex, READ_QUAL_TRACK) = this->BaseQualityColor(qual);
                    accessArray(array, readCounter, fmapindex, READ_MAPQ_TRACK) = this->MappingQualityColor(mapq);
                    accessArray(array, readCounter, fmapindex, READ_ORIENTATION_TRACK) = this->StrandColor(orientation);
                }

                // Every position gets the appropriate position marker
                accessArray(array, readCounter, fmapindex, POSITION_MARKER_TRACK) = this->PositionColor(matrixPosition);

                // Only the last position gets the reference base
                if (i == numItemsToFill - 1)
                {
                    string refBase = this->matrix[matrixPosition].ref;
                    assert(refBase.size() == 1);
                    accessArray(array, readCounter, fmapindex, REF_BASE_TRACK) = this->BaseColor(refBase[0]);
                }
            }

            matrixPosition++;
            ndarrayPosition = this->matrixPositionToNdarrayStartPosition[matrixPosition];
        }

        readCounter++;
    }

    // Copy over to numpy array
    p::tuple shape = p::make_tuple(numSupports, featureLength, numChannels);
    np::dtype dtype = np::dtype::get_builtin<uint8_t>();
    np::ndarray featureMapNp = np::zeros(shape, dtype);

    // Copy over into numpy array
    std::copy(array.begin(), array.end(), reinterpret_cast<uint8_t*>(featureMapNp.get_data()));

    DEBUG << "Completed constructing feature maps";

    return featureMapNp;
}

np::ndarray AlleleSearcherLiteFiltered::computeFeatures(const string& allele, size_t featureLength)
{
    // Determine shape of np::ndarray
    size_t length = featureLength;
    size_t depth = 16;
    p::tuple shape = p::make_tuple(length, depth);
    np::dtype dtype = np::dtype::get_builtin<float>();
    np::ndarray array = np::zeros(shape, dtype);

    // Center the allele
    size_t alleleStartPosition = (featureLength - allele.size()) / 2;
    size_t alleleStopPosition = alleleStartPosition + allele.size();

    if (
        (this->supports.find(allele) == this->supports.end()) && 
        (this->nonUniqueSupports.find(allele) == this->nonUniqueSupports.end())
    ) return array;

    for (size_t iter = 0; iter < 2; iter++)
    {

        Mapping supportItem;
        
        if (iter == 0)
        {
            if (this->supports.find(allele) != this->supports.end())
            {
                supportItem = this->supports.find(allele)->second;
            }
            else
            {
                DEBUG << "Found no support for allele " << allele;
                continue;
            }
        }    
        else
        {
            if (this->nonUniqueSupports.find(allele) != this->nonUniqueSupports.end())
            {
                supportItem = nonUniqueSupports.find(allele)->second;
            }
            else
            {
                DEBUG << "Found no non-unique support for allele " << allele;
                continue;
            }
        }
        

        for (auto& supportingReadEntry : supportItem)
        {
            size_t readId = supportingReadEntry.first;
            DEBUG << "Read " << this->names[readId] << " supports allele " << allele << ((iter == 0) ? " fully" : " partially");
            const vector<size_t>& quality = this->qualities[readId];
            const string& read = this->reads[readId];
            pair<long, long> readRange = supportingReadEntry.second;
            Coupling fillRange;

            DEBUG << "Read range is " << readRange.first << ", " << readRange.second;

            if (readRange.first >= 0)
            {
                pair<size_t, size_t> fillRangeStart;
                pair<size_t, size_t> fillRangeStop;
                fillRangeStart.first = alleleStartPosition > readRange.first ? alleleStartPosition - readRange.first : 0;
                fillRangeStart.second = alleleStartPosition > readRange.first ? 0 : readRange.first - alleleStartPosition;
                size_t featureFillLength = min_(featureLength - fillRangeStart.first, read.size() - fillRangeStart.second);
                fillRangeStop.first = fillRangeStart.first + featureFillLength;
                fillRangeStop.second = fillRangeStart.second + featureFillLength;
                fillRange = Coupling(fillRangeStart, fillRangeStop);
            }
            else if (readRange.second >= 0)
            {
                pair<size_t, size_t> fillRangeStart;
                pair<size_t, size_t> fillRangeStop;
                fillRangeStart.first = alleleStopPosition > readRange.second ? alleleStopPosition - readRange.second : 0;
                fillRangeStart.second = alleleStopPosition > readRange.second ? 0 : readRange.second - alleleStopPosition;
                size_t featureFillLength = min_(featureLength - fillRangeStart.first, read.size() - fillRangeStart.second);
                fillRangeStop.first = fillRangeStart.first + featureFillLength;
                fillRangeStop.second = fillRangeStart.second + featureFillLength;
                fillRange = Coupling(fillRangeStart, fillRangeStop);
            }

            DEBUG << "Filling in feature map in range " << fillRange.first.first << ", " << fillRange.second.first;

            for (size_t i = fillRange.first.first, j = fillRange.first.second; i < fillRange.second.first; i++, j++)
            {
                size_t offset = (alleleStartPosition <= i) && (i < alleleStopPosition) ? 4 : 0;
                offset += (iter == 1) ? 8 : 0;
                size_t q = quality[j];
                string base = read.substr(j, 1);
                float value = this->qualityToProbability[q];
                array[i][offset + AlleleSearcherLiteFiltered::BASE_TO_INDEX.find(base)->second] += value;
            }
        }
    }

    return array;
}

size_t AlleleSearcherLiteFiltered::coverage(size_t position_)
{
    size_t position = position_ - this->windowStart;
    return this->matrix[position].coverage;
}

AlleleSearcherLiteFiltered::~AlleleSearcherLiteFiltered()
{ }
