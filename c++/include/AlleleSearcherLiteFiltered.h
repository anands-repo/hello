#ifndef ALLELE_SEARCHER_LITE_FILTERED_H
#define ALLELE_SEARCHER_LITE_FILTERED_H
#include <vector>
#include <unordered_map>
#include <string>
#include <cstdlib>
#include <unordered_set>
#include <map>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/range/adaptor/reversed.hpp>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <memory>
#include <math.h>
#include <boost/python/stl_iterator.hpp>
#include <boost/log/common.hpp>
#include <boost/log/sinks.hpp>
#include <boost/log/sources/logger.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/core/null_deleter.hpp>
#include <boost/functional/hash.hpp>
#include <exception>
#include "utils.h"
#include "Read.h"

#define CLEAR(data) if (!data.empty()) data.clear();

namespace p = boost::python;
namespace np = boost::python::numpy;
using namespace std;
using namespace boost::log;

typedef vector<pair<size_t, size_t> > Cigar;
typedef vector<pair<long, long> > AlignedPair;
typedef unordered_map<size_t, pair<long, long> > Mapping;
typedef pair<pair<size_t, size_t>, pair<size_t, size_t> > Coupling;

#define LIST_TO_SIZET_VECTOR(x) \
    vector<size_t>( \
        p::stl_input_iterator<size_t>(x), \
        p::stl_input_iterator<size_t>() \
    )

#define LIST_TO_INT_VECTOR(x) \
    vector<int>( \
        p::stl_input_iterator<int>(x), \
        p::stl_input_iterator<int>() \
    )


#define LIST_TO_BOOL_VECTOR(x) \
    vector<bool>( \
        p::stl_input_iterator<bool>(x), \
        p::stl_input_iterator<bool>() \
    )

typedef unordered_map< pair<string, string>, int, boost::hash<pair<string, string> > >  Counts;
typedef unordered_map<pair<string, string>, set<pair<string, string> >, boost::hash<pair<string, string> > > PartialMatchTracker;

// A position in the reference matrix
struct PositionLite
{
    size_t pos;
    string ref;
    vector<string> alt;
    vector<vector<size_t> > qual;
    int score;
    int coverage;
    unordered_set<string> vtypes;
    bool marked;
    PositionLite(const string& ref, const size_t numTracks, const size_t pos);
    void addTrackElement(const size_t trackId, const string& base, const size_t qual, long readPos);
};

struct AlleleCounts {
    long pos;
    int refCount;
    float total = 0;
    Counts altCounts;
    Counts leftPartialAltCounts;
    Counts rightPartialAltCounts;
    void resolvePartials();
};

struct AlleleSearcherLiteFiltered
{
    const static unordered_map<string, size_t> BASE_TO_INDEX;
    vector<string> reads;
    vector<Read> read_objs;
    vector<bool> pacbio;
    vector<vector<size_t> > qualities;
    vector<size_t> referenceStarts;
    vector<size_t> mapq;
    vector<int> orientation;
    size_t windowStart;
    size_t qThreshold;
    string reference;
    vector<PositionLite> matrix;
    unordered_set<string> alleles;
    vector<string> names;
    float qualityToProbability[100];
    sources::severity_logger<int> lg;
    size_t mismatchScore;
    size_t insertScore;
    size_t deleteScore;
    size_t maxAlleleSize;
    string refAllele;
    vector<pair<size_t, size_t> > differingRegions;
    vector<string> allelesAtSite;
    vector<string> filteredContigs;
    vector<Cigar> cigartuples;
    vector<AlignedPair> alignedPairs;
    // unordered_map<string, Mapping> supports;
    // unordered_map<string, Mapping> nonUniqueSupports;
    unordered_map<string, vector<size_t>> supports;
    size_t assemblyStart;
    size_t assemblyStop;
    // Items for advanced feature map creation
    size_t numTotalFeatureItems;
    bool useMapq;
    bool useOrientation;
    bool useQEncoding;
    int base_color_offset_a_and_g;
    int base_color_offset_t_and_c;
    int base_color_stride;
    int base_quality_cap;
    int mapping_quality_cap;
    int positive_strand_color;
    int negative_strand_color;
    int allele_position_color;
    int background_position_color;
    int READ_BASE_TRACK;
    int REF_BASE_TRACK;
    int READ_QUAL_TRACK;
    int READ_MAPQ_TRACK;
    int READ_ORIENTATION_TRACK;
    int POSITION_MARKER_TRACK;
    float snvThreshold;
    float indelThreshold;
    size_t minCount;
    size_t minMapQ;
    unordered_set<string> allelesForAssembly;
    vector<AlleleCounts> counts;
    
    void updateAlleleCounts();
    void listToQuality(const p::list& qualities);
    void listToCigar(const p::list& cigartuples);
    void addRead(
        const size_t index,
        const size_t referenceStart,
        const vector<pair<long,long> >& alignedPairs
    );
    bool indelInCluster(const vector<size_t>&);
    void pushRegions(vector<size_t>&, vector<pair<size_t,size_t> >&);
    bool isTrackEmpty(const size_t, const size_t, const size_t);
    string getBasesInTrack(const size_t, const int, const int);
    vector<size_t> getQualitiesInTrack(size_t, int, int);
    bool doesAlleleMeetQuality(size_t, size_t, size_t);
    void getAlignedPairs();
    bool isEmptyAlleleInCluster(size_t, size_t);
    void determineDifferingRegions();
    void assemble(size_t, size_t, bool);
    np::ndarray computeFeaturesColoredSimple(const string&, size_t, bool);
    size_t numReadsSupportingAllele(const string&);
    size_t numReadsSupportingAlleleStrict(const string&, bool);
    vector<string> determineAllelesAtSite(size_t, size_t);
    size_t coverage(size_t);
    pair<size_t, size_t> expandRegion(size_t, size_t);
    size_t countLeftBases(size_t, size_t);
    size_t countRightBases(size_t, size_t);
    string getLeftBases(size_t, size_t);
    string getRightBases(size_t, size_t);
    int BaseColor(char base);
    int BaseQualityColor(int qual);
    int MappingQualityColor(int qual);
    int StrandColor(int);
    int PositionColor(size_t);
    void addAlleleForAssembly(const string&);
    void clearAllelesForAssembly();
    void prepMatrix();
    void prepReadObjs();

    AlleleSearcherLiteFiltered(
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
        size_t qThreshold = 10,
        bool leftAlign = false
    );
        // Boost.Python places an upper limit of 15 arguments for the init function
        // https://www.boost.org/doc/libs/1_41_0/libs/python/doc/v2/configuration.html

    virtual ~AlleleSearcherLiteFiltered();
};

#endif
