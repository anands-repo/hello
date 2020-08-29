#ifndef _READ_H_
#define _READ_H_
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <sstream>
#include <algorithm>
#include <map>
#include "Reference.h"
#ifndef _READ_TEST_
#include "utils.h"
#include <boost/python.hpp>
namespace p = boost::python;
#endif
#include <iostream>

// This is patchy, but for testing purposes
// this needs to be done
#ifdef _READ_TEST_
#include <iostream>
#define BAM_CMATCH       0
#define BAM_CINS         1
#define BAM_CDEL         2
#define BAM_CREF_SKIP    3
#define BAM_CSOFT_CLIP   4
#define BAM_CHARD_CLIP   5
#define BAM_CPAD         6
#define BAM_CEQUAL       7
#define BAM_CDIFF        8
#define BAM_CBACK        9
#endif

using namespace std;

enum AlignedBaseStatus { Success, Fail, LeftPartial, RightPartial };

template <class T>
vector<T> list_converter(const p::list& array) {
    return vector<T>(
        p::stl_input_iterator<T>(array),
        p::stl_input_iterator<T>()
    );
}

struct AlignedBases {
    string first;
    AlignedBaseStatus second;
    long third;
};

struct TruthSet {
    vector<pair<string, string> > truth_alleles;
    bool valid;
};

struct AllelicRecord {
    string allele;
    long start;
    long stop;
    long min_q;
    AllelicRecord(
        const string& allele_,
        long start_,
        long stop_,
        long min_q_
    ) : allele(allele_), start(start_), stop(stop_), min_q(min_q_) {}
    AllelicRecord() = default;
    AllelicRecord(const AllelicRecord&) = default;
    AllelicRecord(AllelicRecord&&) = default;
    AllelicRecord& operator=(const AllelicRecord&) = default;
};

struct SiteRecord {
    vector<string> alleles;
    long start;
    long stop;
    SiteRecord(
        const vector<string>& alleles_,
        long start_,
        long stop_
    ) : alleles(alleles_), start(start_), stop(stop_) {};
#ifndef _READ_TEST_
    SiteRecord(
        const p::list& alleles_,
        long start_,
        long stop_
    ) : alleles(list_converter<string>(alleles_)), start(start_), stop(stop_) {};
#endif
    SiteRecord() = default;
    SiteRecord(const SiteRecord&) = default;
    SiteRecord(SiteRecord&&) = default;
    SiteRecord& operator=(const SiteRecord&) = default;
};

struct Read {
    /*
    A read class which contains mapping of a read to different reference
    positions and provides alleles in the read from a position 'a' to a position 'b',
    in addition to other functions like allele reinterpretation etc
    */
    string read;
    string name;
    vector<size_t> quality;
    vector<pair<size_t, size_t>> cigartuples;
    long reference_start;
    long last_position;
    unordered_map<long, string> aligned_pairs;
    unordered_map<long, long> aligned_qualities;
    bool partial_start;
    bool partial_stop;
    bool pacbio;
    int read_id;
    long mapq;
    vector<AllelicRecord> alleles;
    AllelicRecord left_partial;
    AllelicRecord right_partial;
    map<pair<long, long>, string> allele_map;
    bool assembled;
    bool has_left_partial;
    bool has_right_partial;

    Read(
        const string& read,
        const string& name,
        const vector<size_t>& quality,
        const vector<pair<size_t, size_t>> cigartuples,
        long reference_start,
        bool pacbio,
        int read_id,
        long mapq
    ) :
    read(read),
    name(name),
    quality(quality),
    cigartuples(cigartuples),
    reference_start(reference_start),
    partial_start(false),
    partial_stop(false),
    last_position(-1),
    pacbio(pacbio),
    read_id(read_id),
    mapq(mapq),
    assembled(false),
    has_left_partial(false),
    has_right_partial(false)
    {
        _get_read_mapping();
    }

    Read() = default;
    Read(const Read& other) = default;
    Read(Read&& other) = default;

    // Get read-reference matching aligned bases
    void _get_read_mapping();
    // Get aligned bases between a reference range for the read
    AlignedBases get_aligned_bases(long, long) const;
    // Extract alleles in the read in a given set of reference ranges
    void extract_alleles(const vector<pair<size_t, size_t>>&);
    // Get the haplotype string of the read between two ranges.
    // This is different from simply copying the read bases because this
    // is formed from the alleles in the read. The alleles in the read are
    // constructed after some noise filtration
    string get_haplotype_string(const Reference&, long, long);
    // Update allelic records in the reads
    void update_allelic_records(const Reference&, const unordered_map<string, vector<AllelicRecord>>&, long, long);
    // Create an allele map from allele records
    void create_allele_map();
};

// A method to enumerate all haplotypes in short reads
void enumerate_all_haplotypes(
    vector<SiteRecord>&,
    const Reference&,
    long,
    long,
    unordered_map<string, vector<AllelicRecord>>&,
    int call_level = 0
);

// Utility function to obtain a list of all reference bases in a certain range
string get_reference_bases(const Reference&, long, long);

#ifndef _READ_TEST_
TruthSet get_ground_truth_alleles(
    const p::list&, const string&, const string&, const string&, long
);
#endif

#endif