#ifndef _TRIE_H_
#define _TRIE_H_
#include <vector>
#include <memory>
#include <forward_list>
#include <string>
#include "Read.h"
#include "Reference.h"

using namespace std;
namespace p = boost::python;

#define MAX_NUM_ERRORS 10000

// A trie representing local variant structure
struct VariantTrie {
    vector<SiteRecord> records;
    shared_ptr<Reference> ref;
    pair<vector<AllelicRecord>, vector<AllelicRecord>> min_haplotypes;
    pair<long, long> num_errors;
    long segment_stop;
    long segment_start;

    VariantTrie(const p::list&, const string&, long left);
    void search_haplotype_pair(const string&, const string&, long, long);
    vector<pair<string, string> > get_best_matching_variants();
    bool search_path(const string&, vector<AllelicRecord>&, long, long, long);
    bool success();
};

#endif