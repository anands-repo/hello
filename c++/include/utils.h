// © 2019 University of Illinois Board of Trustees.  All rights reserved
#ifndef UTILS_H_INCLUDED
#define UTILS_H_INCLUDED
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <cassert>
#ifndef TEST
#include <boost/python.hpp>
#endif

#include <boost/functional/hash.hpp>

#define DEBUG \
    BOOST_LOG_SEV(this->lg, 0)

using namespace std;

#ifndef TEST
vector<string> strListToVector(const boost::python::list&);
void splitStrListToKmers(const boost::python::list&, vector<string>&, size_t);
#endif

void splitStrVectorToKmers(const vector<string>&, vector<string>&, size_t);
size_t baseMapping(char);
char integerMapping(size_t);

// Struct to use as a key in an unordered_map
struct PairForKey
{
    size_t n;
    float p;

    PairForKey(size_t n, float p) : n(n), p(p) {} ;

    bool operator==(const PairForKey& other) const
    {
        return ((n == other.n) && (p == other.p));
    }
};

// Hash function for pair (above)
namespace std {

  template <>
  struct hash<PairForKey>
  {
    std::size_t operator()(const PairForKey& k) const
    {
      size_t seed = 0;
      boost::hash_combine(seed, k.n);
      boost::hash_combine(seed, k.p);
      return seed;
    }
  };

}

// For sorting in descending order
struct PairGreater
{
    bool operator()(PairForKey const& a, PairForKey const& b) const
    {
        if (a.n == b.n)
        {
            return a.p > b.p;
        }
        else
        {
            return a.n > b.n;
        }
    }
};

template <class T>
size_t findNumIntersections(const unordered_set<T>& set1, const unordered_set<T>& set2)
{
    size_t numIntersections = 0;
    const unordered_set<T>* lset = set1.size() > set2.size() ? &set2 : &set1;
    const unordered_set<T>* rset = set1.size() > set2.size() ? &set1 : &set2;

    for (auto& item : *lset)
    {
        if (rset->find(item) != rset->end())
        {
            numIntersections += 1;
        }
    }

    return numIntersections;
}

template <class T1, class T2>
size_t findNumIntersections(const unordered_map<T1,T2>& map1, const unordered_map<T1,T2>& map2)
{
    size_t numIntersections = 0;
    const unordered_map<T1,T2>* lmap = map1.size() > map2.size() ? &map2 : &map1;
    const unordered_map<T1,T2>* rmap = map1.size() > map2.size() ? &map1 : &map2;

    for (auto& item : *lmap)
    {
        if (rmap->find(item.first) != rmap->end())
        {
            numIntersections += 1;
        }
    }

    return numIntersections;
}

template <class T1, class T2>
void findIntersections(const unordered_map<T1,T2>& map1, const unordered_map<T1,T2>& map2, unordered_set<T1>& keys, bool iNOtD = true)
{
    // Finds map1 \cap map2 or map1 \cap ~map2 depending on iNotD (inversion is over the union of keys of map1 and map2)
    for (auto& item : map1)
    {
        if (iNOtD)
        {
            if (map2.find(item.first) != map2.end())
            {
                keys.insert(item.first);
            }
        }
        else
        {
            if (map2.find(item.first) == map2.end())
            {
                keys.insert(item.first);
            }
        }
    }
}

#define RANGE(item) (((int) (item).second) - ((int) (item).first))

template <class K,class V>
void inline ensureKey(unordered_map<K,V>& map_, const K& k)
{
    if (map_.find(k) == map_.end())
    {
        V v;
        map_[k] = v;
    }
}

#define MAX_NUM_CONTIGS 128

vector<string> validateContigs(const vector<string>&, const vector<string>&, const vector<string>&, const vector<string>&, size_t);

bool isOverlapping(const pair<size_t, size_t>&, const pair<size_t, size_t>&);

bool isContaining(const pair<size_t, size_t>&, const pair<size_t, size_t>&);

#define extractFromList(l,x,y,z) \
    p::list(p::list(p::list(l[x])[y]))[z]

#define max_(a,b) (a > b ? a : b)

#define min_(a,b) (a < b ? a : b)

#define pushRegion \
    if ((region.first >= this->regionStart) && \
        (region.second <= this->regionEnd)) \
    { \
        regions.push_back(region); \
    }

#define indelAtSite(site) \
    ((site.vtypes.find("D") != site.vtypes.end()) || (site.vtypes.find("I") != site.vtypes.end()))

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
#define LOWCOV_FRACTION  0.35
#define HIGHCOV_FRACTION 0.2
#define LOWCOV_CUTOFF    9

#endif // UTILS_H_INCLUDED