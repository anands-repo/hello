// © 2019 University of Illinois Board of Trustees.  All rights reserved
#include "utils.h"

using namespace std;

#ifndef TEST
using namespace boost::python;

vector<string> strListToVector(const boost::python::list& data)
{
    vector<string> converted;

    for (size_t i = 0; i < (size_t) len(data); i += 1)
    {
        converted.push_back(extract<string>(data[i]));
    }

    return converted;
}

void splitStrListToKmers(const boost::python::list& strings, vector<string>& kmers, size_t k)
{
    for (size_t i = 0; i < (size_t) len(strings); i += 1)
    {
        size_t numKmers = len(strings[i]) - k + 1;
        string string_  = extract<string>(strings[i]);

        for (size_t j = 0; j < numKmers; j += 1)
        {
            kmers.push_back(string_.substr(j,k));
        }
    }
}
#endif

void splitStrVectorToKmers(const vector<string>& strings, vector<string>& kmers, size_t k)
{
    for (size_t i = 0; i < strings.size(); i += 1)
    {
        string string_  = strings[i];
        size_t numKmers = string_.size() - k + 1;

        for (size_t j = 0; j < numKmers; j += 1)
        {
            kmers.push_back(string_.substr(j,k));
        }
    }
}

size_t baseMapping(char base)
{
    if (base == 'A') return 0;
    if (base == 'C') return 1;
    if (base == 'G') return 2;
    if (base == 'T') return 3;
    if (base == 'N') return 4;
    return -1;
}

char integerMapping(size_t base)
{
    if (base == 0) return 'A';
    if (base == 1) return 'C';
    if (base == 2) return 'G';
    if (base == 3) return 'T';
    if (base == 4) return 'N';
    return -1;
}

bool isOverlapping(const pair<size_t, size_t>& a, const pair<size_t, size_t>& b)
{
    assert(a.second >= a.first);
    assert(b.second >= b.first);

    // a contains the starting of b (superset of "a contains b", "b contains the ending of a")
    if ((a.first <= b.first) && (b.first <= a.second)) return true;

    // b contains the starting of a (superset of "b contains a", "a contains the ending of b")
    if ((b.first <= a.first) && (a.first <= b.second)) return true;
}

bool isContaining(const pair<size_t, size_t>& a, const pair<size_t, size_t>& b)
{
    assert(a.second >= a.first);
    assert(b.second >= b.first);

    return ((a.first <= b.first) & (a.second >= b.second));
}