// © 2019 University of Illinois Board of Trustees.  All rights reserved
#ifndef _LEFT_ALIGN_CIGARS_
#define _LEFT_ALIGN_CIGARS_

#include <string>
#include <vector>
#include <iostream>
#include "Reference.h"

using namespace std;

bool indelInCigar(const vector<pair<size_t, size_t> >&);

size_t countMismatches(
    const string&,
    size_t,
    const vector<pair<size_t, size_t> >&,
    const Reference&
);

bool leftShiftCigar(
    const string&,
    size_t&,
    const size_t,
    size_t,
    const vector<pair<size_t, size_t> >&,
    const Reference&,
    vector<pair<size_t, size_t> >&
);

void simplifyCigartuples(
    vector<pair<size_t, size_t> >&,
    const string& read,
    size_t referenceStart,
    const Reference&
);

pair<size_t, size_t> refReadPosition(const vector<pair<size_t, size_t> >&, size_t);

void removeLeadingDeletions(vector<pair<size_t, size_t> >&, size_t&);

void leftAlignCigars(
    const string&, 
    size_t&,
    vector<pair<size_t, size_t> >&,
    const Reference&,
    bool indelRealigned = false
);

#endif