// © 2019 University of Illinois Board of Trustees.  All rights reserved
#ifndef _REFERENCE_H_
#define _REFERENCE_H_

#include <string>
#include <cassert>

using namespace std;

struct Reference
{
    const string refString;
    const size_t leftPosition;
    const char& operator[](size_t) const;
    Reference(const string&, size_t);
};

#endif