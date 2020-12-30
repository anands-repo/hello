// © 2019 University of Illinois Board of Trustees.  All rights reserved
#include "Reference.h"

Reference::Reference(const string& refString, size_t start) : refString(refString), leftPosition(start)
{ }

const char& Reference::operator[](size_t index) const
{
    assert(index >= leftPosition);
    size_t localIndex = index - leftPosition;
    assert(localIndex < this->refString.size());
    return this->refString[localIndex];
}
