// © 2019 University of Illinois Board of Trustees.  All rights reserved
#include "utils.h"
#include "leftAlignCigars.h"

bool indelInCigar(const vector<pair<size_t, size_t> >& cigartuples)
{
    for (auto& item : cigartuples)
    {
        if (
            (item.first == BAM_CINS) ||
            (item.first == BAM_CDEL)
        )
            return true;
    }

    return false;
}

size_t countMismatches(
    const string& read,
    size_t referenceStart,
    const vector<pair<size_t, size_t> >& cigartuples,
    const Reference& reference
)
{
    size_t numMismatches = 0;
    size_t readCounter = 0;
    size_t refCounter = referenceStart;

    for (auto& cigartuple : cigartuples)
    {
        size_t operation = cigartuple.first;
        size_t length = cigartuple.second;

        switch (operation)
        {
            case BAM_CINS:
            case BAM_CSOFT_CLIP:
            {
                readCounter += length;
                break;
            }
            case BAM_CEQUAL:
            case BAM_CDIFF:
            case BAM_CMATCH:
            {
                for (size_t i = 0; i < length; i++)
                {
                    if (reference[refCounter] != read[readCounter])
                    {
                        numMismatches++;
                    }

                    readCounter++;
                    refCounter++;
                }
                break;
            }
            case BAM_CDEL:
            case BAM_CREF_SKIP:
            {
                refCounter += length;
                break;
            }
            default: break;
        }
    }

    return numMismatches;
}

// Determine the reference and read positions at the end of a cigartuple
pair<size_t, size_t> refReadPosition(const vector<pair<size_t, size_t> >& cigartuples, size_t referenceStart)
{
    size_t readCounter = 0;
    size_t refCounter = referenceStart;

    for (auto& cigar : cigartuples)
    {
        const auto& operation = cigar.first;
        const auto& length = cigar.second;

        switch(operation)
        {
            case BAM_CMATCH:
            case BAM_CDIFF:
            case BAM_CEQUAL:
            {
                readCounter += length;
                refCounter += length;
                break;
            }

            case BAM_CINS:
            case BAM_CSOFT_CLIP:
            {
                readCounter += length;
                break;
            }

            case BAM_CDEL:
            case BAM_CREF_SKIP:
            {
                refCounter += length;
                break;
            }
            default: break;
        }
    }

    return pair<size_t, size_t>(refCounter, readCounter);
}

bool leftShiftCigar(
    const string& read,
    size_t& cigarIndex,
    const size_t origMismatches,
    size_t referenceStart,
    const vector<pair<size_t, size_t> >& cigartuples,
    const Reference& reference,
    vector<pair<size_t, size_t> >& leftShiftedCigars
)
{
    // Proceed if cigar entry at cigarIndex is an indel
    if (
        (cigartuples.at(cigarIndex).first != BAM_CDEL) &&
        (cigartuples.at(cigarIndex).first != BAM_CINS)
    ) return false;

    bool cigarHasShiftedLeft = false;

    // First, check whether the cigar entry to the left of
    // cigarIndex is a match
    if (cigarIndex == 0) return false;
    size_t leftOperation = cigartuples.at(cigarIndex - 1).first;
    if (
        (leftOperation != BAM_CMATCH) &&
        (leftOperation != BAM_CEQUAL)
    )
    {
        return false;
    }

    // If its a cigar match, check if its an actual match (BAM_CMATCH indicates mismatches as well)
    if (leftOperation == BAM_CMATCH)
    {
        vector<pair<size_t, size_t> > cigarslice(cigartuples.begin(), cigartuples.begin() + cigarIndex);
        pair<size_t, size_t> positions = refReadPosition(cigarslice, referenceStart);
        long refPosition = long(positions.first) - 1;
        long readPosition = long(positions.second) - 1;

        if ((refPosition < 0) || (readPosition < 0)) return false;

        // If there is a mismatch to the left of the cigartuple, do not process further
        if (reference[refPosition] != read[readPosition]) return false;
    }

    // Next produce the left-shifted cigartuple //

    // First, fill in cigars until cigarIndex - 1
    for (size_t i = 0; i < cigarIndex - 1; i++)
    {
        leftShiftedCigars.push_back(cigartuples.at(i));
    }

    // Next, modify cigar at index cigarIndex - 1
    pair<size_t, size_t> leftCigar(cigartuples.at(cigarIndex - 1));
    leftCigar.second--;

    if (leftCigar.second > 0)
    {
        leftShiftedCigars.push_back(leftCigar);
    }
    else
    {
        cigarHasShiftedLeft = true;
    }
    
    // Next, insert the cigar at cigarIndex
    leftShiftedCigars.push_back(cigartuples.at(cigarIndex));

    // Next, modify cigar at index cigarIndex + 1, or add a new one
    if (cigarIndex < cigartuples.size() - 1)
    {
        // Modify if the left and right cigars in original were of the same type
        // Otherwise, add a new cigar to the right
        const pair<size_t, size_t>& origLeftCigar = cigartuples.at(cigarIndex - 1);
        const pair<size_t, size_t>& origRightCigar = cigartuples.at(cigarIndex + 1);

        if (origLeftCigar.first == origRightCigar.first)
        {
            pair<size_t, size_t> newRightCigar(origRightCigar);
            newRightCigar.second++;
            leftShiftedCigars.push_back(newRightCigar);
        }
        else
        {
            pair<size_t, size_t> newRightCigar;
            newRightCigar.first = origLeftCigar.first;
            newRightCigar.second = 1;
            leftShiftedCigars.push_back(newRightCigar);
            leftShiftedCigars.push_back(origRightCigar);
        }
    }

    // Next, add in the remaining cigars
    for (size_t i = cigarIndex + 2; i < cigartuples.size(); i++)
    {
        leftShiftedCigars.push_back(cigartuples.at(i));
    }

    // Compute the number of mismatches
    size_t newNumMismatches = countMismatches(read, referenceStart, leftShiftedCigars, reference);

    bool success = (newNumMismatches == origMismatches);

    if (success & cigarHasShiftedLeft)
    {
        cigarIndex--;
    }

    return success;
}

void simplifyCigartuples(
    vector<pair<size_t, size_t> >& cigartuples,
    const string& read,
    size_t referenceStart,
    const Reference& reference
)
{
    // Combine adjacent cigartuples that are identical
    // Merge adjacent insertions and deletions into matches
    bool completed = true;
    bool changeDetected = false;
    vector<pair<size_t, size_t> > simplifiedCigartuples;
    vector<pair<size_t, size_t> > cigartuplesCopy(cigartuples);

    do
    {
        simplifiedCigartuples.clear();
        completed = true;

        size_t refCounter = referenceStart;
        size_t readCounter = 0;

        for (auto& cigar : cigartuplesCopy)
        {
            if (simplifiedCigartuples.empty())
            {
                simplifiedCigartuples.push_back(cigar);
            }
            else
            {
                pair<size_t, size_t> lastCigar(simplifiedCigartuples.back());

                if (lastCigar.first == cigar.first)
                {
                    simplifiedCigartuples.back().second += cigar.second;
                    completed = false;
                }
                else
                {
                    if (
                            (
                                (lastCigar.first == BAM_CINS) &&
                                (cigar.first == BAM_CDEL)
                            ) ||
                            (
                                (lastCigar.first == BAM_CDEL) &&
                                (cigar.first == BAM_CINS)
                            )
                    )
                    {
                        // If insertion and deletion follow each other
                        // and if they are of the same length, AND their combination
                        // results in only matches, then combine them
                        if (lastCigar.second == cigar.second)
                        {
                            bool allMatches = true;

                            for (size_t i = readCounter, j = refCounter; i < readCounter + cigar.second; i++, j++)
                            {
                                if (read[i] != reference[j])
                                {
                                    allMatches = false;
                                    break;
                                }
                            }

                            if (allMatches)
                            {
                                simplifiedCigartuples.back().first = BAM_CMATCH;
                                completed = false;
                            }
                        }
                    }
                    else
                    {
                        simplifiedCigartuples.push_back(cigar);
                    } // else
                } // else
            } // else

            // Move indices
            switch (cigar.first)
            {
                case BAM_CINS:
                case BAM_CSOFT_CLIP:
                {
                    readCounter += cigar.second;
                    break;
                }
                case BAM_CEQUAL:
                case BAM_CDIFF:
                case BAM_CMATCH:
                {
                    readCounter += cigar.second;
                    refCounter += cigar.second;
                    break;
                }
                case BAM_CDEL:
                case BAM_CREF_SKIP:
                {
                    refCounter += cigar.second;
                    break;
                }
                default: break;
            }

        } // for (auto& cigar)

        cigartuplesCopy.clear();
        cigartuplesCopy.insert(cigartuplesCopy.begin(), simplifiedCigartuples.begin(), simplifiedCigartuples.end());

        if (!completed)
        {
            changeDetected = true;
        }
    } while(!completed);

    if (changeDetected)
    {
        cigartuples.clear();
        cigartuples.insert(cigartuples.begin(), simplifiedCigartuples.begin(), simplifiedCigartuples.end());
    }
}

void removeLeadingDeletions(
    vector<pair<size_t, size_t> >& cigartuples, size_t& referenceStart
)
{
    size_t offsetToDelete = 0;

    for (auto& cigar : cigartuples)
    {
        if (cigar.first != BAM_CDEL) break;
        offsetToDelete += 1;
    }

    if (offsetToDelete > 0)
    {
        cigartuples.erase(cigartuples.begin(), cigartuples.begin() + offsetToDelete);
        referenceStart += offsetToDelete;
    }
}

void leftAlignCigars(
    const string& read,
    size_t& referenceStart,
    vector<pair<size_t, size_t> >& cigartuples,
    const Reference& reference,
    bool indelRealigned
)
{
    if (!indelInCigar(cigartuples)) return;

    size_t cigarIndex = 0;
    size_t origMismatches = countMismatches(
        read,
        referenceStart,
        cigartuples,
        reference
    );

    // Perform left shifts
    while (cigarIndex < cigartuples.size())
    {
        vector<pair<size_t, size_t> > leftShiftedCigars;
        while(
            leftShiftCigar(
                read,
                cigarIndex,
                origMismatches,
                referenceStart,
                cigartuples,
                reference,
                leftShiftedCigars
            )
        )
        {
            cigartuples.clear();
            cigartuples.insert(cigartuples.begin(), leftShiftedCigars.begin(), leftShiftedCigars.end());
            leftShiftedCigars.clear();
        }

        cigarIndex++;
    }

    // Simplify cigartuples
    simplifyCigartuples(cigartuples, read, referenceStart, reference);

    // Remove leading deletions
    removeLeadingDeletions(cigartuples, referenceStart);

    // Convert leading insertion to soft clipped bases; if indel realigned
    // do not do this, since soft-clipped bases may have been rescued by the
    // indel realignment process
    if (!indelRealigned)
    {
        if (cigartuples.front().first == BAM_CINS)
        {
            cigartuples.front().first = BAM_CSOFT_CLIP;
        }
    }
}

#ifdef TEST
#include <sstream>
#include <iostream>

void printCigar(const vector<pair<size_t, size_t> >& cigars)
{
    ostringstream sstr;
    for (auto& cigar : cigars)
    {
        sstr << "(" << cigar.first << ", " << cigar.second << "), ";
    }
    cout << sstr.str() << endl;
}

int main()
{
    // See ../python/PileupContainer.py for more details

    string reference_ = "ACGATATATACCAGTATATATATATATATATATATATATAGGATACGATA";
    Reference reference(reference_, 0);
    vector<pair<size_t, size_t> > cigartuples;

    // First test
    string read = "TATACCAGTATATATATATATATATATATATATAGGA";
    size_t referenceStart = 6;
    cigartuples.push_back(pair<size_t, size_t> (BAM_CMATCH, read.size()));
    leftAlignCigars(read, referenceStart, cigartuples, reference);
    printCigar(cigartuples);


    // Second test
    read = "TATACCAGTATATATATATATATATATATATAGGA";
    referenceStart = 6;
    cigartuples.clear();
    cigartuples.push_back(pair<size_t, size_t>(BAM_CMATCH, 25));
    cigartuples.push_back(pair<size_t, size_t>(BAM_CDEL, 2));
    cigartuples.push_back(pair<size_t, size_t>(BAM_CMATCH, 10));
    leftAlignCigars(read, referenceStart, cigartuples, reference);
    printCigar(cigartuples);

    // Third test
    read = "TATACCAGTATATATATATATATATATATATATAGGA";
    referenceStart = 6;
    cigartuples.clear();
    cigartuples.push_back(pair<size_t, size_t>(BAM_CMATCH, 10));
    cigartuples.push_back(pair<size_t, size_t>(BAM_CINS, 2));
    cigartuples.push_back(pair<size_t, size_t>(BAM_CMATCH, 15));
    cigartuples.push_back(pair<size_t, size_t>(BAM_CDEL, 2));
    cigartuples.push_back(pair<size_t, size_t>(BAM_CMATCH, 10));
    leftAlignCigars(read, referenceStart, cigartuples, reference);
    printCigar(cigartuples);

    // Fourth test
    read = "TATACCAGTATAGATATATATATATATATATATAGGA";
    referenceStart = 6;
    cigartuples.clear();
    cigartuples.push_back(pair<size_t, size_t>(BAM_CMATCH, 12));
    cigartuples.push_back(pair<size_t, size_t>(BAM_CINS, 1));
    cigartuples.push_back(pair<size_t, size_t>(BAM_CDEL, 1));
    cigartuples.push_back(pair<size_t, size_t>(BAM_CMATCH, 24));
    leftAlignCigars(read, referenceStart, cigartuples, reference);
    printCigar(cigartuples);

    // Final test
    read = "TATATATATATATATATATATATAGGATACTTTT";
    referenceStart = 14;
    cigartuples.clear();
    cigartuples.push_back(pair<size_t, size_t>(BAM_CMATCH, 2));
    cigartuples.push_back(pair<size_t, size_t>(BAM_CDEL, 2));
    cigartuples.push_back(pair<size_t, size_t>(BAM_CMATCH, 28));
    cigartuples.push_back(pair<size_t, size_t>(BAM_CSOFT_CLIP, 4));
    leftAlignCigars(read, referenceStart, cigartuples, reference);
    printCigar(cigartuples);

    // Add a new test : left shift across mismatch shouldn't happen
    /*
    Note that, at this position, left shift would usually create problems
    if there were a C->T mismatch at this position.
                                         |
    Reference: ACGATATATACCAGTATATATATATACATATATATATATAGGATACGATA
    Read:                 CAGTATATATATATATATATAT--ATATAGG   
    */
    reference_ = "ACGATATATACCAGTATATATATATACATATATATATATAGGATACGATA";
    Reference ref(reference_, 0);
    read = "CAGTATATATATATATATATATATATAGG";
    referenceStart = 11;
    cigartuples.clear();
    cigartuples.push_back(pair<size_t, size_t>(BAM_CMATCH, 22));
    cigartuples.push_back(pair<size_t, size_t>(BAM_CDEL, 2));
    cigartuples.push_back(pair<size_t, size_t>(BAM_CMATCH, 7));
    leftAlignCigars(read, referenceStart, cigartuples, ref);
    printCigar(cigartuples); // Expected 16M, 2D, 13M
                             // Without mismatch check it would be 3M, 2D, 26M

    return -1;
}
#endif