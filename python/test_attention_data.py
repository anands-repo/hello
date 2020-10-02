import libCallability
import numpy as np

"""
                I               |   | 
Reference:  ACATC--GACTGCGCGGATATCGATCGCGTACGACTCGACT
Read0:          CAAGACTGCGCGGATACCG
Read1:                    CGGATATCGAT--CGT
"""

reference = "ACATCGACTGCGCGGATATCGATCGCGTACGACTCGACT"
window_start = 0

M = 0
I = 1
D = 2

### read0
read0 = "CAAGACTGCGCGGATACCG"
cigar0 = [[M, 1], [I, 2], [M, 16]]
start0 = 4
qual0 = [40] * len(read0)

### read1
read1 = "CGGATATCGATCGT"
cigar1 = [[M, 11], [D, 2], [M, 3]] 
start1 = 12
qual1 = [40] * len(read1)

filter = libCallability.AlleleSearcherLite(
    [read0, read1],  # , read1, read1],
    ["read0", "read1"],  # , "read2", "read3"],
    [qual0, qual1],  #  qual1, qual1],
    [cigar0, cigar1],  # cigar1, cigar1],
    [start0, start1],  # start1, start1],
    [40, 40],  # 40, 40],
    [0, 0],  # 0, 0],
    [False, False],  # False, False],
    reference,
    window_start,
    0,
    len(reference),
    False
)

filter.minCount = 1
filter.initialize()
filter.determineDifferingRegions(True)
print([
    [int(d.first), int(d.second)] for d in filter.differingRegions
])

filter.assemble_alleles_from_reads(False)

filter.assemble(18, 19)

filter.gen_features("T", 30, False)
allele_tensor = filter.get_allele_tensor()
feature_tensor = filter.get_feature_tensor()

print(allele_tensor)
print(feature_tensor)

filter.gen_features("C", 30, False)
allele_tensor = filter.get_allele_tensor()
feature_tensor = filter.get_feature_tensor()

print(allele_tensor)
print(feature_tensor)
