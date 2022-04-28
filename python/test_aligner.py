import operator
from numpy.core.einsumfunc import _parse_possible_contraction

from numpy.core.numeric import _tensordot_dispatcher
import libCallability
import random
from functools import reduce, partial
from operator import concat
from collections import defaultdict, namedtuple
import numpy as np
from typing import List, Dict
import logging


READ_BASE_TRACK = 0
REF_BASE_TRACK = 1
READ_QUAL_TRACK = 2
READ_MAPQ_TRACK = 3
READ_ORIENTATION_TRACK = 4
POSITION_MARKER_TRACK = 5
HP_TRACK = 6
BAM_CMATCH = 0
BAM_CINS = 1
BAM_CDEL = 2
BAM_CREF_SKIP = 3
BAM_CSOFT_CLIP = 4
BAM_CHARD_CLIP = 5
BAM_CPAD = 6
BAM_CEQUAL = 7
BAM_CDIFF = 8
BAM_CBACK = 9

BaseOffsets__ = dict(zip("AGTC", [40, 40, 30, 30]))
BaseColor__ = dict(
    map(
        lambda x: (x[1], BaseOffsets__[x[1]] + x[0] * 70),
        enumerate("CTGA")
    )
)
BaseColor__['*'] = 0
StrandColor__ = {-1: 240, +1: 70}
HPColor__ = defaultdict(int)
HPColor__[1] = 120
HPColor__[2] = 240


def QualityColor(qual: int, cap: int) -> int:
    qual = min(qual, cap)
    return int(254 * (qual / cap))


def PositionColor(pos: int, variant_range: tuple) -> int:
    return 240 if variant_range[0] <= pos < variant_range[1] else 70


BaseColor = lambda x: BaseColor__[x]
BaseQualityColor = partial(QualityColor, cap=40)
MAPQColor = partial(QualityColor, cap=60)
StrandColor = lambda x: StrandColor__[x]
HPColor = lambda x: HPColor__[x]


def random_reference(length: int) -> str:
    return "".join(
        reduce(
            lambda a, b: concat(a, b),
            map(lambda i: random.sample("ACGT", k=1), range(length)),
            []
        )
    )


ReadDescriptor = namedtuple(
    "ReadDescriptor",
    ["read", "name", "quality", "cigartuples", "reference_start", "mapq", "orientation", "pacbio", "hp"],
    defaults=[40, 1, False, None,]
)


def write_to_array(
    array: np.ndarray,
    feature_ptr: int,
    ref_base: str,
    read_base: str,
    read_quality: int,
    mapq: int,
    orientation: int,
    ref_ptr: int,
    variant_range: tuple,
    hp: int = None,
):
    array[REF_BASE_TRACK, feature_ptr] = BaseColor(ref_base)
    array[READ_BASE_TRACK, feature_ptr] = BaseColor(read_base)
    array[READ_QUAL_TRACK, feature_ptr] = BaseQualityColor(read_quality)
    array[READ_MAPQ_TRACK, feature_ptr] = MAPQColor(mapq)
    array[READ_ORIENTATION_TRACK, feature_ptr] = StrandColor(orientation)
    array[POSITION_MARKER_TRACK, feature_ptr] = PositionColor(ref_ptr, variant_range)
    if hp is not None:
        array[HP_TRACK, feature_ptr] = HPColor(hp)


def create_read_encoding(
    read_desc: ReadDescriptor,
    reference_string: str,
    feature_length: int,
    variant_range: tuple
) -> np.ndarray:
    channel_size = 7 if read_desc.hp is not None else 6
    array = np.zeros(shape=(channel_size, feature_length), dtype=np.int32)
    midpoint = sum(variant_range) // 2
    start_point = midpoint - feature_length // 2
    end_point = start_point + feature_length
    read_ptr = 0
    ref_ptr = read_desc.reference_start
    alleles_in_region = ""

    for operation, length in read_desc.cigartuples:
        if operation in [BAM_CEQUAL, BAM_CDIFF, BAM_CMATCH]:
            for i in range(length):
                if start_point <= ref_ptr < end_point:
                    feature_ptr = ref_ptr - start_point
                    write_to_array(
                        array,
                        feature_ptr,
                        ref_base=reference_string[ref_ptr],
                        read_base=read_desc.read[read_ptr],
                        read_quality=read_desc.quality[read_ptr],
                        mapq=read_desc.mapq,
                        orientation=read_desc.orientation,
                        ref_ptr=ref_ptr,
                        variant_range=variant_range,
                        hp=read_desc.hp,
                    )

                if variant_range[0] <= ref_ptr < variant_range[1]:
                    alleles_in_region += read_desc.read[read_ptr]

                ref_ptr += 1
                read_ptr += 1
        elif operation in [BAM_CDEL, BAM_CREF_SKIP]:
            if operation == BAM_CDEL:
                for i in range(-1, length):
                    if start_point <= ref_ptr < end_point:
                        feature_ptr = ref_ptr + i - start_point
                        write_to_array(
                            array,
                            feature_ptr,
                            ref_base=reference_string[ref_ptr + i],
                            read_base="*",
                            read_quality=(read_desc.quality[read_ptr - 1] if i == -1 else 0),
                            mapq=read_desc.mapq,
                            orientation=read_desc.orientation,
                            ref_ptr=ref_ptr + i,
                            variant_range=variant_range,
                            hp=read_desc.hp,
                        )
            ref_ptr += length
        elif operation in [BAM_CINS, BAM_CREF_SKIP]:
            if operation == BAM_CINS:
                if start_point <= ref_ptr - 1 < end_point:
                    feature_ptr = ref_ptr - 1 - start_point
                    write_to_array(
                        array,
                        feature_ptr=feature_ptr,
                        ref_base=reference_string[ref_ptr - 1],
                        read_base='*',
                        read_quality=read_desc.quality[read_ptr - 1],
                        mapq=read_desc.mapq,
                        orientation=read_desc.orientation,
                        ref_ptr=ref_ptr - 1,
                        variant_range=variant_range,
                        hp=read_desc.hp,
                    )

            if variant_range[0] <= ref_ptr - 1 < variant_range[1]:
                alleles_in_region += read_desc.read[read_ptr: read_ptr + length]

            read_ptr += length

    return array, alleles_in_region


class TestCase:
    def __init__(self, reference: str, feature_length: int = 10):
        self.reference: str = reference
        self.reads: List[ReadDescriptor] = []
        self.encodings: Dict[str, List[np.ndarray]] = {}
        self.feature_length: int = feature_length
        self._variant_start: int = None
        self._activity_size: int = None

    @property
    def variant_start(self):
        return self._variant_start

    @variant_start.setter
    def variant_start(self, v):
        self._variant_start = v

    @property
    def activity_size(self):
        return self._activity_size

    @activity_size.setter
    def activity_size(self, a):
        self._activity_size = a

    def add_read(self, read_desc: ReadDescriptor) -> None:
        self.reads.append(read_desc)

    def add_read_encodings(self):
        for read_desc in self.reads:
            array, allele = create_read_encoding(
                read_desc,
                self.reference,
                feature_length=self.feature_length,
                variant_range=(self.variant_start, self.variant_start + self.activity_size),
            )
            if allele not in self.encodings:
                self.encodings[allele] = list()
            self.encodings[allele].append(array)

    def run_test(self):
        reads, names, qualities, cigartuples, ref_starts, mapq, orientation, pacbio_flags, hp = list(
            list(x) for x in zip(*self.reads)
        )

        hp = list(map(lambda x: 0 if x is None else x, hp))
        names = [str(x) for x in names]

        aligner = libCallability.AlleleSearcherLite(
            reads,
            names,
            qualities,
            cigartuples,
            ref_starts,
            mapq,
            orientation,
            pacbio_flags,
            hp,
            self.reference,
            0,
            0,
            len(self.reference),
            False
        )
        aligner.minMapQ = 5
        aligner.qThreshold = 10
        aligner.max_reassembly_region_size = 0
        aligner.indelThreshold = 0
        aligner.snvThreshold = 0
        aligner.minCount = 0
        aligner.initialize()

        aligner.determineDifferingRegions(True)
        differing_regions = aligner.differingRegions
        region = [(item.first, item.second) for item in differing_regions][0]
        self.variant_start = region[0]
        self.activity_size = region[1] - region[0]
        self.add_read_encodings()

        aligner.assemble_alleles_from_reads(False)
        aligner.assemble(self.variant_start, self.variant_start + self.activity_size)

        logging.info("Comparing alignment results")

        for allele in self.encodings:
            self_encoding = self.encodings[allele]
            self_encoding = np.stack([s.T for s in self_encoding], axis=0)
            computed_encodings = aligner.computeFeaturesColoredSimple(
                allele, self.feature_length, False, self_encoding.shape[2] == 7,
            )
            assert(np.add.reduce((self_encoding - computed_encodings).flatten() ** 2) == 0)
            print("Allele: %s" % allele)
            print(self_encoding)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")
    """
    Locations:            10   14
    Differing:            | ||||
    Reference: "ACGATACCGTA-CGGATCGGATCGT"
    Read0:               TA----ATCG
    Read1:               TAACGGATCG
    Read2:               TG-CGGATCG
    """

    logging.info("#######################")
    logging.info("Running tagless test")
    logging.info("#######################")

    reference = "ACGATACCGTACGGATCGGATCGT"
    read0 = ReadDescriptor(
        read="TAATCG",
        name=0,
        quality=[26] * 6,
        cigartuples=[[BAM_CMATCH, 2], [BAM_CDEL, 3], [BAM_CMATCH, 4]],
        reference_start=9,
        mapq=30,
        orientation=-1,
        pacbio=False,
        hp=None,
    )
    read1 = ReadDescriptor(
        read="TAACGGATCG",
        name=1,
        quality=[30] * 10,
        cigartuples=[[BAM_CMATCH, 2], [BAM_CINS, 1], [BAM_CMATCH, 7]],
        reference_start=9,
        mapq=44,
        orientation=1,
        pacbio=False,
        hp=None,
    )
    read2 = ReadDescriptor(
        read="TGCGGATCG",
        name=2,
        quality=[15] * 9,
        cigartuples=[[BAM_CMATCH, 9]],
        reference_start=9,
        mapq=75,
        orientation=1,
        pacbio=False,
        hp=None,
    )

    testcase = TestCase(reference, feature_length=10)

    testcase.add_read(read0) 
    testcase.add_read(read1) 
    testcase.add_read(read2) 

    testcase.run_test()

    logging.info("Test passed")

    logging.info("#######################")
    logging.info("Running tagged test")
    logging.info("#######################")

    reference = "ACGATACCGTACGGATCGGATCGT"
    read0 = ReadDescriptor(
        read="TAATCG",
        name=0,
        quality=[26] * 6,
        cigartuples=[[BAM_CMATCH, 2], [BAM_CDEL, 3], [BAM_CMATCH, 4]],
        reference_start=9,
        mapq=30,
        orientation=-1,
        pacbio=False,
        hp=1,
    )
    read1 = ReadDescriptor(
        read="TAACGGATCG",
        name=1,
        quality=[30] * 10,
        cigartuples=[[BAM_CMATCH, 2], [BAM_CINS, 1], [BAM_CMATCH, 7]],
        reference_start=9,
        mapq=44,
        orientation=1,
        pacbio=False,
        hp=0,
    )
    read2 = ReadDescriptor(
        read="TGCGGATCG",
        name=2,
        quality=[15] * 9,
        cigartuples=[[BAM_CMATCH, 9]],
        reference_start=9,
        mapq=75,
        orientation=1,
        pacbio=False,
        hp=2,
    )

    testcase = TestCase(reference, feature_length=10)

    testcase.add_read(read0) 
    testcase.add_read(read1) 
    testcase.add_read(read2) 

    testcase.run_test()

    logging.info("Test passed")