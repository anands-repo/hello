# © 2019 University of Illinois Board of Trustees.  All rights reserved
import pysam


def get_reference_prefixes(ref_name):
    with pysam.FastaFile(ref_name) as fhandle:
        references = fhandle.references

        no_prefix_chromosomes = [str(i) for i in range(1, 23)]

        if len(set(no_prefix_chromosomes).difference(set(references))) == 0:
            return ""
        elif len(set("chr" + i for i in no_prefix_chromosomes).difference(set(references))) == 0:
            return "chr"
        else:
            raise ValueError("Unknown chromosomal names, explicitly specify chromosomes")
