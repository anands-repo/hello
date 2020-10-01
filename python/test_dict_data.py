import h5py
import MemmapDataLite
import _pickle as pickle
import sys
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)


def compare_tensors(a, b):
    a = np.array(a)
    b = np.array(b)

    logging.info("Comparing tensors of sizes %s, %s" % (str(a.shape), str(b.shape)))

    assert(a.shape == b.shape)
    assert(np.sum(np.abs(a - b).flatten()) < 1e-8)


dict_data = pickle.load(open(sys.argv[1], 'rb'))
hdf5_data = h5py.File(sys.argv[2], 'r')

attributes = [
    'label', 'feature', 'feature2', 'supportingReadsStrict', 'supportingReadsStrict2', 'segment'
]

for location in dict_data.locations:
    site_a = dict_data[location]
    site_b = hdf5_data[location]
    logging.info("Alleles at site %s are %s" % (location, str(site_a.keys())))

    for allele in site_a:
        logging.info("Available attributes for allele %s are %s" %(allele, str(site_a[allele].keys())))
        for attribute in attributes:
            logging.info("Comparing %s for allele %s at location %s" % (attribute, allele, location))
            compare_tensors(
                np.array(site_a[allele][attribute]),
                np.array(site_b[allele][attribute])
            )

logging.info("Ensuring that all locations are covered")
assert(set(dict_data.locations) == set(hdf5_data.keys()))

print("Passed")
