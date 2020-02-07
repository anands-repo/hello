import h5py
import argparse
import math
import logging
from multiprocessing import Pool


def determineMaxQParallelWrapper(fhandle, keys, threads=0, batchSize=10000):
    if threads <= 0:
        mapper = map;
    else:
        workers = Pool(threads);
        mapper = workers.imap_unordered;

    numBatches = len(keys) // batchSize;

    if numBatches * batchSize < len(keys):
        numBatches += 1;

    batches = [];

    for i in range(numBatches):
        start = i * batchSize;
        stop = min((i + 1) * batchSize, len(keys));
        batch = (fhandle, keys[start: stop], False);
        batches.append(batch);

    total = 0;

    for i, value in enumerate(mapper(determineMaxQWrapper, batches)):
        total += value;
        logging.info("Completed %d batches" % i);

    logging.info("Maximum theoretical Q for the dataset is %f" % (total / len(keys)));

    return total / len(keys);


def determineMaxQWrapper(args):
    return determineMaxQ(*args);


def determineMaxQ(dataset, indices, returnAvg=True):
    """
    Determine the maximum possible Q value for a set of data
    """
    logging.info("Determining theoretical maximum Q");
    totalQScore = 0;
    totalSites = 0;

    for i, key in enumerate(indices):
        site = dataset[key];
        if site['siteLabel'][0] == 0:
            totalQScore += 0;  # Can predict site at 100% accuracy

        elif site['siteLabel'][0] == 1:
            totalQScore += -math.log(0.5);

        else:
            raise ValueError("Unknown site type %s for location %s" % (str(site['siteLabel'][0]), str(key)));

        if (i + 1) % 10000 == 0:
            logging.info("Completed %d sites" % (i + 1));

        totalSites += 1;

    if returnAvg:
        return totalQScore / totalSites;
    else:
        return totalQScore;


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Determine max possible QLoss");

    parser.add_argument("--hdf5", help="HDF5 file containing training examples", required=True);
    args = parser.parse_args();

    logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(message)s');

    determineMaxQ(args.hdf5, fhandle.keys());
