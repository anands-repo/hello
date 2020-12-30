# © 2019 University of Illinois Board of Trustees.  All rights reserved
import ast
import argparse
import os
import logging
from functools import reduce
import math


def measureDistance(pointA, pointB):
    """
    Determines the distance pointB - pointA

    :param pointA: dict
        Point A

    :param pointB: dict
        Point B

    :return: int
        Distance
    """
    if (pointA['chromosome'] != pointB['chromosome']):
        distance = float('inf');

    if ('position' in pointA) and ('position' in pointB):
        return pointB['position'] - pointA['position'];
    elif ('start' in pointB) and ('stop' in pointA):
        return pointB['start'] - pointA['stop'] + 1;
    else:
        raise ValueError("Bad arguments " + str(pointA) + ", " + str(pointB));


def hotspots_processor(hotspots):
    """
    Generator for clustering adjacent hotspots

    :param hotspots: str
        File containing hotspots
    """
    with open(hotspots, 'r') as fhandle:
        cluster = []

        for line in fhandle:
            point = ast.literal_eval(line)

            if len(cluster) == 0:
                cluster.append(point)
            else:
                if measureDistance(cluster[-1], point) == 1:
                    cluster.append(point)
                else:
                    yield cluster
                    cluster = [point]

        if len(cluster) > 0:
            yield cluster


def count_hotspots(hotspots):
    """
    Count the number of non-adjacent hotspots

    :param hotspots: str
        Hotspot filename

    :return: int
        Number of hotspots
    """
    count = 0

    for _ in hotspots_processor(hotspots):
        count += 1

    return count


def cluster_hotspots(hotspots, min_separation, min_items_per_cluster):
    """
    Cluster hotspot regions

    :param hotspots: str
        Hotspots filename

    :param min_separation: int
        Minimum separation between non-adjacent hotspots

    :param min_items_per_cluster: int
        Minimum number of non-adjacent hotspots
    """
    hgen = hotspots_processor(hotspots)

    cluster = []

    for i, item in enumerate(hgen):
        if len(cluster) < min_items_per_cluster or measureDistance(cluster[-1][-1], item[0]) < min_separation:
            cluster.append(item)
        else:
            flattened_cluster = reduce(lambda a, b: a + b, cluster)
            yield flattened_cluster
            cluster = [item]
    
    if len(cluster) > 0:
        flattened_cluster = reduce(lambda a, b: a + b, cluster)
        yield flattened_cluster
        cluster = []


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shard a set of hotspots for parallel runs");

    parser.add_argument(
        "--hotspots",
        help="File containing all hotspots of interest",
        required=True,
    );

    parser.add_argument(
        "--minSeparation",
        help="Minimum separation between two hotspots for them to belong to two files",
        default=25,
    );

    parser.add_argument(
        "--maxShards",
        help="Maximum number of shards to allow",
        default=500,
    )

    parser.add_argument(
        "--outputPrefix",
        help="Prefix of output files",
        required=True,
    );

    args = parser.parse_args();

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(message)s');

    dirpath, _ = os.path.split(os.path.abspath(args.outputPrefix));

    if os.path.exists(dirpath):
        assert(os.path.isdir(dirpath)), "Invalid path";
        logging.info("Directory %s exists" % dirpath);
    else:
        os.makedirs(dirpath);
        logging.info("Creating directory %s" % dirpath);

    logging.info("Counting hotspots")
    num_hotspots = count_hotspots(args.hotspots)
    min_items_per_cluster = math.ceil(num_hotspots / args.maxShards)
    logging.info("Sharding with >= %d items per cluster" % min_items_per_cluster)
    shard_gen = cluster_hotspots(
        args.hotspots, min_separation=args.minSeparation, min_items_per_cluster=min_items_per_cluster
    )
    counter = 0

    def dumpToFile(cluster_):
        if len(cluster_) == 0:
            return;

        with open(args.outputPrefix + "%d.txt" % counter, 'w') as whandle:
            for line in cluster_:
                whandle.write(str(line) + "\n");

    for cluster in shard_gen:
        dumpToFile(cluster)
        counter += 1
