import ast
import argparse
import os
from trainer import measureDistance
import logging

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
        "--minEntriesPerShard",
        help="Minimum number of entries within a shard",
        default=256,
        type=int,
    );

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

    with open(args.hotspots, 'r') as fhandle:
        cluster = [];
        counter = 0;

        def dumpToFile(cluster_):
            if len(cluster_) == 0:
                return;

            with open(args.outputPrefix + "%d.txt" % counter, 'w') as whandle:
                for line in cluster_:
                    whandle.write(str(line) + "\n");

        for line in fhandle:
            items = ast.literal_eval(line);

            if len(cluster) < args.minEntriesPerShard:
                cluster.append(items);
            else:
                if measureDistance(cluster[-1], items) > args.minSeparation:
                    dumpToFile(cluster);
                    counter += 1;
                    cluster = [items];
                else:
                    cluster.append(items);

        dumpToFile(cluster);
