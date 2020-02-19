import sys
import MemmapData
import pickle
import glob
import os

path = sys.argv[1];

allFiles = glob.glob(path + "/*.index");

print("Found %d files" % len(allFiles));


def fixHdf5Name(obj):
    if not hasattr(obj, 'hdf5'):
        return;

    hdf5name = os.path.split(obj.hdf5)[1];
    newname = os.path.join(path, hdf5name);
    print("Renaming %s to %s" % (obj.hdf5, newname));
    obj.hdf5 = newname;


def fixMemmapperName(obj):
    if not hasattr(obj, 'storageName'):
        return;

    memmapname = os.path.split(obj.storageName)[1];
    newname = os.path.join(path, memmapname);
    print("Renaming %s to %s" % (obj.storageName, newname));
    obj.storageName = newname;


for filename in allFiles:
    data = pickle.load(open(filename, 'rb'));
    fixHdf5Name(data);

    for dset in data.datasets.values():
        fixHdf5Name(dset);
        fixMemmapperName(dset);

    with open(filename, 'wb') as fhandle:
        pickle.dump(data, fhandle);
