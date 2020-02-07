import numpy
import h5py
import _pickle as pickle
import torch.utils.data
import logging
import sys
import numpy as np
import ast
import os
import ctypes
from functools import reduce
import random

try:
    madvise = ctypes.CDLL("libc.so.6").madvise;
    madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int];
    madvise.restype = ctypes.c_int;
except Exception:
    def madvise(*args, **kwargs):
        pass;


def message(i, limit=10000, string=""):
    if ((i + 1) % limit) == 0:
        logging.info("Completed%s %d sites" % (string, i + 1));


class Memmapper(torch.utils.data.Dataset):
    def __init__(self, hdf5, memmapPrefix, key='feature'):
        """
        Given an hdf5 file of features, this class converts that to
        memmap arrays on disk. It then pickles itself on disk. The object
        may be unpickled to obtain a seamless access to the memmap array.
        """
        self.accessKey = key;
        self.numReadsPerAllele = dict();
        self.indicesOfLocation = dict();
        self.locations = list();

        logging.info("Indexing hdf5 file ... ");

        totalNumReads = 0;
        totalNumLabels = 0;
        shape = None;

        with h5py.File(hdf5, 'r') as fhandle:
            for i, location in enumerate(fhandle.keys()):
                keys = fhandle[location].keys();
                numReadsPerAllele = [];

                # Indices within the memmap files where a location starts
                self.indicesOfLocation[location] = (totalNumReads, totalNumLabels);

                for key in keys:
                    if key == 'siteLabel':
                        continue;

                    totalNumLabels += 1;

                    numReadsPerAllele.append(
                        fhandle[location][key][self.accessKey].shape[0]
                    );

                    if shape is None:
                        shape = fhandle[location][key][self.accessKey].shape;

                totalNumReads += sum(numReadsPerAllele);
                self.numReadsPerAllele[location] = numReadsPerAllele;
                self.locations.append(location);
                message(i, 100);

        if shape is None:
            logging.info("Completed index, no data found in hdf5 file. Not building memory map");
            return;

        logging.info("Completed index... building memory map");

        self.features = os.path.abspath(memmapPrefix + ".features.memmap");
        self.labels = os.path.abspath(memmapPrefix + ".labels.memmap");
        self.__features_file = None;
        self.__labels_file = None;
        self.featureShape = (totalNumReads, shape[1], shape[2]);
        self.labelShape = (totalNumLabels, );

        features = np.memmap(
            self.features,
            shape=self.featureShape,
            dtype=np.float32,
            mode='w+',
        );

        labels = np.memmap(
            self.labels,
            shape=self.labelShape,
            dtype=np.float32,
            mode='w+',
        );

        logging.info("Completed building memory maps, copying over data ... ");

        with h5py.File(hdf5, 'r') as fhandle:
            for i, location in enumerate(self.locations):
                findex, lindex = self.indicesOfLocation[location];

                alleleCounter = 0;
                readCounter = 0;

                for key in fhandle[location].keys():
                    if key == 'siteLabel':
                        continue;

                    feature = fhandle[location][key][self.accessKey][()];
                    assert(
                        self.numReadsPerAllele[location][alleleCounter] == feature.shape[0]
                    ), "Mismatch between indexing and copying at site %s" % location;
                    label = fhandle[location][key]['label'][0];

                    features[findex + readCounter: findex + readCounter + feature.shape[0]] = feature;
                    labels[lindex + alleleCounter] = label;

                    alleleCounter += 1;
                    readCounter += feature.shape[0];

                message(i, 100);

        del features;
        del labels;

        self.totalNumReads = totalNumReads;
        self.totalNumLabels = totalNumLabels;
        logging.info("Pickling index as %s" % (memmapPrefix + ".index"));
        pickle.dump(self, open(memmapPrefix + ".index", 'wb'));

    @property
    def featureFile(self):
        if self.__features_file is None:
            self.__features_file = np.memmap(
                self.features,
                shape=self.featureShape,
                dtype=np.float32,
                mode='r',
            );

        return self.__features_file;

    @property
    def labelFile(self):
        if self.__labels_file is None:
            self.__labels_file = np.memmap(
                self.labels,
                shape=self.labelShape,
                dtype=np.float32,
                mode='r',
            );
        return self.__labels_file;

    def __getitem__(self, index):
        location = self.locations[index];
        findex, lindex = self.indicesOfLocation[location];
        numReadsPerAllele = self.numReadsPerAllele[location];

        tensors = [];
        labels = [];

        readCounter = 0;
        alleleCounter = 0;

        for n in numReadsPerAllele:
            tensors.append(self.featureFile[findex + readCounter: findex + readCounter + n]);
            labels.append(self.labelFile[lindex + alleleCounter]);
            readCounter += n;
            alleleCounter += 1;

        return {'tensors': tensors, 'labels': labels};

    def __len__(self):
        return len(self.locations);

    def merge(self, others, newPrefix):
        # Create a large memmap object that contains all items
        totalNumReadsInAll = sum([o.totalNumReads for o in others]) + self.totalNumReads;
        totalNumLabelsInAll = sum([o.totalNumLabels for o in others]) + self.totalNumLabels;
        featureShape = (totalNumReadsInAll, self.featureShape[1], self.featureShape[2]);
        labelShape = (totalNumLabelsInAll, );
        featuresName = newPrefix + ".features.memmap";
        labelsName = newPrefix + ".labels.memmap";

        featureFile = np.memmap(
            featuresName,
            shape=featureShape,
            dtype=np.float32,
            mode='w+',
        );

        labelFile = np.memmap(
            labelsName,
            shape=labelShape,
            dtype=np.float32,
            mode='w+',
        );

        # Initialize using self
        logging.info("Initializing merge with self");
        featureFile[: self.featureFile.shape[0]] = self.featureFile[:];
        labelFile[: self.labelFile.shape[0]] = self.labelFile[:];
        numReadsPerAllele = dict(self.numReadsPerAllele);
        indicesOfLocation = dict(self.indicesOfLocation);
        locations = list(self.locations);
        totalNumReads = self.totalNumReads;
        totalNumLabels = self.totalNumLabels;
        logging.info("Merging remaining files");

        for i, other in enumerate(others):
            featureFile[totalNumReads: totalNumReads + other.totalNumReads] = other.featureFile[:];
            labelFile[totalNumLabels: totalNumLabels + other.totalNumLabels] = other.labelFile[:];
            locations += other.locations;
            numReadsPerAllele.update(other.numReadsPerAllele);
            for item in other.indicesOfLocation:
                f, l = other.indicesOfLocation[item];
                indicesOfLocation[item] = (f + totalNumReads, l + totalNumLabels);
            totalNumReads += other.totalNumReads;
            totalNumLabels += other.totalNumLabels;
            message(i, 1000);

        self.__features_file = None;
        self.__labels_file = None;
        assert(totalNumReads == totalNumReadsInAll);
        assert(totalNumLabels == totalNumLabelsInAll);
        self.featureShape = featureShape;
        self.labelShape = labelShape;
        self.features = featuresName;
        self.labels = labelsName;
        self.locations = locations;
        self.numReadsPerAllele = numReadsPerAllele;
        self.indicesOfLocation = indicesOfLocation;
        self.totalNumReads = totalNumReadsInAll;
        self.totalNumLabels = totalNumLabelsInAll;
        del featureFile;
        del labelFile;
        logging.info("Completed merging. Saving new index file");
        pickle.dump(self, open(newPrefix + ".index", "wb"));


class MemmapperSingleNpy(torch.utils.data.Dataset):
    def __init__(
        self,
        hdf5,
        npyPrefix,
        key='feature',
        concatenate=False,
        pickle_=True,
        mode='separate',
        indexingMode='numeric',
        readOnly=True,
        dtype='float32',
    ):
        """
        Memmapper for converting a single hdf5 file key into
        npy files. It deletes the corresponding key from the original
        hdf5 files if necessary (mode='integrated).

        :param hdf5: str
            HDF5 file name

        :param npyPrefix: str
            Prefix of the storage

        :param key: str
            The specific key that needs to be memmapped

        :param concatenate: str
            Whether the particular tensors being recovered
            should be concatenated

        :param pickle_: bool
            Whether the data-structure needs to be pickled on disk

        :param mode: str
            ['separate', or 'integrated']. In integrated mode, the
            HDF5 file's key is going to be erased; then this object
            is going to be an integrated interface to access everything

        :param indexingMode: str
            'numeric'|'string'; In 'numeric', the specific location information
            isn't used to index. In 'string' mode, location keys are used to
            index into the file

        :param readOnly: bool
            Whether this particular object can modify its underlying datasets
            outside of the __init__ function

        :param dtype: str
            FOR COMPATIBILITY. dtype is unnecessary as np.save keeps data types
            The type of data to be stored. Should work with getattr(numpy, dtype)
        """
        self.accessKey = key;
        self.numItemsPerAllele = dict();
        self.indicesOfLocation = dict();
        self.locations = list();
        self.concatenate = concatenate;
        self.hdf5 = hdf5;
        self.mode = mode;
        self.indexingMode = indexingMode;
        self.readOnly = readOnly;
        self.closeAfterAccess = False;

        # Create a directory for this item
        if not os.path.exists(npyPrefix):
            os.makedirs(npyPrefix);

        with h5py.File(hdf5, 'r') as fhandle:
            for location in fhandle.keys():
                self.locations.append(location);
                filename = os.path.join(npyPrefix, "_%s_%s" % (location, self.accessKey));
                tensors = [];
                numTensorsPerAllele = [];

                for allele in fhandle[location].keys():
                    if allele == 'siteLabel':
                        continue;

                    tensors.append(np.array(fhandle[location][allele][self.accessKey]));
                    numTensorsPerAllele.append(tensors[-1].shape[0]);

                self.indicesOfLocation[location] = filename;
                self.numItemsPerAllele[location] = numTensorsPerAllele;
                tensors = np.concatenate(tensors, axis=0);
                np.save(filename, tensors);

        if pickle_:
            selfname = npyPrefix + ".index";
            logging.info("Pickling index as %s" % selfname);
            pickle.dump(self, open(selfname, 'wb'));

        if mode == 'integrated':
            # In integrated mode, delete the key from the given hdf5 file
            # to prevent duplication and save space
            with h5py.File(hdf5, 'r+') as fhandle:
                for i, location in enumerate(fhandle.keys()):
                    value = fhandle[location];
                    for key in value.keys():
                        if key == 'siteLabel':
                            continue;

                        # Delete the item stored into memmap from the file
                        del value[key][self.accessKey];

    def advise(self):
        pass;

    def delStorageFile(self):
        pass;

    def __len__(self):
        return len(self.locations);

    def __getitem__(self, index):
        if self.indexingMode == 'numeric':
            location = self.locations[index];
        else:
            location = index;

        filename = self.indicesOfLocation[location] + ".npy";
        data = np.load(filename);
        numItemsPerAllele = self.numItemsPerAllele[location];

        counter = 0;
        tensors = [];

        for n in numItemsPerAllele:
            tensors.append(data[counter: counter + n]);
            counter += n;

        if self.indexingMode == 'numeric':
            if self.concatenate:
                return np.concatenate(tensors, axis=0);
            else:
                return tensors;
        else:
            # Provide an interface similar to that of a dictionary
            with h5py.File(self.hdf5, 'r') as fhandle:
                returns = dict();
                alleles = [k for k in fhandle[location].keys() if k != 'siteLabel'];

                for a, t in zip(alleles, tensors):
                    if a not in returns:
                        returns[a] = dict();
                    returns[a][self.accessKey] = t;

                return returns;


class MemmapperSingle(torch.utils.data.Dataset):
    def __init__(
        self,
        hdf5,
        memmapPrefix,
        key='feature',
        concatenate=False,
        pickle_=True,
        mode='separate',
        indexingMode='numeric',
        readOnly=True,
        dtype='float32',
    ):
        """
        Memmapper for converting a single hdf5 file key into
        a memmap file. It deletes the corresponding key from the original
        hdf5 files if necessary (mode='integrated).

        :param hdf5: str
            HDF5 file name

        :param memmapPrefix: str
            Prefix of the memory mapped file into which
            to store data

        :param key: str
            The specific key that needs to be memmapped

        :param concatenate: str
            Whether the particular tensors being recovered
            should be concatenated

        :param pickle_: bool
            Whether the data-structure needs to be pickled on disk

        :param mode: str
            ['separate', or 'integrated']. In integrated mode, the
            HDF5 file's key is going to be erased; then this object
            is going to be an integrated interface to access everything

        :param indexingMode: str
            'numeric'|'string'; In 'numeric', the specific location information
            isn't used to index. In 'string' mode, location keys are used to
            index into the file

        :param readOnly: bool
            Whether this particular object can modify its underlying datasets
            outside of the __init__ function

        :param dtype: str
            The type of data to be stored. Should work with getattr(numpy, dtype)
        """
        self.accessKey = key;
        self.numItemsPerAllele = dict();
        self.indicesOfLocation = dict();
        self.locations = list();
        self.concatenate = concatenate;
        self.hdf5 = hdf5;
        self.mode = mode;
        self.indexingMode = indexingMode;
        self.readOnly = readOnly;
        self.closeAfterAccess = False;

        logging.info("Indexing hdf5 file ... ");

        totalNumItems = 0;
        shape = None;

        with h5py.File(hdf5, 'r') as fhandle:
            for i, location in enumerate(fhandle.keys()):
                keys = fhandle[location].keys();
                numItemsPerSingleAllele = [];

                # Indices within the memmap files where a location starts
                self.indicesOfLocation[location] = totalNumItems;

                for key in keys:
                    if key == 'siteLabel':
                        continue;

                    numItemsPerSingleAllele.append(
                        fhandle[location][key][self.accessKey].shape[0],
                    );

                    if shape is None:
                        shape = fhandle[location][key][self.accessKey].shape;

                totalNumItems += sum(numItemsPerSingleAllele);
                self.numItemsPerAllele[location] = numItemsPerSingleAllele;
                self.locations.append(location);
                message(i, 100, " (indexing for %s)" % self.accessKey);

        if shape is None:
            logging.info("Completed index, no data found in hdf5 file. Not building memory map");
            return;

        logging.info("Completed index... building memory map");

        self.storageName = memmapPrefix + ".%s.memmap" % self.accessKey;
        self.__storage_file = None;
        self.storageShape = tuple([totalNumItems] + list(shape[1:]));

        # This hoop jumping is for backward compatibility
        if dtype != 'float32':
            self.dtype = getattr(np, dtype);
            dtype = self.dtype;
        else:
            dtype = np.float32;

        storage = np.memmap(
            self.storageName,
            shape=self.storageShape,
            dtype=dtype,
            mode='w+',
        );

        logging.info("Completed building memory maps, copying over data ... ");

        with h5py.File(hdf5, 'r') as fhandle:
            for i, location in enumerate(self.locations):
                findex = self.indicesOfLocation[location];

                alleleCounter = 0;
                itemCounter = 0;

                for key in fhandle[location].keys():
                    if key == 'siteLabel':
                        continue;

                    item = np.array(fhandle[location][key][self.accessKey]);
                    assert(
                        self.numItemsPerAllele[location][alleleCounter] == item.shape[0]
                    ), "Mismatch between indexing and copying at site %s" % location;

                    storage[findex + itemCounter: findex + itemCounter + item.shape[0]] = item;

                    alleleCounter += 1;
                    itemCounter += item.shape[0];

                message(i, 100, "(storing for %s)" % self.accessKey);

        del storage;

        self.totalNumItems = totalNumItems;

        if pickle_:
            logging.info("Pickling index as %s" % (memmapPrefix + ".index"));
            pickle.dump(self, open(memmapPrefix + ".index", 'wb'));

        if mode == 'integrated':
            # In integrated mode, delete the key from the given hdf5 file
            # to prevent duplication and save space
            with h5py.File(hdf5, 'r+') as fhandle:
                for i, location in enumerate(fhandle.keys()):
                    value = fhandle[location];
                    for key in value.keys():
                        if key == 'siteLabel':
                            continue;

                        # Delete the item stored into memmap from the file
                        del value[key][self.accessKey];


    @property
    def keys(self):
        if self.indexingMode == 'string':
            return iter(self.locations);
        else:
            return range(len(self.locations));

    @property
    def bytesPerItem(self):
        if not hasattr(self, '_bytesPerItem'):
            if len(self.storageShape) == 1:
                bytesPerItem = self.storageFile.dtype.itemsize;
            else:
                bytesPerItem = reduce(lambda x, y: x * y, self.storageShape[1:], 1) * self.storageFile.dtype.itemsize;

            self._bytesPerItem = bytesPerItem;

        return self._bytesPerItem;

    def advise(self):
        self._advise = True;

    @property
    def storageFile(self):
        if self.__storage_file is None:
            # For backward compatibility
            if hasattr(self, 'dtype'):
                dtype = self.dtype;
            else:
                dtype = np.float32;

            self.__storage_file = np.memmap(
                self.storageName,
                shape=self.storageShape,
                dtype=dtype,
                mode='r',
            );

            # When using iterable data, this is the way to go
            if hasattr(self, '_readIntoMem') and self._readIntoMem:
                self.__storage_file = np.array(self.__storage_file);

        return self.__storage_file;

    def delStorageFile(self):
        if self.__storage_file is not None:
            del self.__storage_file;

        self.__storage_file = None;

    def __getitem__(self, index):
        if hasattr(self, '_advise') and self._advise:
            # Advise kernel to fetch everything the first time it is accessed
            madvise(
                self.storageFile.ctypes.data,
                reduce(lambda x, y: x * y, self.storageShape, 1) * self.storageFile.dtype.itemsize,
                3,
            );
            self._advise = False;

        if self.indexingMode == 'numeric':
            location = self.locations[index];
        else:
            location = index;

        findex = self.indicesOfLocation[location];
        numItemsPerAllele = self.numItemsPerAllele[location];

        tensors = [];

        itemCounter = 0;
        alleleCounter = 0;

        for n in numItemsPerAllele:
            # We will explicitly call np.array to copy the data. The memmap file may be closed
            # before the data is returned or used.
            tensors.append(
                np.array(self.storageFile[findex + itemCounter: findex + itemCounter + n])
            );
            itemCounter += n;
            alleleCounter += 1;

        if self.indexingMode == 'numeric':
            if self.concatenate:
                tensors = np.concatenate(tensors, axis=0);

            return tensors;
        else:
            # Provide an interface similar to that of a dictionary
            with h5py.File(self.hdf5, 'r') as fhandle:
                returns = dict();
                alleles = [k for k in fhandle[location].keys() if k != 'siteLabel'];
                for a, t in zip(alleles, tensors):
                    if a not in returns:
                        returns[a] = dict();
                    returns[a][self.accessKey] = t;

                return returns;

    def prefetch(self):
        _ = self.storageFile[:self.storageShape[0]];

    def __len__(self):
        return len(self.locations);


class MemmapperCompound:
    def __init__(
        self,
        hdf5,
        prefix,
        keys,
        concatenations,
        hybrid=False,
        mode='separate',
        indexingMode='numeric',
        dtypes=None,
        memtype='memmap',
    ):
        self.datasets = dict();
        self.hybrid = hybrid;  # This is an internal flag set for DataLoader et al

        if dtypes is None:
            dtypes = {key: 'float32' for key in keys};

        StorageType = MemmapperSingle if memtype == 'memmap' else MemmapperSingleNpy;
        self.memtype = memtype;

        for key, concatenation in zip(keys, concatenations):
            self.datasets[key] = StorageType(
                hdf5,
                prefix,
                key=key,
                concatenate=concatenation,
                pickle_=False,
                mode=mode,
                indexingMode=indexingMode,
                dtype=dtypes[key],
            );

        self.hdf5 = hdf5;
        self.keys = keys;
        self.indexingMode = indexingMode;
        logging.info("Pickling data ... ");
        pickle.dump(self, open(prefix + ".index", "wb"));

    def __len__(self):
        return len(self.datasets[self.keys[0]]);

    def pruneHomozygousIndices(self, keep=0, onlySNVs=False):
        newLocations = list();

        with h5py.File(self.hdf5, 'r') as fhandle:
            for l in self.locations:
                data = fhandle[l];
                alleles = [k for k in data.keys() if k != 'siteLabel'];
                isIndelSite = any(len(a) != 1 for a in alleles);
                keepIt = random.uniform(0, 1) <= keep;
                if (onlySNVs and isIndelSite) or (len(alleles) > 1) or keepIt:
                    newLocations.append(l);

        for dset in self.datasets.values():
            dset.locations = newLocations;

    def __getitem__(self, index):
        if self.indexingMode == 'numeric':
            returns = dict();

            for key in self.keys:
                returns[key] = self.datasets[key][index];

            return returns;
        else:
            returns = dict();

            with h5py.File(self.hdf5, 'r') as fhandle:
                site = fhandle[index];

                for key in site.keys():
                    if key == 'siteLabel':
                        returns[key] = site[key][:];
                        continue;

                    group = site[key];
                    returns[key] = {k: group[k][:] for k in group.keys()};

            for _, dset in self.datasets.items():
                r = dset[index];
                for key in r:
                    # Here, each key is an allele containing a dictionary
                    returns[key].update(r[key]);

            return returns;

    def prefetch(self):
        for d in self.datasets.values():
            d.prefetch();

    @property
    def locations(self):
        return iter(self.datasets[self.keys[0]].locations);

    def setIndexingMode(self, mode):
        self.indexingMode = mode;
        for dset in self.datasets.values():
            dset.indexingMode = mode;

    def setPurgeMemmapFileAfterAccess(self, _value):
        for dset in self.datasets.values():
            dset.closeAfterAccess = _value;

    def addField(self, location, allele, key, value):
        value = np.array(value);
        with h5py.File(self.hdf5, 'r+') as fhandle:
            site = fhandle[location];
            alleleGroup = site[allele];
            alleleGroup.create_dataset(key, shape=value.shape, dtype=value.dtype);
            alleleGroup[key][:] = value[:];

    def purgeFiles(self):
        for dset in self.datasets.values():
            dset.delStorageFile();

    def advise(self):
        for dset in self.datasets.values():
            dset.advise();

    @property
    def loadIntoMem(self):
        if hasattr(self, '_loadIntoMem'):
            return self._loadIntoMem;
        else:
            return False;

    @loadIntoMem.setter
    def loadIntoMem(self, _value):
        self._loadIntoMem = _value;
        for dset in self.datasets.values():
            dset._advise = _value;

    def countFrequency(self):
        frequency = {'indels': np.array([0, 0]), 'snv': np.array([0, 0])};

        with h5py.File(self.hdf5, 'r') as fhandle:
            for location in self.locations:
                data = fhandle[location];
                siteLabel = data['siteLabel'][0];
                alleles = [k for k in list(data.keys()) if k != 'siteLabel'];
                vtype = 'indels' if any(len(a) != 1 for a in alleles) else 'snv';

                if siteLabel == 0:
                    frequency[vtype][1] += 1;
                    frequency[vtype][0] += len(alleles) - 1;
                    assert(len(alleles) >= 1);
                else:
                    frequency[vtype][1] += 2;
                    frequency[vtype][0] += len(alleles) - 2;
                    assert(len(alleles) >= 2);

        return frequency;


def copy(inhandle, outhandle, location):
    """
    Copies items from one hdf5 file into another

    :param inhandle: h5py.File
        Input file handle

    :param outhandle: h5py.File
        Output file handle

    :param location: str
        Location key
    """
    outgroup = outhandle.create_group(location);
    ingroup = inhandle[location];

    for allele in ingroup.keys():
        if allele == 'siteLabel':
            dset = outgroup.create_dataset('siteLabel', dtype=np.int32, shape=(1, ));
            dset[:] = ingroup['siteLabel'][:];
            continue;

        alleleInGroup = ingroup[allele];
        alleleOutGroup = outgroup.create_group(allele);

        for attribute in alleleInGroup.keys():
            dset = alleleOutGroup.create_dataset(
                attribute, dtype=alleleInGroup[attribute].dtype, shape=alleleInGroup[attribute].shape
            );
            dset[:] = alleleInGroup[attribute][:];


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(message)s');

    if (len(sys.argv) < 6) or (len(sys.argv[1]) == "help"):
        logging.error("Found %s arguments" % str(sys.argv));
        logging.info("Usage: python MemmapData.py <hdf5 filename> <memmap prefix> <dictionary of key:concatenation> <hybrid|single> <integrated|separate>");
        # logging.info("Usage 2: python MemmapData.py merge <list of index filenames>");
        sys.exit(0);

    hdf5 = sys.argv[1];
    memmap = sys.argv[2];
    keyDict = ast.literal_eval(sys.argv[3]);
    keys, concatenations = tuple(zip(*list(keyDict.items())));
    hybrid = sys.argv[4] == 'hybrid';
    _ = MemmapperCompound(hdf5, memmap, keys, concatenations, hybrid, sys.argv[5]);

    # In integrated mode, overwrite the given hdf5 file to free space
    if sys.argv[5] == "integrated":
        logging.info("Releasing space ... will create a temporary file");
        orig_ = hdf5;
        temp_ = hdf5 + ".tmp.hdf5";
        inhandle = h5py.File(orig_, 'r');
        outhandle = h5py.File(temp_, 'w');
        for location in inhandle.keys():
            copy(inhandle, outhandle, location);
        inhandle.close();
        outhandle.close();
        os.rename(temp_, orig_);
