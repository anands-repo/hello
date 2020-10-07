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
from collections import defaultdict

try:
    madvise = ctypes.CDLL("libc.so.6").madvise
    madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    madvise.restype = ctypes.c_int
except Exception:
    def madvise(*args, **kwargs):
        pass


def message(i, limit=10000, string=""):
    if ((i + 1) % limit) == 0:
        logging.info("Completed%s %d sites" % (string, i + 1))


class MemmapperSingle(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dict,
        memmapPrefix,
        key='feature',
        concatenate=False,
        pickle_=True,
        dtype='float32',
    ):
        """
        Memmapper for converting dictionary type data to one stored on disk

        :param data_dict: dict
            Dictionary holding data in memory

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

        :param dtype: str
            The type of data to be stored. Should work with getattr(numpy, dtype)
        """
        self.accessKey = key
        self.numItemsPerAllele = dict()
        self.indicesOfLocation = dict()
        self.locations = list()
        self.concatenate = concatenate
        self.closeAfterAccess = False
        self.alleles = dict()

        logging.info("Indexing hdf5 file ... ")

        totalNumItems = 0
        shape = None
        fhandle = data_dict

        for i, location in enumerate(fhandle.keys()):
            keys = fhandle[location].keys()
            numItemsPerSingleAllele = []
            self.alleles[location] = list()

            # Indices within the memmap files where a location starts
            self.indicesOfLocation[location] = totalNumItems

            for key in keys:
                if key == 'siteLabel':
                    continue

                numItemsPerSingleAllele.append(
                    fhandle[location][key][self.accessKey].shape[0],
                )

                self.alleles[location].append(key)

                if shape is None:
                    shape = fhandle[location][key][self.accessKey].shape

            totalNumItems += sum(numItemsPerSingleAllele)
            self.numItemsPerAllele[location] = numItemsPerSingleAllele
            self.locations.append(location)
            message(i, 100, " (indexing for %s)" % self.accessKey)

        if shape is None:
            logging.info("Completed index, no data found in hdf5 file. Not building memory map")
            return

        logging.info("Completed index... building memory map")

        self.storageName = memmapPrefix + ".%s.memmap" % self.accessKey
        self.__storage_file = None
        self.storageShape = tuple([totalNumItems] + list(shape[1:]))

        # This hoop jumping is for backward compatibility
        self.dtype = getattr(np, dtype)

        storage = np.memmap(
            self.storageName,
            shape=self.storageShape,
            dtype=self.dtype,
            mode='w+',
        )

        logging.info("Completed building memory maps, copying over data ... ")

        for i, location in enumerate(self.locations):
            findex = self.indicesOfLocation[location]

            alleleCounter = 0
            itemCounter = 0

            for key in fhandle[location].keys():
                if key == 'siteLabel':
                    continue

                item = fhandle[location][key][self.accessKey].copy()
                assert(
                    self.numItemsPerAllele[location][alleleCounter] == item.shape[0]
                ), "Mismatch between indexing and copying at site %s" % location

                storage[findex + itemCounter: findex + itemCounter + item.shape[0]] = item[:]

                alleleCounter += 1
                itemCounter += item.shape[0]

            message(i, 100, "(storing for %s)" % self.accessKey)

        # Flush the storage
        del storage

        self.totalNumItems = totalNumItems

        if pickle_:
            logging.info("Pickling index as %s" % (memmapPrefix + ".index"))
            pickle.dump(self, open(memmapPrefix + ".index", 'wb'))

    @property
    def keys(self):
        return iter(self.locations)

    @property
    def storageFile(self):
        if self.__storage_file is None:
            dtype = self.dtype

            self.__storage_file = np.memmap(
                self.storageName,
                shape=self.storageShape,
                dtype=dtype,
                mode='r',
            )

            # When using iterable data, this is the way to go
            if hasattr(self, '_readIntoMem') and self._readIntoMem:
                self.__storage_file = np.array(self.__storage_file)

        return self.__storage_file

    def __getitem__(self, index):
        location = index
        itemCounter = 0
        results = defaultdict(dict)
        findex = self.indicesOfLocation[location]

        for a, n in zip(self.alleles[location], self.numItemsPerAllele[location]):
            results[a][self.accessKey] = np.array(
                self.storageFile[findex + itemCounter: findex + itemCounter + n]
            )
            itemCounter += n

        return results

    def __len__(self):
        return len(self.locations)


class MemmapperCompound:
    def __init__(
        self,
        data_dict,
        prefix,
        keys,
        concatenations,
        hybrid=False,
        dtypes=None,
        *args,
        **kwargs,
    ):
        self.datasets = dict()
        self.hybrid = hybrid;  # This is an internal flag set for DataLoader et al

        # Store main items into memmappers
        for key, concatenation in zip(keys, concatenations):
            self.datasets[key] = MemmapperSingle(
                data_dict,
                prefix,
                key=key,
                concatenate=concatenation,
                pickle_=False,
                dtype=dtypes[key],
            )

        self.keys = keys

        # Store remaining items as a dictionary
        self.sundry = dict() 

        for location in data_dict.keys():
            for allele in data_dict[location]:
                if allele == 'siteLabel':
                    continue

                if location not in self.sundry:
                    self.sundry[location] = dict()

                if allele not in self.sundry[location]:
                    self.sundry[location][allele] = dict()

                for attribute in data_dict[location][allele]:
                    if attribute in keys:
                        continue

                    self.sundry[location][allele][attribute] = data_dict[location][allele][attribute]

        logging.info("Pickling data ... ")
        pickle.dump(self, open(prefix + ".index", "wb"))

    def __len__(self):
        return len(self.sundry.keys())

    def __getitem__(self, index):
        returns = defaultdict(dict)
        site = self.sundry[index]

        # Take sundry data first
        for allele in site:
            for attribute, value in site[allele].items():
                returns[allele][attribute] = value

        # Take memmap data next
        for attribute, dset in self.datasets.items():
            dset_returns = dset[index]
            for allele in dset_returns:
                returns[allele][attribute] = dset_returns[allele][attribute]

        return returns

    @property
    def locations(self):
        return iter(self.sundry.keys())

    def setIndexingMode(self, mode):
        pass
