import torch
import torch.utils.data
import MemmapData
import pickle
import math
import copy
import random
import logging
from functools import reduce


def tensorify(siteData):
    alleles = [a for a in siteData if a != 'siteLabel'];
    tensors0 = [];
    tensors1 = [];
    labels = [];

    for a in alleles:
        tensors0.append(
            torch.transpose(
                torch.ByteTensor(siteData[a]['feature']),
                1, 2
            )
        );
        tensors1.append(
            torch.transpose(
                torch.ByteTensor(siteData[a]['feature2']),
                1, 2
            )
        );
        labels.append(
            siteData[a]['label'][0]
        );

    totalReadDepth = (
        sum(siteData[a]['supportingReadsStrict'][0] for a in alleles),
        sum(siteData[a]['supportingReadsStrict2'][0] for a in alleles),
    );

    tensors = list(zip(tensors0, tensors1));

    return tuple(tensors + labels), totalReadDepth;


class IterableMemmapDataset(torch.utils.data.IterableDataset):
    def __init__(self, memmaplist, *args, **kwargs):
        super().__init__();
        self.memmaplist = memmaplist;
        self._start = None;
        self._stop = None;

        # Test whether all location data is available
        if ('testmode' in kwargs) and (kwargs['testmode']):
            self.testmode = True;
        else:
            self.testmode = False;

        self.tensorify = True;

    @property
    def subsampledLocales(self):
        if hasattr(self, '_locales'):
            return self._locales;
        else:
            return None;

    @subsampledLocales.setter
    def subsampledLocales(self, _locales):
        self._locales = _locales;

    def __iter__(self):
        workerInfo = torch.utils.data.get_worker_info();
        iteratorCopy = copy.deepcopy(self);

        if workerInfo is None:
            iteratorCopy._start = 0;
            iteratorCopy._stop = len(self.memmaplist);
        else:
            numFilesPerWorker = math.ceil(len(self.memmaplist) / workerInfo.num_workers);
            iteratorCopy._start = numFilesPerWorker * workerInfo.id;
            iteratorCopy._stop = min(iteratorCopy._start + numFilesPerWorker, len(self.memmaplist));

        return iteratorCopy;

    def _initNextFile(self):
        self._locationCounter = 0;
        self._fileLocations = list();

        while len(self._fileLocations) == 0:
            self._fileObject = pickle.load(
                open(self.memmaplist[self._fileCounter], 'rb')
            );

            # Force loading the whole file into memory
            for dset in self._fileObject.datasets.values():
                dset._readIntoMem = True;

            self._fileLocations = list(self._fileObject.locations);
            self._fileObject.setIndexingMode('string');

            # If subsampled locations have been provided externally, use them instead
            if self.subsampledLocales is not None:
                self._fileLocations = self.subsampledLocales[self.memmaplist[self._fileCounter]];

            if len(self._fileLocations) == 0:
                self._fileCounter += 1;
                if self._fileCounter >= self._stop:
                    raise StopIteration;

        random.shuffle(self._fileLocations);

    def __next__(self):
        assert(self._start is not None), "Iterator not initialized";

        if not hasattr(self, '_fileCounter'):
            self._fileCounter = self._start;
            self._initNextFile();

        if self._locationCounter >= len(self._fileLocations):
            self._fileObject = None;   # Depopulate to deallocate memory
            self._fileCounter += 1;
            if self._fileCounter >= self._stop:
                raise StopIteration;
            self._initNextFile();

        nextLocation = self._fileLocations[self._locationCounter];
        self._locationCounter += 1;

        if self.testmode:
            return nextLocation;

        data = self._fileObject[nextLocation];

        if self.tensorify:
            return tensorify(data);

        return data;


def collate_fn(batchData):
    return batchData;


if __name__ == "__main__":
    import sys
    testList = [r.rstrip() for r in open(sys.argv[1], 'r').readlines()];
    dataset = IterableMemmapDataset(testList, testmode=True);
    loader = torch.utils.data.DataLoader(dataset, num_workers=2, batch_size=13, collate_fn=collate_fn);

    items = reduce(
        lambda x, y: x + y,
        list(loader),
        []
    );

    targetItems = [];
    for fhandle in testList:
        locations = list(
            pickle.load(open(fhandle, 'rb')).locations
        );
        targetItems += locations;

    assert(len(set(targetItems)) == len(targetItems)), "Duplication found in original data!";
    assert(len(items) == len(set(items))), "Duplication found!";
    items = set(items);
    targetItems = set(targetItems);

    print("Found %d target Items and %d items" % (len(targetItems), len(items)));
    assert(len(items.difference(targetItems)) == 0);
    assert(len(targetItems.difference(items)) == 0);
