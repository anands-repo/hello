import torch
import torch.utils.data
import MemmapData
import pickle
import math
import copy
import random
import logging
from functools import reduce
import collections
import numpy as np


class NoTrueAlleles(Exception):
    def __init__(self, message):
        super().__init__(message)


def subsample(siteData, maxNumReads, tensorKey, tensorNumKey):
    allelesOfImport = [a for a in siteData if ((a != 'siteLabel') and (siteData[a][tensorNumKey][0] > 0))];
    readIds = [(a, i) for a in allelesOfImport for i in range(siteData[a][tensorNumKey][0])];
    subsampledIds = random.sample(readIds, maxNumReads);
    readsToSubsample = collections.defaultdict(list);

    for key, id_ in subsampledIds:
        readsToSubsample[key].append(id_);

    for key in allelesOfImport:
        reads = readsToSubsample[key];
        siteData[key][tensorNumKey][0] = len(reads);

        if len(reads) == 0:
            siteData[key][tensorKey] = np.zeros_like(siteData[key][tensorKey])[0: 1];
        else:
            siteData[key][tensorKey] = siteData[key][tensorKey][reads];


def num_reads_supporting(site_data, key, alleles):
    total = 0

    for a in alleles:
        if a in site_data and key in site_data[a]:
            total += sum(site_data[a][key])

    return total

def tensorify(siteData, maxNumReads=0):
    alleles = [a for a in siteData if a != 'siteLabel'];
    tensors0 = [];
    tensors1 = [];
    labels = [];

    true_alleles = [a for a in alleles if siteData[a]['label'][0] > 0]

    if len(true_alleles) == 0:
        raise NoTrueAlleles("No true alleles at site");

    if (maxNumReads > 0) and (num_reads_supporting(siteData, 'supportingReadsStrict', alleles) > maxNumReads):
        subsample(siteData, maxNumReads, 'feature', 'supportingReadsStrict');

    if (maxNumReads > 0) and (num_reads_supporting(siteData, 'supportingReadsStrict2', alleles) > maxNumReads):
        subsample(siteData, maxNumReads, 'feature2', 'supportingReadsStrict2');

    for a in alleles:
        tensors0.append(
            torch.transpose(
                torch.ByteTensor(siteData[a]['feature']),
                1, 2
            )
        );
        if 'feature2' in siteData[a]:
            tensors1.append(
                torch.transpose(
                    torch.ByteTensor(siteData[a]['feature2']),
                    1, 2
                )
            );
        else:
            tensors1.append(torch.zeros_like(tensors0[-1])[:1].byte())
        labels.append(
            siteData[a]['label'][0]
        );

    totalReadDepth = (
        sum(siteData[a]['supportingReadsStrict'][0] for a in alleles),
        sum(siteData[a]['supportingReadsStrict2'][0] for a in alleles) if \
            ('supportingReadsStrict2' in siteData[alleles[0]]) \
            else 0
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

        # If there is a max reads restriction, enforce it
        if 'maxReadsPerSite' in kwargs:
            self.maxReadsPerSite = kwargs['maxReadsPerSite'];
        else:
            self.maxReadsPerSite = 0;

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

        try:
            if self.tensorify:
                return tensorify(data, self.maxReadsPerSite);
        except NoTrueAlleles:
            return None

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
