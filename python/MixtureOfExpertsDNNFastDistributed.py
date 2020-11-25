import torch
import torch.utils.data
import os
import importlib
import logging
import h5py
import random
import argparse
import math
import numpy as np
import sys
import _pickle as pickle
import shutil
from functools import reduce
from multiprocessing import Pool
import ast
from LRSchedulers import CosineAnnealingWarmRestarts, SineAnnealingWarmRestarts
import MixtureOfExpertsTools
import MixtureOfExpertsAdvanced
import MemmapDatasetLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
import time

RANK = float('inf')
WORLD_SIZE = -1 


# For testing purposes (determinism)
def deterministicBackEnd():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def samplefromMultiNomial(dict_):
    """
    Samples from a multinomial distribution defined by dict_

    :param dict_: dict
        A dictionary indicating, item: probability

    :return: object
        Key that has been sampled
    """
    keys, values = tuple(zip(*dict_.items()))
    sampledIndex = np.argmax(np.random.multinomial(1, values))
    return keys[sampledIndex]


def countNumCorrect(labels, predictions):
    l_ = labels.cpu().data.numpy()
    p = (predictions > 0).float().cpu().data.numpy()
    return np.add.reduce((l_ == p).flatten())


CHECKPOINT_FREQ = 100
TRAIN_MESSAGE_INTERVAL = 100

# No HDF5 file locking necessary. We are only reading stuff.
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

try:
    profile
except Exception:
    def profile(x):
        return x


class BinaryClassifierLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.BCELoss()

    def forward(self, predictions, target, *args, **kwargs):
        target = torch.unsqueeze(target, dim=1)
        return self.loss(torch.sigmoid(predictions), target.float())


class Payload:
    def __init__(self, devices, batches, listTypes=list()):
        """
        Packages a set of batches for consumption by ReadConvolverHybridDNNForDataParallel

        :param devices: list
            List of devices on to which data should be transferred

        :param batches: list
            List of batches encoded as a dictionary

        :param listTypes: list
            List of items to be converted to lists instead of cuda tensors
        """
        assert(len(batches) == len(devices));
        for batch, device in zip(batches, devices):
            for key, value in batch.items():
                if key in listTypes:
                    if (type(value) is tuple) or (type(value) is list):
                        value = tuple(v.cpu().data.tolist() for v in value);
                    else:
                        value = value.cpu().data.tolist();
                else:
                    if (type(value) is tuple) or (type(value) is list):
                        # value = tuple(v.cuda(device=device, non_blocking=True) for v in value);
                        value = tuple(v.to(device=device) for v in value);
                    else:
                        # This is a special case for 'multiplierMode'
                        if type(value) is not str:
                            # value = value.cuda(device=device, non_blocking=True);
                            value = value.to(device=device);
                setattr(self, key + "%d" % device.index, value);


class WrapperForDataParallel(torch.nn.Module):
    """
    A module to allow dataparallel wrapping of our favorite DNNs

    forward accepts a Payload object that encodes data for the specific instance
    as payload.<tensorname><device id>
    """
    def __init__(self, dnn):
        super().__init__();
        self.dnn = dnn;

    def forward(self, payload, *args):
        device = next(self.parameters()).get_device();
        tensors = getattr(payload, 'tensors%d' % device);
        numAllelesPerSite = getattr(payload, 'numAllelesPerSite%d' % device);
        numReadsPerAllele = getattr(payload, 'numReadsPerAllele%d' % device);
        numReadsPerSite = getattr(payload, 'numReadsPerSite%d' % device);
        reference_segments = getattr(payload, 'reference_segments%d' % device)

        return self.dnn(
            tensors,
            numAllelesPerSite,
            numReadsPerAllele,
            reference_segments,
            numReadsPerSite,
        );


def initMetaToUniform(moe):
    """
    Utility function to initialize meta expert to choose all experts
    with approximately equal probability

    :param moe: MoEMergedWrapper
        Mixture of experts model in which to make initialization
    """
    meta = moe.meta

    # See torch.nn.Sequential/torch.nn.Module
    # Pick the last item in ordered dictionary for torch.nn.Sequential
    # This is an iterator, hence cannot index directoy
    for _ in meta.network._modules.values():
        lastLayer = _

    if hasattr(lastLayer, 'weight') and hasattr(lastLayer, 'bias'):
        if RANK == 0: logging.info("Initializing meta-expert's final layer")
        lastLayer.weight.data.normal_(0.0, 1e-2)  # Small weights
        lastLayer.bias.data.fill_(0.33)  # Relatively larger biases
    else:
        if RANK == 0: logging.info("Cannot initialize meta-expert since last layer doesn't have bias (or weight)")


def countParams(model):
    return sum(p.numel() for p in model.parameters())


def collate_function(batch):
    """
    Collate function for hybrid mixture-of-experts model

    :param batch: list
        List of tensors arranged as follows:
        [(tensor0_allele0, tensor1_allele0), (tensor0_allele1, tensor1_allele1), etc]

    :return tuple
        - tuple of torch.Tensors, each [total#reads in batch, featureLength, 18]
        - list: list of torch.Tensor objects indicating labels per site
        - tuple of lists; each list is the number of reads per each allele in the batch
        - list: number of alleles per site
    """
    numReadsPerAllele0, numReadsPerAllele1 = [], []
    numAllelesPerSite = []
    allTensors0, allTensors1 = [], []
    labelsPerSite = []
    numReadsPerSite0 = []
    numReadsPerSite1 = []
    ref_segments = []

    for items_ in batch:
        # Anand: Hack added for Platinum Genomes. Need to check in detail later.
        if items_ is None:
            continue

        items = items_[0]
        depth = items_[1]
        ref_segments.append(items_[2])
        assert(len(items) % 2 == 0)
        numAlleles = len(items) // 2
        numAllelesPerSite.append(numAlleles)
        tensors = items[:numAlleles]
        labels_ = items[numAlleles:]
        tensorSet0, tensorSet1 = tuple(zip(*tensors))
        allTensors0.extend(list(tensorSet0))
        allTensors1.extend(list(tensorSet1))
        numReadsPerAllele0.extend([t.shape[0] for t in tensorSet0])
        numReadsPerAllele1.extend([t.shape[0] for t in tensorSet1])
        labelsPerSite.append(torch.Tensor(labels_))
        numReadsPerSite0.append(sum([t.shape[0] for t in tensorSet0]))
        numReadsPerSite1.append(sum([t.shape[0] for t in tensorSet1]))

    labels = torch.cat(labelsPerSite, dim=0)
    ref_segments = torch.stack(ref_segments, dim=0)  # [batch, length, 5]

    return (
        (torch.cat(allTensors0, dim=0), torch.cat(allTensors1, dim=0)),
        labels,
        (torch.LongTensor(numReadsPerAllele0), torch.LongTensor(numReadsPerAllele1)),
        torch.LongTensor(numAllelesPerSite),
        (torch.LongTensor(numReadsPerSite0), torch.LongTensor(numReadsPerSite1)),
        ref_segments,
    )


def determineLength(filename):
    try:
        return len(list(pickle.load(open(filename, 'rb')).locations))
    except Exception:
        logging.error("Error in filename, %s" % filename)
        raise Exception


def determineLengthList(filenames):
    return sum(determineLength(f) for f in filenames)


def computeLabelOccurrence(memmapfile):
    with open(memmapfile, 'rb') as fhandle:
        data = pickle.load(fhandle)
        frequency = data.countFrequency()

    return frequency


def pruneHomozygous(args):
    memmapfile, keep, onlySNVs = args
    siteLocations = list()
    pruned = 0

    with open(memmapfile, 'rb') as fhandle:
        data = pickle.load(fhandle)
        with h5py.File(data.hdf5, 'r') as hhandle:
            for l in data.locations:
                siteData = hhandle[l]
                alleles = [a for a in siteData.keys() if a != 'siteLabel']
                isIndelSite = any(len(a) != 1 for a in alleles)
                keepIt = random.uniform(0, 1) <= keep
                if (onlySNVs and isIndelSite) or (len(alleles) > 1) or keepIt:
                    siteLocations.append(l)
                else:
                    pruned += 1

    return memmapfile, siteLocations, pruned


class DataLoaderLocal:
    def __init__(self, memmaplist, batchSize=128, numWorkers=10, homSNVKeepRate=1, maxReadsPerSite=0):
        self.memmaplist = list(memmaplist)
        random.shuffle(self.memmaplist)
        self.batchSize = batchSize
        self.numWorkers = numWorkers
        self.maxReadsPerSite = maxReadsPerSite

        # Determine length
        if numWorkers > 0:
            workers = Pool(numWorkers)
            mapper = workers.imap_unordered
        else:
            mapper = map

        if RANK == 0: logging.info("Determining dataset length")

        if homSNVKeepRate < 1:
            self.homSNVKeepRate = homSNVKeepRate
            self._pruneSNVSites()
        else:
            # If pruning is enabled, _pruneSNVSites will compute the length
            # Otherwise, compute it independently
            self._length = reduce(lambda a, b: a + b, mapper(determineLength, self.memmaplist)) // batchSize

        if RANK == 0: logging.info("Number of batches = %d" % self._length)

        # self._computeWeightLabels()
        self.snvRelativeFrequency = None
        self.indelRelativeFrequency = None

    @property
    def max_length(self):
        if hasattr(self, '_max_length'):
            return self._max_length
        else:
            return self._length

    @max_length.setter
    def max_length(self, _m):
        assert(_m <= self._length), "Max length should be <= length"
        self._max_length = _m

    def _pruneSNVSites(self):
        if RANK == 0: logging.info("Will prune some obviously homozygous sites (SNV sites with single allele at site)")
        args = [(memmapfile, self.homSNVKeepRate, True) for memmapfile in self.memmaplist]
        self.localeDictionary = dict()
        self._length = 0
        numPruned = 0

        if self.numWorkers > 0:
            workers = Pool(self.numWorkers)
            mapper = workers.imap_unordered
        else:
            mapper = map

        for i, returns in enumerate(mapper(pruneHomozygous, args)):
            self.localeDictionary[returns[0]] = returns[1]
            self._length += len(returns[1])
            numPruned += returns[2]
            if (i + 1) % 500 == 0:
                if RANK == 0: logging.info("Completed pruning %d files" % (i + 1))

        if RANK == 0: logging.info("Pruned %d sites" % numPruned)
        self._length = self._length // self.batchSize

    def _computeWeightLabels(self):
        if self.numWorkers > 0:
            workers = Pool(self.numWorkers)
            mapper = workers.imap_unordered
        else:
            mapper = map

        if RANK == 0: logging.info("Determining label occurrence frequencies")
        frequency = {'indels': np.array([0, 0]), 'snv': np.array([0, 0])}

        for i, r in enumerate(mapper(computeLabelOccurrence, self.memmaplist)):
            frequency['snv'] += r['snv']
            frequency['indels'] += r['indels']
            if (i + 1) % 500 == 0:
                if RANK == 0: logging.info("Completed processing %d files" % (i + 1))

        self.snvLabelFrequency = frequency['snv']
        self.indelLabelFrequency = frequency['indels']

        def computeRelativeFrequency(array):
            return array / (np.add.reduce(array) + 1e-15)

        self.snvRelativeFrequency = computeRelativeFrequency(self.snvLabelFrequency)
        self.indelRelativeFrequency = computeRelativeFrequency(self.indelLabelFrequency)

    def __iter__(self):
        random.shuffle(self.memmaplist)
        iterableData = MemmapDatasetLoader.IterableMemmapDataset(
            self.memmaplist, maxReadsPerSite=self.maxReadsPerSite)

        # If we have subsampled sites, enforce the use of the subset rather than
        # the complete set of sites in the training set
        if hasattr(self, 'localeDictionary'):
            iterableData.subsampledLocales = self.localeDictionary

        # print("Before making actual dataloader in rank %d" % self.rank, flush=True)
        loader = torch.utils.data.DataLoader(
            iterableData,
            batch_size=self.batchSize,
            collate_fn=collate_function,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
        )

        self.iterator = iter(loader)
        self.count = 0
        return self

    def __next__(self):
        if self.count >= self.max_length:
            del self.iterator
            del self.count
            raise StopIteration

        self.count += 1
        return next(self.iterator)

    def state_dict(self):
        return {'memmaplist': self.memmaplist}

    def load_state_dict(self, dict_):
        self.memmaplist = dict_['memmaplist']

    def __len__(self):
        return self._length


def dataLoader(
    numWorkers,
    batchSize,
    files,
    worldSize,
    ranks,
    trainPct=0.9,
    overfit=False,
    pruneHomozygous=False,
    valData=None,
    homSNVKeepRate=1,
    maxReadsPerSite=0,
):
    """
    Provide a data loader for training

    :param numWorkers: int
        Number of worker threads for fetching data

    :param batchSize: int
        Size of a batch

    :param files: str
        The input data file

    :param worldSize: int
        Size of the world

    :param ranks: int
        Ranks for which memmap data is desired

    :param trainPct: float
        Percentage of data used for train

    :param overfit: bool
        For testing purpose, allow to overfit to training set (set train and val sets to be the same)

    :param pruneHomozygous: bool
        Prune homozygous locations

    :param valData: str
        A separate validation set if necessary

    :return: tuple
        Two torch.utils.data.DataLoader objects for training and validation
    """
    random.seed(13)

    memmaplist = [r.rstrip() for r in open(files, 'r').readlines()]
    random.shuffle(memmaplist)

    print("Counting minimum epoch size of each rank")

    rank_size = math.ceil(len(memmaplist) / worldSize)
    rank_splits = [
        memmaplist[i * rank_size: (i + 1) * rank_size] for i in range(worldSize)
    ]

    workers = Pool(min(20, len(rank_splits)))
    split_lengths = [i // batchSize for i in workers.map(determineLengthList, rank_splits)]
    print("Split lengths are", split_lengths)
    minimum_length = min(split_lengths)
    print("Minimum epoch size is %d" % minimum_length)

    loaders = []

    for j in ranks:
        tLoader = DataLoaderLocal(
            memmaplist[j * rank_size: (j + 1) * rank_size],
            batchSize=batchSize,
            numWorkers=0,
            homSNVKeepRate=homSNVKeepRate,
            maxReadsPerSite=maxReadsPerSite
        )
        tLoader.max_length = minimum_length
        loaders.append(tLoader)

    return loaders


@profile
def train(
    worldSize,
    rank,
    gpu,
    numEpochs=10,
    lr=1e-3,
    configFile=None,
    cuda=True,
    outputPrefix='/tmp/model',
    overfit=False,
    optimizer="Adam",
    momentum=0.9,
    weightDecay=1e-4,
    numEarlyStopIterations=2,
    lrScheduleFactor=-1,
    dataloader=None,
    checkpoint=None,
    checkpointArchive=None,
    enableMultiGPU=False,
    minLr=0,
    maxLr=1e-2,
    T0=10,
    Tmult=2,
    onlyEval=False,
    model=None,
    warmup=False,
    initMeta=False,
    usePredictionLossInVal=False,
    useAccuracyInVal=False,
    logWriter=None,
    entropyRegularizer=0.1,
    entropyDecay=0.5,
    rangeTest=False,
    rangeStep=1,
    detachMeta=False,
    detachExperts=False,
    useSeparateMeta=False,
    weightLabels=False,
    useSeparateValLoss=False,
    pretrain=False,
    smoothing=0,
    aux_loss=0,
    binaryClassifier=False,
    moeType="advanced",
):
    # moeType = "advanced"
    tLoader = dataloader
    configDict = importlib.import_module(configFile).configDict

    if moeType == "advanced":
        raise NotImplementedError
    elif moeType == "attention":
        # searcher = WrapperForDataParallel(
        #     MixtureOfExpertsAdvanced.create_moe_attention_model(
        #         configDict
        #     )
        # )
        searcher = MixtureOfExpertsAdvanced.create_moe_attention_model(configDict)
    else:
        raise NotImplementedError("Only advanced MixtureOfExperts is accepted")

    if detachMeta:
        # Do not propagate gradients downstream from meta-expert
        searcher.dnn.detachMeta = True

    if detachExperts:
        # Do not propagate gradients through experts
        searcher.dnn.detachExperts = True

    if initMeta:
        initMetaToUniform(searcher.dnn)

    # print("Moving to gpu %d from rank %d" % (gpu, gpu), flush=True)

    devices = [torch.device("cuda:%d" % gpu)]
    # devices = [gpu]
    numParams = countParams(searcher)
    searcher.to(devices[0])
    searcher.train(True)

    # print("Rank %d moved to device" % gpu, flush=True)

    searcher = DDP(searcher, device_ids=devices)

    # print("Started DDP on rank %d" % gpu, flush=True)

    if onlyEval:
        if RANK == 0: logging.info("Loading model for evaluation from path %s" % model)
        searcher = torch.load(model).module
        if enableMultiGPU:
            searcher = torch.nn.DataParallel(searcher)
    elif model is not None:
        if RANK == 0: logging.info("Loading initial parameters from model %s" % model)
        model_ = torch.load(model, map_location='cpu')
        searcher.load_state_dict(model_.state_dict())

    # If we are doing lr scan, then we start with minLr
    if rangeTest:
        if RANK == 0: logging.info("For lr range test, initializing learning rate to minimum %0.10f" % minLr)
        lr = minLr

    if optimizer == "Adam":
        if RANK == 0: logging.info("Using the Adam optimizer")
        # Note: for warmup using Sine scheduling, learning rate starts at max lr. This is because
        # sine scheduler uses a phase-shifted version of the cosine scheduler
        optim = torch.optim.Adam(searcher.parameters(), lr=(lr if not warmup else maxLr))
    else:
        if RANK == 0: logging.info("Using the SGD(R) optimizer")

        if (optimizer == "SGDR") or warmup:
            lr = maxLr

        optim = torch.optim.SGD(searcher.parameters(), lr=lr, momentum=momentum, weight_decay=weightDecay)

    weights = [1, 1] if not weightLabels else tLoader.relativeFrequency
    prevLoss = float("inf")

    if binaryClassifier:
        qLossFn = BinaryClassifierLoss()
        vLossFn = BinaryClassifierLoss()
    else:
        qLossFn = MixtureOfExpertsTools.MoELoss(
            regularizer=entropyRegularizer, decay=entropyDecay, provideIndividualLoss=True, weights=weights, smoothing=smoothing, aux_loss=aux_loss,
        )

        if usePredictionLossInVal:
            if RANK == 0: logging.info("Using prediction loss in validation")
            vLossFn = MixtureOfExpertsTools.PredictionLoss()
        elif useAccuracyInVal:
            if RANK == 0: logging.info("Using accuracy in validation")
            vLossFn = MixtureOfExpertsTools.Accuracy()
        elif useSeparateValLoss:
            if RANK == 0: logging.info("Using separate validation loss function")
            vLossFn = MixtureOfExpertsTools.MoELoss(
                provideIndividualLoss=True
            )
        else:
            if RANK == 0: logging.info("Reusing training loss in validation")
            vLossFn = qLossFn

    # print("Moving q loss function to device in GPU %d" % gpu, flush=True)
    qLossFn.train(True)
    qLossFn.to(devices[0])
    # print("Moved q loss function to device in GPU %d" % gpu, flush=True)

    totalLoss = 0
    numIterLossDecrease = 0
    scheduler = None

    if not warmup:
        if lrScheduleFactor > 0:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optim, factor=lrScheduleFactor, patience=0, cooldown=1, verbose=True, min_lr=minLr
            )

        if optimizer == "SGDR":
            scheduler = CosineAnnealingWarmRestarts(optim, T_0=T0, T_mult=Tmult, eta_min=minLr)
    else:
        scheduler = SineAnnealingWarmRestarts(optim, T_0=T0, T_mult=Tmult, eta_min=minLr)

    maxQ = None
    totalQ = 0

    if not enableMultiGPU:
        devices = [devices[0]]
    else:
        assert(CHECKPOINT_FREQ % len(devices) == 0), "Checkpoint frequency should be a multiple of number of devices in multi-GPU mode"

    if checkpoint is not None:
        if RANK == 0: logging.info("Loading from checkpoint %s ... " % checkpoint)
        logging.warning(
            "We only perform warm-starting through checkpointing - only model parameters are restored"
        )
        checkpoint = torch.load(checkpoint)
        searcher.load_state_dict(checkpoint['model_checkpoint'])

    def performCheckpoint(epoch, batch, itertype, prevloss):
        if onlyEval:
            if RANK == 0: logging.info("Not performing checkpoint since this is an eval-only run")
            return

        seed = torch.get_rng_state()
        checkpoint = {
            'model_checkpoint': searcher.state_dict(),
            'optimizer_checkpoint': optim.state_dict(),
            'itertype': itertype,
            'epoch': epoch,
            'batch': batch,
            'maxQ': maxQ,
            'seed': seed,
            'prevloss': prevloss,
            'tLoaderState': tLoader.state_dict(),
            'randomState': random.getstate(),
            'numIterLossDecrease': numIterLossDecrease,
        }

        if scheduler is not None:
            checkpoint['lr_scheduler_checkpoint'] = scheduler.state_dict()

        torch.save(checkpoint, outputPrefix + ".epoch%d.checkpoint" % epoch)
        if RANK == 0: logging.info("Performed checkpointing; epoch: %d, batch: %d, itertype: %s" % (epoch, batch, itertype))

    trainIterNumber = 0

    if RANK == 0: logging.info("Starting training of model with %d parameters" % numParams)

    # print("Before epochs rank %d" % gpu, flush=True)

    for j in range(numEpochs):
        # Only training iteration is performed for distributed training
        itertypeList = ['train']

        # print("Entered epoch rank %d" % gpu, flush=True)

        for iterType in itertypeList:
            totalLoss = 0
            totalSize = 0
            totalQ = 0
            loader = tLoader if iterType == "train" else vLoader

            # # At the start of each training iteration, shuffle the indices
            # # Currently, we do not shuffle the indices for the validation iteration
            # # Note, this shuffling happens only if we are starting at iteration 0, because
            # # in this case, it means either we didn't restore from a checkpoint, or
            # # we simply checkpointed at the start of the last validation iteration (and not
            # # during the training iteration. This means that the indices have not been shuffled for
            # # the training iteration in this epoch)
            # if (iterType == "train") and (batchStart == 0):
            #     tLoader.sampler.shuffle()

            i_ = 0

            # print("Before making iterator %d" % gpu, flush=True)

            loaderIter = iter(loader)

            # print("After making iterator %d" % gpu, flush=True)
            numCorrect = 0
            numLabels = 0

            # print("Here1 rank %d" % gpu, flush=True)

            # while True:
            for i, batch in enumerate(loaderIter):
                indiv = None

                logging.debug("Starting batch")

                batches = []
                labels = []
                numAllelesPerSiteAll = []

                tensors = batch[0]
                labels = batch[1]
                numReadsPerAllele = batch[2]
                numAllelesPerSite = batch[3]
                numReadsPerSite = batch[4]
                reference_segments = batch[5]
                numAllelesPerSiteAll = numAllelesPerSite.cpu().tolist()

                # print("Here2 rank %d" % gpu)

                def cudify(data):
                    if type(data) is tuple or type(data) is list:
                        return tuple(d.to(devices[0]) for d in data)
                    else:
                        return data.to(devices[0])

                def listify(data):
                    if type(data) is tuple or type(data) is list:
                        return tuple(d.cpu().tolist() for d in data)
                    else:
                        return data.cpu().tolist()
                
                # print("Preparing payload for rank %d" % gpu, flush=True)
                tensors = cudify(tensors)
                labels = cudify(labels)
                numReadsPerAllele = listify(numReadsPerAllele)
                numAllelesPerSite = listify(numAllelesPerSite)
                numReadsPerSite = listify(numReadsPerSite)
                reference_segments = cudify(reference_segments)
                # print("Completed preparing payload for rank %d" % gpu, flush=True)

                if iterType == "train":
                    # Use either SGDR scheduling or learning-rate warmup as needed
                    if warmup:
                        if j == 0:
                            scheduler.step(j + i / len(tLoader))
                        elif optimizer == "SGDR":
                            scheduler.step(j - 1 + i / len(tLoader))
                    elif optimizer == "SGDR":
                        scheduler.step(j + i / len(tLoader))

                    trainIterNumber += 1
                    results = searcher(
                        tensors,
                        numAllelesPerSite,
                        numReadsPerAllele,
                        reference_segments,
                        numReadsPerSite
                    )
                    losses_ = qLossFn(results, labels, numAllelesPerSiteAll)

                    # if len(losses_) > 1:
                    if ((type(losses_) is tuple) or (type(losses_) is list)) and (len(losses_) > 1):
                        losses, indiv, posterior = losses_
                        if pretrain:
                            losses = torch.sum(indiv) / 3
                    else:
                        losses = losses_

                    optim.zero_grad()

                    try:
                        losses.backward()
                    except Exception:
                        logging.error("Caught exception in backward in rank %d" % gpu)
                        logging.error("Saving model parameters, and data that resulted in error, and exiting ... ")
                        torch.save(searcher, os.path.abspath(outputPrefix + ".rank%d.err.dnn" % gpu))
                        torch.save(payload, os.path.abspath(outputPrefix + ".rank%d.payload.pth" % gpu))
                        raise ValueError

                    optim.step()
                else:
                    with torch.no_grad():
                        results = searcher(payload)
                        losses_ = vLossFn(results, labels, numAllelesPerSiteAll)
                        numLabels += labels.shape[0]

                        if ((type(losses_) is tuple) or (type(losses_) is list)) and (len(losses_) > 1):
                            losses, indiv, posterior = losses_
                            if pretrain:
                                losses = torch.sum(indiv) / 3
                        else:
                            losses = losses_

                floss = float(losses.cpu().data.numpy().flatten()[0])

                if logWriter is not None:
                    if iterType == "train":
                        if indiv is not None:
                            loss0, loss1, loss2 = indiv.cpu().data.numpy().tolist()
                            if rank == 0:
                                logWriter.add_scalar("trainLoss", floss, trainIterNumber)
                                logWriter.add_scalar("trainLoss0", loss0, trainIterNumber)
                                logWriter.add_scalar("trainLoss1", loss1, trainIterNumber)
                                logWriter.add_scalar("trainLoss2", loss2, trainIterNumber)
                        else:
                            if rank == 0:
                                logWriter.add_scalar("trainLoss", floss, trainIterNumber)

                        if rank == 0:
                            for l, param_group in enumerate(optim.param_groups):
                                logWriter.add_scalar("lr_%d" % l, param_group['lr'], trainIterNumber)

                totalLoss += floss if (not binaryClassifier) else floss * labels.numel()
                totalSize += labels.numel()

                if i % TRAIN_MESSAGE_INTERVAL == 0:
                    if RANK == 0: logging.info("Completed %d-th %s iteration, loss = %f" % (i, iterType, floss))

                    if rangeTest:
                        lr = lr * rangeStep

                        if lr >= maxLr:
                            if RANK == 0: logging.info("Completed lr range test data collection")
                            return

                        if RANK == 0: logging.info("Increasing learning rate to %.10f" % lr)

                        for param_group in optim.param_groups:
                            param_group['lr'] = lr

            if rangeTest:
                if RANK == 0: logging.info("Terminating range test at the end of iteration")
                return

            if not onlyEval:
                totalLoss /= totalSize

            if logWriter is not None and rank == 0:
                if iterType == "train":
                    logWriter.add_scalar("avg_train_loss", totalLoss, trainIterNumber)
                else:
                    logWriter.add_scalar("avg_val_loss", totalLoss, trainIterNumber)

                logWriter.add_scalar("epoch_marker", 0.0, trainIterNumber - 1)
                logWriter.add_scalar("epoch_marker", 1.0, trainIterNumber)

            # If learning-rate warm-up is used, then delete the warmup scheduler
            # and recreate a scheduler based on lrFactor, or SGDR as necessary
            if warmup and (j == 0) and (iterType == "train"):
                if RANK == 0: logging.info("LR-warmup was used. Deleting scheduler after the first epoch.")
                scheduler = None

                # Note: we shouldn't have to touch base_lrs in the schedulers. This is because by the time
                # epoch 0 training is over the optimizer is warmed up to its maximum learning rate
                if lrScheduleFactor > 0:
                    if RANK == 0: logging.info("LR scheduler to be instanciated in place of warmup")
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optim, factor=lrScheduleFactor, patience=0, cooldown=1, verbose=True, min_lr=minLr
                    )
                elif optimizer == "SGDR":
                    if RANK == 0: logging.info("Cosine annealing scheduler to be instanciated in place of warmup")
                    scheduler = CosineAnnealingWarmRestarts(optim, T_0=T0, T_mult=Tmult, eta_min=minLr)

        if onlyEval:
            if RANK == 0: logging.info("Completed validation, averge accuracy = %f" % (totalLoss / numLabels))
            return

        # Reset itertype - this may have been loaded from a checkpoint
        itertype = None

        # learning-rate schedule (this is after the validation iteration)
        # For SGDR, the update happens every batch, so its not done here
        if (scheduler is not None) and (optimizer != "SGDR"):
            scheduler.step(totalLoss)

        if rank == 0:
            performCheckpoint(j, 0, "train", prevLoss)
            if RANK == 0: logging.info("Saving model; total loss = %f" % totalLoss)
            torch.save(searcher, os.path.abspath(outputPrefix + "%d.dnn" % j))
            numIterLossDecrease = 0

        # dist.barrier()

    if RANK == 0: logging.info("Completed all training iterations")


def main(gpu, args, dataloaders):
    global RANK
    global WORLD_SIZE
    RANK = args.rank + gpu
    WORLD_SIZE = args.nodes * args.num_gpus

    logging.basicConfig(
        level=(logging.INFO if not args.debug else logging.DEBUG),
        format='%(asctime)-15s %(message)s',
        filename=args.outputPrefix + "_logging_information.log"
    )

    """ Initialize multi-node """
    dist.init_process_group(
        backend='nccl', init_method='env://', world_size=WORLD_SIZE, rank=RANK
    )

    if RANK == 0:
        if args.tensorLog is not None:
            import torch.utils.tensorboard as tensorboard
            logWriter = tensorboard.SummaryWriter(args.tensorLog)
        else:
            logWriter = None

    if args.lrRangeTest:
        assert(args.tensorLog is not None), "Provide tensorlog path for range test"

    if RANK == 0: logging.info("Optimizer is %s" % args.optimizer)

    torch.manual_seed(args.seed)
    random.seed(13)
    np.random.seed(13)

    if args.test:
        deterministicBackEnd()
        TRAIN_MESSAGE_INTERVAL = 1

    CHECKPOINT_FREQ = args.checkPointFreq

    args.useAccuracyInVal = False

    # dataloader = dataLoader(
    #     numWorkers=args.numWorkers,
    #     batchSize=args.batchSize,
    #     files=args.data,
    #     worldSize=WORLD_SIZE,
    #     rank=RANK,
    #     overfit=args.overfit,
    #     pruneHomozygous=args.pruneHomozygous,
    #     valData=args.valData,
    #     homSNVKeepRate=args.homSNVKeepRate,
    #     maxReadsPerSite=args.maxReadsPerSite,
    # )
    dataloader = dataloaders[gpu]

    def determineDecayRate(startRate, endRate, numSteps):
        # rate * (x ^ nTrain) = 1e-10 (e.g., if we want to decay to 1e-10 by end of epoch)
        # nTrain * log(x) = log(1e-10 / rate)
        # x = exp(1 / nTrain * log(1e-10 / rate))
        return math.exp(
            1 / numSteps * math.log(endRate / startRate)
        )

    if args.entropyDecay == -1:
        nTrain = len(dataloader[0])
        endOfEpochRate = 1e-12
        args.entropyDecay = determineDecayRate(args.entropyRegularizer, endOfEpochRate, nTrain)
        if RANK == 0: logging.info(
            "Setting entropy decay rate to %f for %d iterations with starting entropy rate %f" % (
                args.entropyDecay, nTrain, args.entropyRegularizer
            )
        )

    if args.individualityDecay == -1:
        nTrain = len(dataloader[0])
        endOfEpochRate = 1e-12
        args.individualityDecay = determineDecayRate(args.individuality, endOfEpochRate, nTrain)
        if RANK == 0: logging.info(
            "Setting individuality decay rate to %f for %d iterations with starting individuality rate %f" % (
                args.individualityDecay, nTrain, args.individuality
            )
        )

    train(
        gpu=gpu,
        worldSize=WORLD_SIZE,
        rank=RANK,
        numEpochs=args.numEpochs,
        lr=args.lr,
        configFile=args.config,
        cuda=args.cuda,
        outputPrefix=args.outputPrefix,
        overfit=args.overfit,
        optimizer=args.optimizer,
        momentum=args.momentum,
        weightDecay=args.weightDecay,
        numEarlyStopIterations=args.numEarlyStopIterations,
        lrScheduleFactor=args.lrFactor,
        dataloader=dataloader,
        checkpoint=args.checkpoint,
        checkpointArchive=args.checkpointArchive,
        enableMultiGPU=args.useMultiGPU,
        minLr=args.minLr,
        maxLr=args.maxLr,
        T0=args.T0,
        Tmult=args.Tmult,
        onlyEval=args.onlyEval,
        model=args.model,
        warmup=args.warmup,
        initMeta=args.initMeta,
        usePredictionLossInVal=args.usePredictionLossInVal,
        useAccuracyInVal=args.useAccuracyInVal,
        logWriter=logWriter,
        entropyRegularizer=args.entropyRegularizer,
        entropyDecay=args.entropyDecay,
        rangeTest=args.lrRangeTest,
        rangeStep=args.lrRangeStep,
        detachMeta=args.detachMeta,
        detachExperts=args.detachExperts,
        useSeparateMeta=args.useSeparateMeta,
        weightLabels=args.weightLabels,
        useSeparateValLoss=args.useSeparateValLoss,
        pretrain=args.pretrain,
        smoothing=args.smoothing,
        aux_loss=args.aux_loss,
        binaryClassifier=args.binaryClassifier,
        moeType=args.moeType,
    )

    if logWriter is not None:
        logWriter.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train variant caller")

    parser.add_argument(
        "--data",
        help="HDF5 format data or list of files with such data",
        required=True,
    )

    parser.add_argument(
        "--valData",
        help="If a separate validation set is to be used provide that",
        required=False,
    )

    parser.add_argument(
        "--config",
        help="Config for Mixture-of-Experts",
        default="initConfig",
        required=False,
    )

    parser.add_argument(
        "--numWorkers",
        help="Number of CPU threads for data fetch",
        default=10,
        type=int,
    )

    parser.add_argument(
        "--numEpochs",
        help="Number of epochs of training",
        default=10,
        type=int,
    )

    parser.add_argument(
        "--batchSizePerNode",
        help="Batch size for training",
        default=512,
        type=int,
    )

    parser.add_argument(
        "--lr",
        help="learning rate",
        default=1e-3,
        type=float,
    )

    parser.add_argument(
        "--cuda",
        help="To use GPU or not",
        action="store_true",
        default=True,  # Note: CUDA option is enabled by default
    )

    parser.add_argument(
        "--outputPrefix",
        help="Prefix of output file",
        required=True,
    )

    parser.add_argument(
        "--overfit",
        help="Enable overfitting for testing purposes",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--optimizer",
        default="Adam",
        help="Which optimizer to use",
        choices=["SGD", "Adam", "SGDR"],
    )

    parser.add_argument(
        "--maxLr",
        help="Maximum learning rate for a learning rate scheduler",
        default=2e-2,
        type=float,
    )

    parser.add_argument(
        "--T0",
        help="T0 parameter for SGDR",
        default=1,
        type=int,
    )

    parser.add_argument(
        "--Tmult",
        help="Tmult parameter for SGDR",
        default=2,
        type=int,
    )

    parser.add_argument(
        "--weightDecay",
        help="Weight decay for SGD",
        default=1e-4,  # From the ResNet paper
        type=float,
    )

    parser.add_argument(
        "--momentum",
        help="Momentum for SGD",
        default=0.9,    # From the ResNet paper
        type=float,
    )

    parser.add_argument(
        "--numEarlyStopIterations",
        help="Number of early stop iterations to use",
        default=2,
        type=int,
    )

    parser.add_argument(
        "--lrFactor",
        help="Factor for learning rate scheduling (positive values trigger lr scheduling)",
        default=-1,
        type=float,
    )

    parser.add_argument(
        "--checkpoint",
        help="Checkpoint from which to restore training",
        default=None,
    )

    parser.add_argument(
        "--checkpointArchive",
        help="Path to store older checkpoints",
        default=None,
    )

    parser.add_argument(
        "--debug",
        help="Enable debug messages",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "--checkPointFreq",
        help="Frequency of checkpoints in number of batches",
        default=1000,
        type=int,
    )

    parser.add_argument(
        "--useMultiGPU",
        help="Enable multi-GPU training",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "--test",
        help="Enable testing",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "--minLr",
        help="Minimum learning rate for a learning rate scheduler",
        default=0,
        type=float,
    )

    parser.add_argument(
        "--pruneHomozygous",
        help="Exclude clearly homozygous locations from training",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "--keepPct",
        help="Percentage of clearly homozygous locations to keep",
        default=0.1,
        type=float,
    )

    parser.add_argument(
        "--onlyEval",
        help="Only perform evaluation",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "--model",
        help="Model path for evaluation",
        default=None,
    )

    parser.add_argument(
        "--seed",
        help="Seed value",
        type=int,
        default=3654553191  # od -vAn -N4 -tu4 < /dev/urandom (From DeepVariant)
    )

    parser.add_argument(
        "--warmup",
        default=False,
        help="Use learning-rate warmup; lr is warmed up from minLr to maxLr over an epoch",
        action="store_true",
    )

    parser.add_argument(
        "--initMeta",
        help="Initialize meta-expert to almost uniform probabilities for each expert",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "--usePredictionLossInVal",
        help="Use prediction loss in validation set",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "--tensorLog",
        default=None,
        help="directory for storing tensorboard logs"
    )

    parser.add_argument(
        "--prefetch",
        help="Load all data into memory at the start of every epoch",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "--entropyRegularizer",
        help="Entropy-based regularizing cost function",
        default=0.0,
        type=float,
    )

    parser.add_argument(
        "--entropyDecay",
        help="Entropy regularizer decay rate; set to -1 for auto-set for decay over epoch",
        default=0.5,
        type=float,
    )

    parser.add_argument(
        "--individuality",
        help="Individual experts will be independently trained outside of the MoE cost function",
        default=0,
        type=float,
    )

    parser.add_argument(
        "--individualityDecay",
        help="Decay rate for individuality factor",
        default=0.5,
        type=float,
    )

    parser.add_argument(
        "--lrRangeTest",
        help="Perform lr range test",
        default=False,
        action="store_true"
    )

    parser.add_argument(
        "--lrRangeStep",
        help="Learning rate range steps",
        default=(10 ** 0.25),
        type=float,
    )

    parser.add_argument(
        "--detachMeta",
        help="Detach meta-expert from downstream gradient propagation",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "--detachExperts",
        help="Detach experts from training",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "--useSeparateMeta",
        help="Use separate (unshared) convolvers for the meta-expert",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "--weightLabels",
        help="Apply weights to labels according to their frequency",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "--useSeparateValLoss",
        help="Force use of a separate validation loss function",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "--pretrain",
        help="Pretraining iteration where individual experts are trained",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "--smoothing",
        help="Label smoothing co-efficient",
        default=0,
        type=float,
    )

    parser.add_argument(
        "--homSNVKeepRate",
        help="What fraction of clearly homozygous SNVs to keep",
        default=1,
        type=float,
    )

    parser.add_argument(
        "--aux_loss",
        default=0.0,
        type=float,
        help="Weight for auxiliary loss function",
    )

    parser.add_argument(
        "--maxReadsPerSite",
        help="Maximum number of reads to feed the DNN per site",
        default=0,
        type=int,
    )

    parser.add_argument(
        "--binaryClassifier",
        help="Indicate that we are using a simple binary classifier and not using Mixture of Experts",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "--moeType",
        help="Type of MoE to use",
        default="advanced",
        choices=["advanced", "attention"]
    )

    parser.add_argument(
        "--nodes",
        help="Number of nodes to train with. Each node is assumed to carry a single GPU. Initialize docker containers accordingly",
        required=True,
        type=int,
    )

    parser.add_argument(
        "--rank",
        help="Rank of this node within the cluster",
        required=True,
        type=int,
    )

    parser.add_argument(
        "--master",
        help="Name of the master node",
        required=False,
    )

    parser.add_argument(
        "--port",
        help="Port to use for communication on the master",
        required=False,
    )

    parser.add_argument("--num_gpus", help="Number of gpus/node", default=4, type=int)

    args = parser.parse_args()

    args.batchSize = args.batchSizePerNode // args.num_gpus

    if args.master is not None:
        os.environ["MASTER_ADDR"] = args.master
        os.environ["MASTER_PORT"] = args.port

    dataloaders = dataLoader(
        0,
        args.batchSize,
        args.data,
        worldSize=args.nodes * args.num_gpus,
        ranks=[args.rank + i for i in range(args.num_gpus)],
        maxReadsPerSite=args.maxReadsPerSite,
    )

    mp.spawn(main, nprocs=args.num_gpus, args=(args, dataloaders))
