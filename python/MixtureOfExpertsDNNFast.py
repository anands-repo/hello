import torch
import torch.utils.data
from torch.utils.data.sampler import SubsetRandomSampler
import os
import importlib
import logging
import h5py
import random
import argparse
import math
import numpy as np
from determineMaxQLoss import determineMaxQ, determineMaxQParallelWrapper
import sys
import _pickle as pickle
import shutil
from functools import reduce
from multiprocessing import Pool
import ast
from LRSchedulers import CosineAnnealingWarmRestarts, SineAnnealingWarmRestarts
import AlleleSearcherDNN
import MixtureOfExperts
import MixtureOfExpertsAdvanced
import MemmapDatasetLoader

random.seed(13);
np.random.seed(13);


# For testing purposes (determinism)
def deterministicBackEnd():
    torch.backends.cudnn.deterministic = True;
    torch.backends.cudnn.benchmark = False;


def samplefromMultiNomial(dict_):
    """
    Samples from a multinomial distribution defined by dict_

    :param dict_: dict
        A dictionary indicating, item: probability

    :return: object
        Key that has been sampled
    """
    keys, values = tuple(zip(*dict_.items()));
    sampledIndex = np.argmax(np.random.multinomial(1, values));
    return keys[sampledIndex];


def countNumCorrect(labels, predictions):
    l_ = labels.cpu().data.numpy();
    p = (predictions > 0).float().cpu().data.numpy();
    return np.add.reduce((l_ == p).flatten());


CHECKPOINT_FREQ = 100;
TRAIN_MESSAGE_INTERVAL = 100;

# No HDF5 file locking necessary. We are only reading stuff.
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE";

try:
    profile
except Exception:
    def profile(x):
        return x;


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
    numReadsPerAllele0, numReadsPerAllele1 = [], [];
    numAllelesPerSite = [];
    allTensors0, allTensors1 = [], [];
    labelsPerSite = [];
    numReadsPerSite0 = [];
    numReadsPerSite1 = [];

    for items_ in batch:
        items = items_[0];
        depth = items_[1];
        assert(len(items) % 2 == 0);
        numAlleles = len(items) // 2;
        numAllelesPerSite.append(numAlleles);
        tensors = items[:numAlleles];
        labels_ = items[numAlleles:];
        tensorSet0, tensorSet1 = tuple(zip(*tensors));
        allTensors0.extend(list(tensorSet0));
        allTensors1.extend(list(tensorSet1));
        numReadsPerAllele0.extend([t.shape[0] for t in tensorSet0]);
        numReadsPerAllele1.extend([t.shape[0] for t in tensorSet1]);
        labelsPerSite.append(torch.Tensor(labels_));
        numReadsPerSite0.append(sum([t.shape[0] for t in tensorSet0]));
        numReadsPerSite1.append(sum([t.shape[0] for t in tensorSet1]));

    labels = torch.cat(labelsPerSite, dim=0);

    return (
        (torch.cat(allTensors0, dim=0), torch.cat(allTensors1, dim=0)),
        labels,
        (torch.LongTensor(numReadsPerAllele0), torch.LongTensor(numReadsPerAllele1)),
        torch.LongTensor(numAllelesPerSite),
        (torch.LongTensor(numReadsPerSite0), torch.LongTensor(numReadsPerSite1)),
    );


def determineLength(filename):
    try:
        return len(list(pickle.load(open(filename, 'rb')).locations));
    except Exception:
        logging.error("Error in filename, %s" % filename);
        raise Exception;


def computeLabelOccurrence(memmapfile):
    with open(memmapfile, 'rb') as fhandle:
        data = pickle.load(fhandle);
        frequency = data.countFrequency()

    return frequency;


def pruneHomozygous(args):
    memmapfile, keep, onlySNVs = args;
    siteLocations = list();
    pruned = 0;

    with open(memmapfile, 'rb') as fhandle:
        data = pickle.load(fhandle);
        with h5py.File(data.hdf5, 'r') as hhandle:
            for l in data.locations:
                siteData = hhandle[l];
                alleles = [a for a in siteData.keys() if a != 'siteLabel'];
                isIndelSite = any(len(a) != 1 for a in alleles);
                keepIt = random.uniform(0, 1) <= keep;
                if (onlySNVs and isIndelSite) or (len(alleles) > 1) or keepIt:
                    siteLocations.append(l);
                else:
                    pruned += 1;

    return memmapfile, siteLocations, pruned;


class DataLoaderLocal:
    def __init__(self, memmaplist, batchSize=128, numWorkers=10, homSNVKeepRate=1, maxReadsPerSite=0):
        self.memmaplist = memmaplist;
        random.shuffle(self.memmaplist);
        self.batchSize = batchSize;
        self.numWorkers = numWorkers;
        self.maxReadsPerSite = maxReadsPerSite;

        # Determine length
        if numWorkers > 0:
            workers = Pool(numWorkers);
            mapper = workers.imap_unordered;
        else:
            mapper = map;

        logging.info("Determining dataset length");

        if homSNVKeepRate < 1:
            self.homSNVKeepRate = homSNVKeepRate;
            self._pruneSNVSites();
        else:
            # If pruning is enabled, _pruneSNVSites will compute the length
            # Otherwise, compute it independently
            self._length = reduce(lambda a, b: a + b, mapper(determineLength, self.memmaplist)) // batchSize;

        logging.info("Number of batches = %d" % self._length);

        # self._computeWeightLabels();
        self.snvRelativeFrequency = None;
        self.indelRelativeFrequency = None;

    def _pruneSNVSites(self):
        logging.info("Will prune some obviously homozygous sites (SNV sites with single allele at site)");
        args = [(memmapfile, self.homSNVKeepRate, True) for memmapfile in self.memmaplist];
        self.localeDictionary = dict();
        self._length = 0;
        numPruned = 0;

        if self.numWorkers > 0:
            workers = Pool(self.numWorkers);
            mapper = workers.imap_unordered;
        else:
            mapper = map;

        for i, returns in enumerate(mapper(pruneHomozygous, args)):
            self.localeDictionary[returns[0]] = returns[1];
            self._length += len(returns[1]);
            numPruned += returns[2];
            if (i + 1) % 500 == 0:
                logging.info("Completed pruning %d files" % (i + 1));

        logging.info("Pruned %d sites" % numPruned);
        self._length = self._length // self.batchSize;

    def _computeWeightLabels(self):
        if self.numWorkers > 0:
            workers = Pool(self.numWorkers);
            mapper = workers.imap_unordered;
        else:
            mapper = map;

        logging.info("Determining label occurrence frequencies");
        frequency = {'indels': np.array([0, 0]), 'snv': np.array([0, 0])};

        for i, r in enumerate(mapper(computeLabelOccurrence, self.memmaplist)):
            frequency['snv'] += r['snv'];
            frequency['indels'] += r['indels'];
            if (i + 1) % 500 == 0:
                logging.info("Completed processing %d files" % (i + 1))

        self.snvLabelFrequency = frequency['snv'];
        self.indelLabelFrequency = frequency['indels'];

        def computeRelativeFrequency(array):
            return array / (np.add.reduce(array) + 1e-15);

        self.snvRelativeFrequency = computeRelativeFrequency(self.snvLabelFrequency);
        self.indelRelativeFrequency = computeRelativeFrequency(self.indelLabelFrequency);

    def __iter__(self):
        random.shuffle(self.memmaplist);
        iterableData = MemmapDatasetLoader.IterableMemmapDataset(self.memmaplist, maxReadsPerSite=self.maxReadsPerSite);

        # If we have subsampled sites, enforce the use of the subset rather than
        # the complete set of sites in the training set
        if hasattr(self, 'localeDictionary'):
            iterableData.subsampledLocales = self.localeDictionary;

        loader = torch.utils.data.DataLoader(
            iterableData,
            batch_size=self.batchSize,
            collate_fn=collate_function,
            num_workers=self.numWorkers,
            pin_memory=True,
            drop_last=True,  # Drop the last batch of the data for single GPU deployments
        );
        return iter(loader);

    def state_dict(self):
        return {'memmaplist': self.memmaplist};

    def load_state_dict(self, dict_):
        self.memmaplist = dict_['memmaplist'];

    def __len__(self):
        return self._length;


def dataLoader(
    numWorkers,
    batchSize,
    hdf5,
    trainPct=0.9,
    overfit=False,
    useBlockSampler=False,
    blockSize=500,
    pruneHomozygous=False,
    keepPct=0.1,
    valData=None,
    loadIntoMem=False,
    homSNVKeepRate=1,
    maxReadsPerSite=0,
):
    """
    Provide a data loader for training

    :param numWorkers: int
        Number of worker threads for fetching data

    :param batchSize: int
        Size of a batch

    :param hdf5: str
        The input data file

    :param trainPct: float
        Percentage of data used for train

    :param overfit: bool
        For testing purpose, allow to overfit to training set (set train and val sets to be the same)

    :param padLength: int
        Length to which input tensors are to be padded

    :param useBlockSampler: bool
        Enable sampling per block rather than global sampling

    :param blockSize: int
        Size of a block of data

    :param pruneHomozygous: bool
        Prune homozygous locations

    :param keepPct: float
        Fraction of homozygous regions to keep if pruning

    :param valData: str
        A separate validation set if necessary

    :param loadIntoMem: bool
        Load all data into memory before training/val iteration

    :return: tuple
        Two torch.utils.data.DataLoader objects for training and validation
    """
    memmaplist = [r.rstrip() for r in open(hdf5, 'r').readlines()];
    random.shuffle(memmaplist);

    if overfit:
        tLoader = DataLoaderLocal(memmaplist, batchSize=batchSize, numWorkers=numWorkers, homSNVKeepRate=homSNVKeepRate, maxReadsPerSite=maxReadsPerSite);
        vLoader = DataLoaderLocal(memmaplist, batchSize=batchSize, numWorkers=numWorkers, maxReadsPerSite=maxReadsPerSite);
    elif valData is None:
        tBound = int(len(memmaplist) * trainPct);
        tList = memmaplist[:tBound];
        vList = memmaplist[tBound:];

        tLoader = DataLoaderLocal(
            tList, batchSize=batchSize, numWorkers=numWorkers, homSNVKeepRate=homSNVKeepRate, maxReadsPerSite=maxReadsPerSite
        );
        vLoader = DataLoaderLocal(
            vList, batchSize=batchSize, numWorkers=numWorkers, maxReadsPerSite=maxReadsPerSite
        );
    else:
        logging.info("Using separate training and validation datasets");
        vList = [r.rstrip() for r in open(valData, 'r').readlines()];
        tList = memmaplist;
        tLoader = DataLoaderLocal(
            tList, batchSize=batchSize, numWorkers=numWorkers, homSNVKeepRate=homSNVKeepRate, maxReadsPerSite=maxReadsPerSite
        );
        vLoader = DataLoaderLocal(
            vList, batchSize=batchSize, numWorkers=numWorkers, maxReadsPerSite=maxReadsPerSite
        );

    logging.info("Compiled %d training examples and %d validation examples" % (len(tLoader) * batchSize, len(vLoader) * batchSize));

    logging.info("Relative snv label occurrence frequency in training set = %s" % str(tLoader.snvRelativeFrequency));
    logging.info("Relative indel label occurrence frequency in training set = %s" % str(tLoader.indelRelativeFrequency));
    logging.info("Relative snv label occurrence frequency in validation set = %s" % str(vLoader.snvRelativeFrequency));
    logging.info("Relative indel label occurrence frequency in validation set = %s" % str(vLoader.indelRelativeFrequency));

    return tLoader, vLoader, len(tLoader) * batchSize, len(vLoader) * batchSize;


@profile
def train(
    numEpochs=10,
    batchSize=64,
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
    pretrained=None,
    pretrainedVal=None,
    checkpointArchive=None,
    enableMultiGPU=False,
    minLr=0,
    keepAll=False,
    maxLr=1e-2,
    T0=10,
    Tmult=2,
    onlyEval=False,
    model=None,
    moeType="unmerged",
    warmup=False,
    initMeta=False,
    usePredictionLossInVal=False,
    useAccuracyInVal=False,
    logWriter=None,
    prefetchData=False,
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
):
    tLoader, vLoader, lTrain, lVal = dataloader;
    configDict = importlib.import_module(configFile).configDict;

    if moeType == "advanced":
        searcher = MixtureOfExperts.WrapperForDataParallel(MixtureOfExpertsAdvanced.createMoEFullMergedAdvancedModel(configDict, useSeparateMeta=useSeparateMeta));
    elif moeType == "unmerged":
        searcher = MixtureOfExperts.WrapperForDataParallel(MixtureOfExperts.createMoEModel(configDict));
    elif moeType == "merged":
        searcher = MixtureOfExperts.WrapperForDataParallel(MixtureOfExperts.createMoEMergedModel(configDict));
    elif moeType == "full":
        searcher = MixtureOfExperts.WrapperForDataParallel(MixtureOfExperts.createMoEFullMergedModel(configDict));
    else:
        searcher = MixtureOfExperts.WrapperForDataParallel(MixtureOfExperts.createMoEFullMergedConditionalModel(configDict));

    if detachMeta:
        # Do not propagate gradients downstream from meta-expert
        searcher.dnn.detachMeta = True;

    if detachExperts:
        # Do not propagate gradients through experts
        searcher.dnn.detachExperts = True;

    if initMeta:
        MixtureOfExperts.initMetaToUniform(searcher.dnn);

    if enableMultiGPU:
        searcher = torch.nn.DataParallel(searcher);

    if onlyEval:
        logging.info("Loading model for evaluation from path %s" % model);
        searcher = torch.load(model).module;
        if enableMultiGPU:
            searcher = torch.nn.DataParallel(searcher);
    elif model is not None:
        logging.info("Loading initial parameters from model %s" % model);
        model_ = torch.load(model, map_location='cpu');
        searcher.load_state_dict(model_.state_dict());

    if cuda:
        searcher.cuda();

    # If we are doing lr scan, then we start with minLr
    if rangeTest:
        logging.info("For lr range test, initializing learning rate to minimum %0.10f" % minLr);
        lr = minLr;

    if optimizer == "Adam":
        logging.info("Using the Adam optimizer");
        # Note: for warmup using Sine scheduling, learning rate starts at max lr. This is because
        # sine scheduler uses a phase-shifted version of the cosine scheduler
        optim = torch.optim.Adam(searcher.parameters(), lr=(lr if not warmup else maxLr));
    else:
        logging.info("Using the SGD(R) optimizer");

        if (optimizer == "SGDR") or warmup:
            lr = maxLr;

        optim = torch.optim.SGD(searcher.parameters(), lr=lr, momentum=momentum, weight_decay=weightDecay);

    weights = [1, 1] if not weightLabels else tLoader.relativeFrequency;
    prevLoss = float("inf");
    qLossFn = MixtureOfExperts.MoELoss(
        regularizer=entropyRegularizer, decay=entropyDecay, provideIndividualLoss=True, weights=weights, smoothing=smoothing, aux_loss=aux_loss,
    );

    if usePredictionLossInVal:
        logging.info("Using prediction loss in validation");
        vLossFn = MixtureOfExperts.PredictionLoss();
    elif useAccuracyInVal:
        logging.info("Using accuracy in validation");
        vLossFn = MixtureOfExperts.Accuracy();
    elif useSeparateValLoss:
        logging.info("Using separate validation loss function");
        vLossFn = MixtureOfExperts.MoELoss(
            provideIndividualLoss=True
        );
    else:
        logging.info("Reusing training loss in validation");
        vLossFn = qLossFn;

    if cuda:
        qLossFn.cuda();

    if useSeparateValLoss and cuda:
        vLossFn.cuda();

    totalLoss = 0;
    numIterLossDecrease = 0;
    scheduler = None;

    if not warmup:
        if lrScheduleFactor > 0:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optim, factor=lrScheduleFactor, patience=0, cooldown=1, verbose=True, min_lr=minLr
            );

        if optimizer == "SGDR":
            scheduler = CosineAnnealingWarmRestarts(optim, T_0=T0, T_mult=Tmult, eta_min=minLr);
    else:
        scheduler = SineAnnealingWarmRestarts(optim, T_0=T0, T_mult=Tmult, eta_min=minLr);

    maxQ = None;
    totalQ = 0;
    devices = [torch.device('cuda:%d' % i) for i in range(torch.cuda.device_count())];

    if not enableMultiGPU:
        devices = [devices[0]];
    else:
        assert(CHECKPOINT_FREQ % len(devices) == 0), "Checkpoint frequency should be a multiple of number of devices in multi-GPU mode";

    if checkpoint is not None:
        logging.info("Loading from checkpoint %s ... " % checkpoint);
        checkpoint = torch.load(checkpoint);
        searcher.load_state_dict(checkpoint['model_checkpoint']);
        optim.load_state_dict(checkpoint['optimizer_checkpoint']);
        itertype = checkpoint['itertype'];
        epochStart = checkpoint['epoch'];
        batchStart = checkpoint['batch'];
        maxQ = checkpoint['maxQ'];
        seed = checkpoint['seed'];
        tLoaderState = checkpoint['tLoaderState'];
        vLoaderState = checkpoint['vLoaderState'];
        tLoader.load_state_dict(tLoaderState);
        vLoader.load_state_dict(vLoaderState);
        torch.set_rng_state(seed);
        prevLoss = checkpoint['prevloss'];
        if 'randomState' in checkpoint:
            random.setstate(checkpoint['randomState']);
        if 'numIterLossDecrease' in checkpoint:
            numIterLossDecrease = checkpoint['numIterLossDecrease'];

        # # If we are in the training iteration, restore sampler index number
        # if itertype == "train":
        #     tLoader.sampler.nextIdx = batchSize * batchStart;
        # elif itertype == "val":
        #     vLoader.sampler.nextIdx = batchSize * batchStart;

        if 'lr_scheduler_checkpoint' in checkpoint:
            # If we were doing warmup and have passed epoch 0 training iteration, then reinstanciate the scheduler
            # to the correct type. If no scheduler required, do not do anything.
            if warmup and ((epochStart > 0) or (itertype == "val")):
                logging.info("Reloading from checkpoint. Passed epoch 0 training, deleting warmup scheduler");
                scheduler = None;

                if lrScheduleFactor > 0:
                    logging.info("LR scheduler to be instanciated in place of warmup");
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optim, factor=lrScheduleFactor, patience=0, cooldown=1, verbose=True, min_lr=minLr
                    );
                elif optimizer == "SGDR":
                    logging.info("Cosine annealing scheduler to be instanciated in place of warmup");
                    scheduler = CosineAnnealingWarmRestarts(optim, T_0=T0, T_mult=Tmult, eta_min=minLr);

            if scheduler:
                scheduler.load_state_dict(checkpoint['lr_scheduler_checkpoint']);
        else:
            assert(scheduler is None), "Need an lr-scheduler, but none found in checkpoint!";
    else:
        if pretrained is not None:
            raise NotImplementedError("Currently pre-trained model loading is not supported");

        itertype = None;
        epochStart = 0;
        batchStart = 0;

    def performCheckpoint(epoch, batch, itertype, prevloss):
        if onlyEval:
            logging.info("Not performing checkpoint since this is an eval-only run");
            return;

        # If checkpoint already exists, archive it
        if (checkpointArchive is not None) and (os.path.exists(outputPrefix + ".checkpoint")):
            try:
                shutil.copy(outputPrefix + ".checkpoint", checkpointArchive);
                logging.info("Archived older checkpoint");
            except IOError:
                logging.error("Cannot archive older checkpoint! Overwriting ... ");

        seed = torch.get_rng_state();
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
            'vLoaderState': vLoader.state_dict(),
            'randomState': random.getstate(),
            'numIterLossDecrease': numIterLossDecrease,
        };

        if scheduler is not None:
            checkpoint['lr_scheduler_checkpoint'] = scheduler.state_dict();

        torch.save(checkpoint, outputPrefix + ".checkpoint");

        logging.info("Performed checkpointing; epoch: %d, batch: %d, itertype: %s" % (epoch, batch, itertype));

    numParams = AlleleSearcherDNN.countParams(searcher);
    trainIterNumber = 0;

    logging.info("Starting training of model with %d parameters" % numParams);

    assert(cuda), "This program only works with GPUs";

    for j in range(epochStart, numEpochs):
        if itertype is not None:
            if itertype == "val":
                itertypeList = ['val'];
            else:
                itertypeList = ["train", "val"];
        else:
            itertypeList = ["train", "val"];

        if onlyEval:
            itertypeList = ["val"];

        # Pre-fetch all data at the start of each epoch
        if prefetchData:
            prefetch(tLoader.dataset);

        for iterType in itertypeList:
            totalLoss = 0;
            totalQ = 0;
            loader = tLoader if iterType == "train" else vLoader;
            searcher.train(iterType == "train");
            qLossFn.train(iterType == "train");

            # # At the start of each training iteration, shuffle the indices
            # # Currently, we do not shuffle the indices for the validation iteration
            # # Note, this shuffling happens only if we are starting at iteration 0, because
            # # in this case, it means either we didn't restore from a checkpoint, or
            # # we simply checkpointed at the start of the last validation iteration (and not
            # # during the training iteration. This means that the indices have not been shuffled for
            # # the training iteration in this epoch)
            # if (iterType == "train") and (batchStart == 0):
            #     tLoader.sampler.shuffle();

            # Checkpoint before every validation iteration
            if (iterType == "val"):
                performCheckpoint(j, 0, "val", prevLoss);

            i_ = 0;

            loaderIter = iter(loader);
            numCorrect = 0;
            numLabels = 0;

            while True:
                collectedBatches = [];
                dummyPadLength = 0;
                dummyPadLengthAlleles = 0;
                indiv = None;

                try:
                    for _ in range(len(devices)):
                        collectedBatches.append(next(loaderIter));
                        i_ += 1;
                except StopIteration:
                    logging.info("Completed epoch");
                    # The last multi-batch (multi-GPU case) is discarded.
                    # Since shuffling is on, this isn't a problem
                    break;

                i = i_ + batchStart;

                logging.debug("Starting batch");

                batches = [];
                labels = [];
                numAllelesPerSiteAll = [];

                for batch in collectedBatches:
                    tensors = batch[0];
                    labels.append(batch[1]);
                    numReadsPerAllele = batch[2];
                    numAllelesPerSite = batch[3];
                    numReadsPerSite = batch[4];

                    batchDict = {
                        'tensors': tensors,
                        'numReadsPerAllele': numReadsPerAllele,
                        'numAllelesPerSite': numAllelesPerSite,
                        'numReadsPerSite': numReadsPerSite,
                    };

                    numAllelesPerSiteAll.append(numAllelesPerSite);
                    batches.append(batchDict);

                numAllelesPerSiteAll = torch.cat(numAllelesPerSiteAll, dim=0).tolist();
                payload = AlleleSearcherDNN.Payload(
                    devices, batches, listTypes=['numReadsPerAllele', 'numAllelesPerSite', 'numReadsPerSite']
                );
                labels = torch.cat(labels, dim=0);
                labels = (labels.cuda(non_blocking=True) > 0);

                if iterType == "train":
                    # Use either SGDR scheduling or learning-rate warmup as needed
                    if warmup:
                        if j == 0:
                            scheduler.step(j + i / len(tLoader));
                        elif optimizer == "SGDR":
                            scheduler.step(j - 1 + i / len(tLoader));
                    elif optimizer == "SGDR":
                        scheduler.step(j + i / len(tLoader));

                    trainIterNumber += 1;
                    results = searcher(payload);
                    losses_ = qLossFn(results, labels, numAllelesPerSiteAll);

                    # if len(losses_) > 1:
                    if ((type(losses_) is tuple) or (type(losses_) is list)) and (len(losses_) > 1):
                        losses, indiv, posterior = losses_;
                        if pretrain:
                            losses = torch.sum(indiv) / 3;
                    else:
                        losses = losses_;

                    optim.zero_grad();

                    try:
                        losses.backward();
                    except Exception:
                        logging.error("Caught exception in backward");
                        logging.error("Saving model parameters, and data that resulted in error, and exiting ... ");
                        torch.save(searcher, os.path.abspath(outputPrefix + ".err.dnn"));
                        torch.save(payload, os.path.abspath(outputPrefix + ".payload.pth"));
                        sys.exit(0);

                    # if (logWriter is not None) and (i % 1000 == 0):
                    #     logging.info("Logging gradients with tensorboard");
                    #     MixtureOfExperts.addGradientsToTensorBoard(searcher.module.dnn, logWriter, trainIterNumber);

                    optim.step();
                else:
                    with torch.no_grad():
                        results = searcher(payload);
                        losses_ = vLossFn(results, labels, numAllelesPerSiteAll);
                        numLabels += labels.shape[0];

                        if ((type(losses_) is tuple) or (type(losses_) is list)) and (len(losses_) > 1):
                            losses, indiv, posterior = losses_;
                            if pretrain:
                                losses = torch.sum(indiv) / 3;
                        else:
                            losses = losses_;

                floss = float(losses.cpu().data.numpy().flatten()[0]);

                if logWriter is not None:
                    if iterType == "train":
                        if indiv is not None:
                            loss0, loss1, loss2 = indiv.cpu().data.numpy().tolist();
                            logWriter.add_scalar("trainLoss", floss, trainIterNumber);
                            logWriter.add_scalar("trainLoss0", loss0, trainIterNumber);
                            logWriter.add_scalar("trainLoss1", loss1, trainIterNumber);
                            logWriter.add_scalar("trainLoss2", loss2, trainIterNumber);
                        else:
                            logWriter.add_scalar("trainLoss", floss, trainIterNumber);

                        for l, param_group in enumerate(optim.param_groups):
                            logWriter.add_scalar("lr_%d" % l, param_group['lr'], trainIterNumber);

                totalLoss += floss;

                # # Perform checkpoint every CHECKPOINT_FREQ-th (TRAIN) iteration
                # if iterType == "train":
                #     if (i > 0) and (i % CHECKPOINT_FREQ == 0):
                #         performCheckpoint(j, i, iterType, prevLoss);

                if i % TRAIN_MESSAGE_INTERVAL == 0:
                    logging.info("Completed %d-th %s iteration, loss = %f" % (i, iterType, floss));

                    if rangeTest:
                        lr = lr * rangeStep;

                        if lr >= maxLr:
                            logging.info("Completed lr range test data collection");
                            return;

                        logging.info("Increasing learning rate to %.10f" % lr);

                        for param_group in optim.param_groups:
                            param_group['lr'] = lr;

            if rangeTest:
                logging.info("Terminating range test at the end of iteration");
                return;

            if not onlyEval:
                totalLoss /= lTrain if iterType == "train" else lVal;

            if logWriter is not None:
                if iterType == "train":
                    logWriter.add_scalar("avg_train_loss", totalLoss, trainIterNumber);
                else:
                    logWriter.add_scalar("avg_val_loss", totalLoss, trainIterNumber);

                logWriter.add_scalar("epoch_marker", 0.0, trainIterNumber - 1);
                logWriter.add_scalar("epoch_marker", 1.0, trainIterNumber);

            # Reset batch start - this may have been loaded from a checkpoint
            batchStart = 0;

            # If learning-rate warm-up is used, then delete the warmup scheduler
            # and recreate a scheduler based on lrFactor, or SGDR as necessary
            if warmup and (j == 0) and (iterType == "train"):
                logging.info("LR-warmup was used. Deleting scheduler after the first epoch.");
                scheduler = None;

                # Note: we shouldn't have to touch base_lrs in the schedulers. This is because by the time
                # epoch 0 training is over the optimizer is warmed up to its maximum learning rate
                if lrScheduleFactor > 0:
                    logging.info("LR scheduler to be instanciated in place of warmup");
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optim, factor=lrScheduleFactor, patience=0, cooldown=1, verbose=True, min_lr=minLr
                    );
                elif optimizer == "SGDR":
                    logging.info("Cosine annealing scheduler to be instanciated in place of warmup");
                    scheduler = CosineAnnealingWarmRestarts(optim, T_0=T0, T_mult=Tmult, eta_min=minLr);

        if onlyEval:
            logging.info("Completed validation, averge accuracy = %f" % (totalLoss / numLabels));
            return;

        # Reset itertype - this may have been loaded from a checkpoint
        itertype = None;

        # learning-rate schedule (this is after the validation iteration)
        # For SGDR, the update happens every batch, so its not done here
        if (scheduler is not None) and (optimizer != "SGDR"):
            scheduler.step(totalLoss);

        if prevLoss > totalLoss:
            logging.info("Model improves in iteration (%d), saving model; total loss = %f, best loss = %f" % (j, totalLoss, 0));
            prevLoss = totalLoss;

            # Turned off for now ... can dump the model after training
            # wrapper = MixtureOfExperts.getWrappedDNN(searcher.module.dnn);
            # torch.save(wrapper, os.path.abspath(outputPrefix + ".wrapper.dnn"));
            torch.save(searcher, os.path.abspath(outputPrefix + ".dnn"));
            numIterLossDecrease = 0;
        else:
            logging.info("Model fails to improve in curent iteration (%d); total loss = %f" % (j, totalLoss));
            numIterLossDecrease += 1;
            if (not overfit) and (numIterLossDecrease >= numEarlyStopIterations):
                logging.info("Ending training");
                break;

    logging.info("Theoretical Max Q in validation set = %f, achieved best validation Q = %f" % (0, prevLoss));


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AlleleSearcherDNN");

    parser.add_argument(
        "--data",
        help="HDF5 format data or list of files with such data",
        required=True,
    );

    parser.add_argument(
        "--valData",
        help="If a separate validation set is to be used provide that",
        required=False,
    );

    parser.add_argument(
        "--config",
        help="Config for Mixture-of-Experts",
        default="initConfig",
        required=False,
    );

    parser.add_argument(
        "--moeType",
        help="Type of mixture of experts to use",
        default="unmerged",
        choices=["merged", "unmerged", "full", "fullConditional", "advanced"],
    );

    parser.add_argument(
        "--numWorkers",
        help="Number of CPU threads for data fetch",
        default=10,
        type=int,
    );

    parser.add_argument(
        "--numEpochs",
        help="Number of epochs of training",
        default=10,
        type=int,
    );

    parser.add_argument(
        "--batchSize",
        help="Batch size for training",
        default=128,
        type=int,
    );

    parser.add_argument(
        "--lr",
        help="learning rate",
        default=1e-3,
        type=float,
    );

    parser.add_argument(
        "--cuda",
        help="To use GPU or not",
        action="store_true",
        default=True,  # Note: CUDA option is enabled by default
    );

    parser.add_argument(
        "--outputPrefix",
        help="Prefix of output file",
        required=True,
    );

    parser.add_argument(
        "--overfit",
        help="Enable overfitting for testing purposes",
        action="store_true",
        default=False,
    );

    parser.add_argument(
        "--optimizer",
        default="Adam",
        help="Which optimizer to use",
        choices=["SGD", "Adam", "SGDR"],
    );

    parser.add_argument(
        "--maxLr",
        help="Maximum learning rate for a learning rate scheduler",
        default=2e-2,
        type=float,
    );

    parser.add_argument(
        "--T0",
        help="T0 parameter for SGDR",
        default=1,
        type=int,
    );

    parser.add_argument(
        "--Tmult",
        help="Tmult parameter for SGDR",
        default=2,
        type=int,
    );

    parser.add_argument(
        "--weightDecay",
        help="Weight decay for SGD",
        default=1e-4,  # From the ResNet paper
        type=float,
    );

    parser.add_argument(
        "--momentum",
        help="Momentum for SGD",
        default=0.9,    # From the ResNet paper
        type=float,
    );

    parser.add_argument(
        "--numEarlyStopIterations",
        help="Number of early stop iterations to use",
        default=2,
        type=int,
    );

    parser.add_argument(
        "--lrFactor",
        help="Factor for learning rate scheduling (positive values trigger lr scheduling)",
        default=-1,
        type=float,
    );

    parser.add_argument(
        "--checkpoint",
        help="Checkpoint from which to restore training",
        default=None,
    );

    parser.add_argument(
        "--checkpointArchive",
        help="Path to store older checkpoints",
        default=None,
    );

    parser.add_argument(
        "--debug",
        help="Enable debug messages",
        default=False,
        action="store_true",
    );

    parser.add_argument(
        "--checkPointFreq",
        help="Frequency of checkpoints in number of batches",
        default=1000,
        type=int,
    );

    parser.add_argument(
        "--useMultiGPU",
        help="Enable multi-GPU training",
        default=False,
        action="store_true",
    );

    parser.add_argument(
        "--test",
        help="Enable testing",
        default=False,
        action="store_true",
    );

    parser.add_argument(
        "--minLr",
        help="Minimum learning rate for a learning rate scheduler",
        default=0,
        type=float,
    );

    parser.add_argument(
        "--useBlockSampler",
        help="Use block randomization instead of global randomization",
        action="store_true",
        default=False,
    );

    parser.add_argument(
        "--blockSize",
        help="Block size for block sampler",
        type=int,
        default=500,
    );

    parser.add_argument(
        "--pruneHomozygous",
        help="Exclude clearly homozygous locations from training",
        default=False,
        action="store_true",
    );

    parser.add_argument(
        "--keepPct",
        help="Percentage of clearly homozygous locations to keep",
        default=0.1,
        type=float,
    );

    parser.add_argument(
        "--useReadDepth",
        help="Whether read depth normalization should be used",
        default=False,
        action="store_true",
    );

    parser.add_argument(
        "--depthMode",
        help="How read depth is to be used",
        choices=["normalize", "addendum", None],
        default=None,
    );

    parser.add_argument(
        "--keepAll",
        default=False,
        help="Keep all models trained with improved validation loss",
        action="store_true",
    );

    parser.add_argument(
        "--onlyEval",
        help="Only perform evaluation",
        default=False,
        action="store_true",
    );

    parser.add_argument(
        "--model",
        help="Model path for evaluation",
        default=None,
    );

    parser.add_argument(
        "--seed",
        help="Seed value",
        type=int,
        default=3654553191  # od -vAn -N4 -tu4 < /dev/urandom (From DeepVariant)
    );

    parser.add_argument(
        "--warmup",
        default=False,
        help="Use learning-rate warmup; lr is warmed up from minLr to maxLr over an epoch",
        action="store_true",
    );

    parser.add_argument(
        "--initMeta",
        help="Initialize meta-expert to almost uniform probabilities for each expert",
        default=False,
        action="store_true",
    );

    parser.add_argument(
        "--usePredictionLossInVal",
        help="Use prediction loss in validation set",
        default=False,
        action="store_true",
    );

    parser.add_argument(
        "--tensorLog",
        default=None,
        help="directory for storing tensorboard logs"
    );

    parser.add_argument(
        "--prefetch",
        help="Load all data into memory at the start of every epoch",
        default=False,
        action="store_true",
    );

    parser.add_argument(
        "--entropyRegularizer",
        help="Entropy-based regularizing cost function",
        default=0.0,
        type=float,
    );

    parser.add_argument(
        "--entropyDecay",
        help="Entropy regularizer decay rate; set to -1 for auto-set for decay over epoch",
        default=0.5,
        type=float,
    );

    parser.add_argument(
        "--individuality",
        help="Individual experts will be independently trained outside of the MoE cost function",
        default=0,
        type=float,
    );

    parser.add_argument(
        "--individualityDecay",
        help="Decay rate for individuality factor",
        default=0.5,
        type=float,
    );

    parser.add_argument(
        "--lrRangeTest",
        help="Perform lr range test",
        default=False,
        action="store_true"
    );

    parser.add_argument(
        "--lrRangeStep",
        help="Learning rate range steps",
        default=(10 ** 0.25),
        type=float,
    );

    parser.add_argument(
        "--detachMeta",
        help="Detach meta-expert from downstream gradient propagation",
        default=False,
        action="store_true",
    );

    parser.add_argument(
        "--detachExperts",
        help="Detach experts from training",
        default=False,
        action="store_true",
    );

    parser.add_argument(
        "--useSeparateMeta",
        help="Use separate (unshared) convolvers for the meta-expert",
        default=False,
        action="store_true",
    );

    parser.add_argument(
        "--weightLabels",
        help="Apply weights to labels according to their frequency",
        default=False,
        action="store_true",
    );

    parser.add_argument(
        "--useSeparateValLoss",
        help="Force use of a separate validation loss function",
        default=False,
        action="store_true",
    );

    parser.add_argument(
        "--pretrain",
        help="Pretraining iteration where individual experts are trained",
        default=False,
        action="store_true",
    );

    parser.add_argument(
        "--smoothing",
        help="Label smoothing co-efficient",
        default=0,
        type=float,
    );

    parser.add_argument(
        "--homSNVKeepRate",
        help="What fraction of clearly homozygous SNVs to keep",
        default=1,
        type=float,
    );

    parser.add_argument(
        "--aux_loss",
        default=0.0,
        type=float,
        help="Weight for auxiliary loss function",
    );

    parser.add_argument(
        "--maxReadsPerSite",
        help="Maximum number of reads to feed the DNN per site",
        default=0,
        type=int,
    );

    args = parser.parse_args();

    logging.basicConfig(level=(logging.INFO if not args.debug else logging.DEBUG), format='%(asctime)-15s %(message)s');

    if args.tensorLog is not None:
        import torch.utils.tensorboard as tensorboard
        logWriter = tensorboard.SummaryWriter(args.tensorLog);
    else:
        logWriter = None;

    if args.lrRangeTest:
        assert(args.tensorLog is not None), "Provide tensorlog path for range test";

    logging.info("Optimizer is %s" % args.optimizer);

    torch.manual_seed(args.seed);

    if args.test:
        deterministicBackEnd();
        TRAIN_MESSAGE_INTERVAL = 1;

    CHECKPOINT_FREQ = args.checkPointFreq;

    args.useAccuracyInVal = False;
    # if args.onlyEval:
    #     # args.usePredictionLossInVal = True;
    #     args.useAccuracyInVal = True;
    #     logging.info("Running evaluation");
    # else:
    #     args.useAccuracyInVal = False;

    dataloader = dataLoader(
        numWorkers=args.numWorkers,
        batchSize=args.batchSize,
        hdf5=args.data,
        overfit=args.overfit,
        useBlockSampler=args.useBlockSampler,
        blockSize=args.blockSize,
        pruneHomozygous=args.pruneHomozygous,
        keepPct=args.keepPct,
        valData=args.valData,
        loadIntoMem=args.prefetch,
        homSNVKeepRate=args.homSNVKeepRate,
        maxReadsPerSite=args.maxReadsPerSite,
    );

    def determineDecayRate(startRate, endRate, numSteps):
        # rate * (x ^ nTrain) = 1e-10 (e.g., if we want to decay to 1e-10 by end of epoch)
        # nTrain * log(x) = log(1e-10 / rate)
        # x = exp(1 / nTrain * log(1e-10 / rate))
        return math.exp(
            1 / numSteps * math.log(endRate / startRate)
        );

    if args.entropyDecay == -1:
        nTrain = len(dataloader[0]);
        endOfEpochRate = 1e-12;
        args.entropyDecay = determineDecayRate(args.entropyRegularizer, endOfEpochRate, nTrain);
        logging.info(
            "Setting entropy decay rate to %f for %d iterations with starting entropy rate %f" % (
                args.entropyDecay, nTrain, args.entropyRegularizer
            )
        );

    if args.individualityDecay == -1:
        nTrain = len(dataloader[0]);
        endOfEpochRate = 1e-12;
        args.individualityDecay = determineDecayRate(args.individuality, endOfEpochRate, nTrain);
        logging.info(
            "Setting individuality decay rate to %f for %d iterations with starting individuality rate %f" % (
                args.individualityDecay, nTrain, args.individuality
            )
        );

    train(
        numEpochs=args.numEpochs,
        batchSize=args.batchSize,
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
        pretrained=None,
        pretrainedVal=None,
        checkpointArchive=args.checkpointArchive,
        enableMultiGPU=args.useMultiGPU,
        minLr=args.minLr,
        keepAll=args.keepAll,
        maxLr=args.maxLr,
        T0=args.T0,
        Tmult=args.Tmult,
        onlyEval=args.onlyEval,
        model=args.model,
        moeType=args.moeType,
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
    );

    if logWriter is not None:
        logWriter.close();
