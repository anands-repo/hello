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
import ReadConvolver
from MemmapData import Memmapper, MemmapperSingle, MemmapperCompound
# import _pickle as pickle
import pickle
import shutil
from functools import reduce
from multiprocessing import Pool
import ast
from LRSchedulers import CosineAnnealingWarmRestarts

random.seed(13);
torch.manual_seed(3654553191);  # od -vAn -N4 -tu4 < /dev/urandom (From DeepVariant)
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


class QLoss(torch.nn.Module):
    def __init__(self):
        super().__init__();

    def forward(self, predictions, targets):
        total = 0;

        for p, t in zip(predictions, targets):
            total += torch.sum(-t * torch.log(p + 1e-10));

        return total / len(predictions);


class Flatten(torch.nn.Module):
    def __init__(self):
        super().__init__();

    def forward(self, tensor):
        return tensor.view(tensor.shape[0], -1);


class GlobalPool(torch.nn.Module):
    def __init__(self):
        super().__init__();

    def forward(self, tensor):
        return torch.sum(tensor, dim=2);


class IndexSampler(torch.utils.data.sampler.Sampler):
    """
    Samples items from a set of indices. Doesn't perform random permutations of indices.
    Random permutations if any, should be performed externally using a fixed random seed.
    """
    def __init__(self, indices):
        self.indices = indices;
        self.nextIdx = 0;

    def __len__(self):
        return len(self.indices);

    def __iter__(self):
        return self;

    def shuffle(self):
        random.shuffle(self.indices);

    def __next__(self):
        if (self.nextIdx == len(self)):
            self.nextIdx = 0;
            raise StopIteration;

        elem = self.indices[self.nextIdx];
        self.nextIdx += 1;
        return elem;

    def state_dict(self):
        return self.indices;

    def load_state_dict(self, _dict):
        self.indices = _dict;


class BlockIndexSampler(torch.utils.data.sampler.Sampler):
    """
    Special randomization function for shuffling items
    in a random subset of memmap files rather than shuffling globally
    """
    def __init__(self, indices, dataset, blockSize=1000):
        self.indices = sorted(indices);
        self.dataset = dataset;
        self.blockSize = blockSize;
        self.idxFiles = list();
        counter = 0;
        dsetIdx = 0;
        cluster = [];

        for idx in self.indices:
            start = counter;
            stop = start + len(self.dataset.memmappers[dsetIdx]);

            while not(start <= idx < stop):
                if len(cluster) > 0:
                    self.idxFiles.append(cluster);
                    cluster = [];

                if dsetIdx >= len(self.dataset.memmappers):
                    raise ValueError("indices exceed the amount of data");

                counter = stop;
                dsetIdx += 1;
                start = counter;
                stop = start + len(self.dataset.memmappers[dsetIdx]);

            cluster.append(idx);

        if len(cluster) > 0:
            self.idxFiles.append(cluster);

        self._indexes = list(self.indices);
        self.nextIdx = 0;

    def _setIndicesForIteration(self):
        self._indexes = list();
        currentPerm = list(range(len(self.idxFiles)));
        random.shuffle(currentPerm);
        numBlocks = len(currentPerm) // self.blockSize;

        if numBlocks * self.blockSize != numBlocks:
            numBlocks += 1;

        for i in range(numBlocks):
            blockStart = i * self.blockSize;
            blockStop = min((i + 1) * self.blockSize, len(self.idxFiles));
            blockIds = currentPerm[blockStart: blockStop];
            idxPool = reduce(lambda x, y: x + y, [self.idxFiles[x] for x in blockIds], []);
            random.shuffle(idxPool);
            self._indexes += idxPool;

    def shuffle(self):
        self._setIndicesForIteration();

    def __len__(self):
        return len(self.indices);

    def __iter__(self):
        return self;

    def __next__(self):
        if (self.nextIdx == len(self)):
            self.nextIdx = 0;
            raise StopIteration;

        elem = self._indexes[self.nextIdx];
        self.nextIdx += 1;
        return elem;

    def state_dict(self):
        return self._indexes;

    def load_state_dict(self, _dict):
        self._indexes = _dict;


def determineShape(l, config):
    """
    Given a 1D conv input tensor's length, and a list of convolutional kernels (including pooling),
    determine the output shape

    :param l: int
        Input tensor length l

    :param config: list
        list of layer configurations

    :return: int
        Output tensor's expected length dimension
    """
    currentLength = l;

    def extract(dict_, key, default_=0):
        return dict_["kwargs"][key] if key in dict_["kwargs"] else default_;

    for layer in config:
        if layer['type'] in ["Conv1d", "MaxPool1d", "AvgPool1d"]:
            padding = extract(layer, "padding");
            dilation = extract(layer, "dilation", 1);
            kernel_size = extract(layer, "kernel_size", None);
            stride = extract(layer, "stride", 1);
            currentLength = math.floor(
                (
                    currentLength + 2 * padding - dilation * (kernel_size - 1) - 1
                ) / stride + 1
            );

    return currentLength;


class ResidualBlock(torch.nn.Module):
    """
    Creates a residual block of convolutional layers
    """
    def __init__(self, **kwargs):
        super().__init__();
        # 'feedforward' is a list of feed-forward layers
        # 'shortcut' indicates the shortcut connection
        ffConfigs = kwargs['feedforward'];
        shConfigs = kwargs['shortcut'];
        self.ffNetwork = Network(ffConfigs);
        self.shNetwork = Network(shConfigs);  # This is likely a single torch.nn.ConstantPad1d layer

    def forward(self, tensor):
        return self.ffNetwork(tensor) + self.shNetwork(tensor);


class Noop(torch.nn.Module):
    """
    Noop layer does no operation; for shortcut connections
    """
    def __init__(self):
        super().__init__();

    def forward(self, tensor):
        return tensor;


torch.nn.Flatten = Flatten;
torch.nn.GlobalPool = GlobalPool;
torch.nn.ResidualBlock = ResidualBlock;
torch.nn.Noop = Noop;


class Classifier:
    """
    A wrapper for network for use with PileupAnalyzer
    """
    def __init__(self, network):
        """
        :param network: Network
            The network object being wrapped
        """
        self.network = network;
        self.network.train(False);

    def predict_proba(self, tensor):
        with torch.no_grad():
            result = self.network(tensor);
            return torch.cat(
                (torch.zeros(1, 1) - 1e9, result),
                dim=1
            ).cpu().data.numpy();
            # Make sure value 0 is never called


class Network(torch.nn.Module):
    """
    Base Network class
    """
    def __init__(self, config):
        super().__init__();
        layers = [];

        for i, configuration in enumerate(config):
            layerType = getattr(torch.nn, configuration['type']);
            layer = layerType(**configuration['kwargs']);
            initCNN(layer);
            layers.append(layer);

        self.network = torch.nn.Sequential(*layers);

    def forward(self, tensors, *args, **kwargs):
        return self.network(tensors);


class NetworkL(torch.nn.Module):
    """
    Network implements layers as NetworkList instead of
    as Sequential
    """
    def __init__(self, config):
        super().__init__();

        for i, configuration in enumerate(config):
            layerType = getattr(torch.nn, configuration['type']);
            layer = layerType(**configuration['kwargs']);
            initCNN(layer);
            layers.append(layer);

        self.network = torch.nn.ModuleList(layers);

    def forward(self, tensors, *args, **kwargs):
        modifier = lambda x: x if x != 0 else len(self.network) - 1;
        index = set({len(self.network) - 1});
        outputs = [];

        # The index argument allows the network to be run
        # so that we obtain a set of layers' outputs
        # to enable transfer learning. If the index is 0
        # then we obtain the final layer's output. In this case,
        # we do not allow obtaining the zero-th layer's outputs
        # for transfer learning purposes. If that is desired, use
        # a noop layer at the start
        if ('index' in kwargs) and (kwargs['index'] != 0):
            if type(kwargs['index']) is int:
                index = set({modifier(index)});
            else:
                index = set([modifier(x) for x in kwargs['index']]);

        for i, layer in enumerate(self.network):
            tensors = layer(tensors);
            if i in index:
                outputs.append(tensors);

        if len(outputs) == 1:
            return outputs[0];
        else:
            return tuple(outputs);


class ScoringWrapper(torch.nn.Module):
    """
    Wrapper for scoring function
    """
    def __init__(self, network, padlength=0):
        """
        :param network: torch.nn.Module
            Torch module (or neural network)

        :param padlength: int
            Padding length
        """
        super().__init__();
        self.network = network;
        self.padlength = padlength;

    def forward(self, tensor):
        tensor = torch.transpose(tensor, 0, 1);

        if self.padlength > 0:
            tensor = sizeTensor(tensor, self.padlength);

        tensor = torch.unsqueeze(tensor, dim=0);

        return self.network(tensor);


class HeterozygosityWrapper(torch.nn.Module):
    """
    Wrapper for determining heterozygosity
    """
    def __init__(self, network, networkType, padlength=0):
        """
        :param network: torch.nn.Module
            The neural network module

        :param networkType: str
            "MLP", "Conv", or "XGB"

        :param padlength: int
            Length to which conv feature maps should be padded
        """
        super().__init__();
        self.network = network;
        self.networkType = networkType;
        self.padlength = padlength;
        assert(networkType in ["Conv", "MLP", "XGB"]);

    def forward(self, item):
        """
        :param item: dict
            Feature dictionary

        :return: tensor
            Result tensor
        """
        forSorting = [];

        for allele in item['features']:
            likelihood = item['features'][allele][0];
            forSorting.append((likelihood, allele));

        topTwo = sorted(forSorting, reverse=True)[:2];
        f, a = tuple(zip(*topTwo));
        a0, a1 = a;

        if self.networkType in ['MLP', 'XGB']:
            features = item['features'];
            f = [features[a0][0], features[a1][0], features[a0][1], features[a1][1], features[a0][2], features[a1][2], features[a0][3], features[a1][3]];

            if self.networkType == 'MLP':
                tensor = torch.unsqueeze(torch.Tensor(f), dim=0);
                return self.network(tensor), a;
            else:
                tensor = np.expand_dims(np.array(f), axis=0);
                return self.network.predict_proba(tensor), a;
        else:
            tensors = item['tensors'];
            tensor0 = sizeTensor(torch.transpose(tensors[a0], 0, 1), self.padlength);
            tensor1 = sizeTensor(torch.transpose(tensors[a1], 0, 1), self.padlength);
            tensor = torch.unsqueeze(torch.cat((tensor0, tensor1), dim=0), dim=0);
            return self.network(tensor), a;


def initCNN(layer):
    if (type(layer) is torch.nn.Conv1d) or (type(layer) is torch.nn.Linear):
        logging.info("Initializing layer");
        if hasattr(layer, 'weight'):
            if layer.weight is not None:
                torch.nn.init.kaiming_uniform_(layer.weight);
        if hasattr(layer, 'bias'):
            if layer.bias is not None:
                layer.bias.data.fill_(0.1);


def countParams(network):
    num = 0;

    for param in network.parameters():
        num += param.cpu().data.numel();

    return num;


class ConvolutionalNetwork(torch.nn.Module):
    """
    The network class (CNN). This class may be used to create a scorer to rank alleles,
    and also to create a classifier to make predictions.
    """
    def __init__(self, config):
        """
        :param config: list
            A list of layer configurations
        """
        super().__init__();
        self.config = config;
        preAdaptivePoolingLayers = [];
        postAdaptivePoolingLayers = [];
        self.preAdaptivePoolingLayerConfigs = [];
        foundAdaptivePool = False;

        for configuration in config:
            layerType = getattr(torch.nn, configuration['type']);
            layer = layerType(**configuration['kwargs']);

            if layerType in ["Conv1d", "Linear"]:
                initCNN(layer);

            if configuration['type'] in ["AdaptiveAvgPool1d", "AdaptiveMaxPool1d"]:
                foundAdaptivePool = True;
                self.adaptiveLayer = layer;
                continue;

            if foundAdaptivePool:
                postAdaptivePoolingLayers.append(layer);
            else:
                self.preAdaptivePoolingLayerConfigs.append(configuration);
                preAdaptivePoolingLayers.append(layer);

        if not foundAdaptivePool:
            raise AttributeError("Need to use adaptive pooling layer");

        self.layerSet0 = torch.nn.Sequential(*preAdaptivePoolingLayers);
        self.layerSet1 = torch.nn.Sequential(*postAdaptivePoolingLayers);

    def forward(self, tensors, paddings=None):
        # First perform pre adaptive-pooling layer operations
        results0 = self.layerSet0(tensors);

        # If tensors are padded, then unpad them individually (serial), perform adaptive pooling
        # then put them back together to continue batch processing
        if (tensors.shape[0] > 1) and (paddings is not None):
            individualResults0 = torch.unbind(results0, dim=0);
            adaptivePooledLayers = [];

            for individualResult, poolLength in zip(individualResults0, paddings.cpu().tolist()):
                originalLength = tensors.shape[-1] - poolLength;
                targetShape = determineShape(originalLength, self.preAdaptivePoolingLayerConfigs);
                prunedResult = torch.unsqueeze(individualResult, dim=0)[:, :, :targetShape];
                adaptivePooledLayers.append(self.adaptiveLayer(prunedResult));

            results0 = torch.cat(adaptivePooledLayers, dim=0);
        else:
            results0 = self.adaptiveLayer(results0);

        return self.layerSet1(results0);


class RecurrentNetwork(torch.nn.Module):
    """
    Recurrent neural network implementation for sequence filtration
    """
    def __init__(self, config):
        """
        :param config: list
            List of layer configurations
        """
        super().__init__();
        preLinearLayers = [];
        linearLayers = [];
        foundLinear = False;
        numHidden = None;
        numLayers = None;

        for configuration in config:
            layerType = getattr(torch.nn, configuration['type']);
            layer = layerType(**configuration['kwargs']);

            if configuration['type'] in ["GRU", "LSTM"]:
                if configuration['type'] == "LSTM":
                    raise NotImplementedError("Cannot handle LSTM as of now");

                if numHidden is None:
                    numHidden = configuration['kwargs']['hidden_size'];
                    numLayers = configuration['kwargs']['num_layers'];
                else:
                    raise ValueError("Cannot handle more than one recurrent instanciation");

            if configuration['type'] == "Linear":
                foundLinear = True;

            if not foundLinear:
                preLinearLayers.append(layer);
            else:
                linearLayers.append(layer);

        if not foundLinear:
            raise ValueError("Expect linear layers");

        self.preLinear = preLinearLayers[0];
        self.linear = torch.nn.Sequential(*linearLayers);
        self.h0 = torch.nn.Parameter(torch.Tensor(numLayers, 1, numHidden).normal_(0, 0.1));
        indices = torch.LongTensor(range(1024));
        self.register_buffer('indices', indices);

    def forward(self, tensors, padding=None):
        # Tensors are in the shape: [batch, channels, sequenceLength]
        tensors = torch.transpose(tensors, 0, 1);  # [channels, batch, sequenceLength]
        tensors = torch.transpose(tensors, 0, 2);  # [sequenceLength, batch, channels]
        batchSize = tensors.shape[1];
        init = self.h0.repeat(1, batchSize, 1);
        rnnResults, _ = self.preLinear(tensors, init);

        if padding is not None:
            rnnResults = rnnResults[tensors.shape[0] - padding - 1, self.indices[:batchSize]];
        else:
            rnnResults = rnnResults[-1];

        results = self.linear(rnnResults);

        return results;


def sizeTensor(tensor, length):
    """
    Size a given tensor to a length

    :param tensor: torch.Tensor
        Tensor to be sized

    :param length: int
        Length to which the tensor's last dim should be sized
    """
    if tensor.shape[-1] < length:
        numToPad = length - tensor.shape[-1]
        left = numToPad // 2 if numToPad % 2 == 0 else numToPad // 2 + 1;
        right = numToPad // 2;
        pad = (left, right);
        tensor = torch.nn.functional.pad(tensor, pad);

    elif tensor.shape[-1] != length:
        numToChop = tensor.shape[-1] - length;
        left = numToChop // 2 if numToChop % 2 == 0 else numToChop // 2 + 1;
        right = numToChop // 2;
        tensor = tensor[:, left:];
        if (right > 0):
            tensor = tensor[:, :-right];

    return tensor;


def transposer(t):
    return torch.transpose(t, 1, 2) if (len(t.shape) == 3) else torch.transpose(t, 0, 1);


def combiner(t1, t2):
    return torch.cat((t1, t2), dim=0) if ((len(t1.shape) == 2) and (len(t2.shape) == 2)) else (t1, t2);


def pruneFunction(args_):
    index, keep = args_;
    index.pruneHomozygousIndices(keep);
    return index;


def pruneHomozygousLocations(memmappers, numThreads, keep=0):
    logging.info("Going to prune away clearly homozygous locations from training set, keeping %f fraction" % keep);
    mapper = map;
    prunedMappers = [];

    if numThreads > 1:
        workers = Pool(numThreads);
        mapper = workers.map;

    pruneArgs = [(m, keep) for m in memmappers];

    for i, item in enumerate(mapper(pruneFunction, pruneArgs)):
        prunedMappers.append(item);
        if (i + 1) % 1000 == 0:
            logging.info("Pruned %d locations" % (i + 1));

    return prunedMappers;


class Subsampler:
    def __init__(self, subsampleRates, debug=False):
        if type(subsampleRates) is dict:
            self.subsampleRates = subsampleRates;
            self.hybrid = False;
            logging.info("Initialized subsampler in single mode. Expected use is for single sequencing platform.");
        else:
            self.subsampleRates0 = subsampleRates[0];
            self.subsampleRates1 = subsampleRates[1];
            self.hybrid = True;
            logging.info("Initialized subsampler in hybrid mode. Expected use is for hybrid data.");

        self.debug = debug;

    def subsampleAllele(self, siteValues, allele, key1, key2, rate):
        logging.debug("Subsampling tensor %s for allele %s at rate %f" % (key1, allele, rate));

        # If 'readsKept' is pre-loaded, do not recalculate it (allows for static subsampling)
        if 'readsKept' not in siteValues[allele]:
            numReadTensorsInInput = siteValues[allele][key1].shape[0];  # e.g., siteValues['A']['feature'].shape
            readsToBeKept = np.nonzero(np.random.uniform(0, 1, numReadTensorsInInput) <= rate)[0];  # nonzero returns a tuple
            siteValues[allele]['readsKept'] = readsToBeKept;
        else:
            logging.debug("Found pre-subsampled information, using");
            numReadTensorsInInput = -1;  # Don't care in this case
            readsToBeKept = siteValues[allele]['readsKept'];

        logging.debug("Keeping %d rows out of %d" % (len(readsToBeKept), numReadTensorsInInput));

        if len(readsToBeKept) > 0:
            siteValues[allele][key1] = siteValues[allele][key1][readsToBeKept];
            siteValues[allele][key2][0] = len(readsToBeKept);
        else:
            shape = siteValues[allele][key1].shape;
            shape = tuple([1] + list(shape[1:]));
            siteValues[allele][key1] = np.zeros(shape);
            siteValues[allele][key2][0] = 0;

    def sampleTensors(self, siteValues, key, sampleRate):
        supportingReadsKey = 'supportingReadsStrict';

        if key == 'feature2':
            supportingReadsKey = 'supportingReadsStrict2';

        for allele in siteValues:
            if allele in ['siteLabel', 'isUsable', 'readsKept']:
                continue;

            _ = self.subsampleAllele(siteValues, allele, key, supportingReadsKey, sampleRate);

    def areGTAllelesCovered(self, siteValues, key1, key2):
        flag = True;

        for allele in siteValues:
            if allele in ['siteLabel', 'isUsable', 'readsKept']:
                continue;

            if siteValues[allele][key2][0] == 0:
                assert(
                    np.logical_and.reduce(
                        (siteValues[allele][key1] == 0).flatten()
                    )
                ), "Allele with no support should have zero tensor for feature";

            if siteValues[allele]['label'][0] > 0.4:
                if siteValues[allele][key2][0] == 0:
                    flag = False;

        return flag;

    def __call__(self, siteValues):
        if self.hybrid:
            raise NotImplementedError;
        else:
            sampleRate = samplefromMultiNomial(self.subsampleRates);
            self.sampleTensors(siteValues, 'feature', sampleRate);

            # Check whether ground truth alleles have support
            siteValues['isUsable'] = self.areGTAllelesCovered(siteValues, 'feature', 'supportingReadsStrict');

            # Clear alleles which have no supports
            allelesToDiscard = [];

            for allele in siteValues:
                if allele in ['siteLabel', 'isUsable', 'readsKept']:
                    continue;

                if siteValues[allele]['supportingReadsStrict'][0] == 0:
                    allelesToDiscard.append(allele);

            for allele in allelesToDiscard:
                del siteValues[allele];

            # Check whether there are atleast two candidate alleles at site
            siteValues['isUsable'] = siteValues['isUsable'] and (len(siteValues) > 2);


def subsamplerNoop(siteValues):
    siteValues['isUsable'] = True;


def presubsample(memmapper, filename, subsampleRates):
    if os.path.exists(filename):
        subsampleDict = pickle.load(open(filename, 'rb'));
        logging.info("%s found, using it" % filename);
        return subsampleDict;

    subsampleDict = dict();

    for location in memmapper.locations:
        data = memmapper[location];

        # Not interested in purely homozygous sites
        if len(data.keys()) < 3:
            continue;

        subsampleDict[location] = dict();
        samplingRate = samplefromMultiNomial(subsampleRates);

        for allele in data.keys():
            if allele in ['siteLabel']:
                continue;

            tensorSize = data[allele]['feature'].shape[0];
            readsKept = np.nonzero(np.random.uniform(0, 1, tensorSize) <= samplingRate)[0];
            subsampleDict[location][allele] = readsKept;

        subsampleDict[location]['origTotalReads'] = sum(
            [data[allele]['feature'].shape[0] for allele in data.keys() if allele != 'siteLabel']
        );

    with open(filename, 'wb') as fhandle:
        pickle.dump(subsampleDict, fhandle);

    return subsampleDict;


def presubsampleWrapper(args):
    return presubsample(*args);


class QualityController:
    """
    Checks whether ground-truth allele has very little support, and filters out such cases.
    This is likely an artifact of dynamic-type sampling. This is also fashioned after a DeepVariant
    filter, hence this should be an acceptable filter to use during training with dynamic sampling,
    or static sampling after hotspot detection.
    """
    def __init__(self, fraction):
        self.fraction = fraction;

    def __call__(self, site):
        if self.fraction == 0:
            return;

        if not site['isUsable']:
            return;

        allelesAtSite = [allele for allele in site if allele not in ['siteLabel', 'isUsable']];
        totalNumSupports = sum([site[allele]['supportingReadsStrict'][0] for allele in allelesAtSite]);

        for allele in allelesAtSite:
            if (site[allele]['label'][0] > 0.4) and (site[allele]['supportingReadsStrict'][0] / totalNumSupports < self.fraction):
                site['isUsable'] = False;
                break;


class Hdf5DatasetMemmap(torch.utils.data.Dataset):
    def __init__(self, memmapindexlist, *args, **kwargs):
        self.indexmap = list();
        self.memmappers = list();
        self.subsampler = subsamplerNoop;
        self.presubsamplers = list();
        self.presubsample = ('presubsample' in kwargs) and kwargs['presubsample'];

        if 'fractionalCutoff' in kwargs:
            fractionalCutoff = kwargs['fractionalCutoff'];
        else:
            fractionalCutoff = 0;

        self.qc = QualityController(fractionalCutoff);

        if self.presubsample:
            self.presubsampleTag = kwargs['presubsampleTag'];
            self.subsampleRates = kwargs['subsampleRates'];
        else:
            self.presubsampleTag = None;
            self.subsampleRates = None;

        logging.info("Loading memmap indices");

        presubsampleArgs = [];

        with open(memmapindexlist, 'r') as fhandle:
            for j, line in enumerate(fhandle):
                line = line.strip();
                memmapper = pickle.load(open(line, 'rb'));
                memmapper.setIndexingMode('string');
                # Enable loading of all training data into memory
                if ('loadIntoMem' in kwargs) and kwargs['loadIntoMem']:
                    memmapper.loadIntoMem = True;
                self.memmappers.append(memmapper);
                if self.presubsample:
                    presubsampleArgs.append(
                        (memmapper, line + self.presubsampleTag + ".pkl", self.subsampleRates)
                    );

        numThreads = kwargs['numThreads'] if 'numThreads' in kwargs else 1;

        if ('subsampler' in kwargs) and (kwargs['subsampler'] is not None):
            logging.debug("Found subsampler, will randomly subsample data at each site to simulate coverage variations");
            self.subsampler = kwargs['subsampler'];

        if ('pruneHomozygous' in kwargs) and kwargs['pruneHomozygous']:
            keep = kwargs['keepHomozygous'] if 'keepHomozygous' in kwargs else 0;
            self.memmappers = pruneHomozygousLocations(self.memmappers, numThreads=numThreads, keep=keep);

        if self.presubsample:
            logging.info("Performing static subsampling of reads");
            map_fn = map if numThreads <= 1 else Pool(numThreads).map;
            self.presubsampleDict = dict();
            for subsampleDict in map_fn(presubsampleWrapper, presubsampleArgs):
                self.presubsampleDict.update(subsampleDict);

        if ('advise' in kwargs) and kwargs['advise']:
            for memmapper in self.memmappers:
                memmapper.advise();

        for j, memmapper in enumerate(self.memmappers):
            if not hasattr(self, 'hybrid'):
                if hasattr(memmapper, 'hybrid'):
                    self.hybrid = memmapper.hybrid;
                else:
                    self.hybrid = False;
            else:
                if hasattr(memmapper, 'hybrid'):
                    assert(self.hybrid == memmapper.hybrid), "Conflicting hybrid settings!";

            for i in memmapper.locations:
                self.indexmap.append((memmapper, i));

            if (j + 1) % 1000 == 0:
                logging.info("Loaded %d indices" % (j + 1));

        logging.info("Completed loading memmap indices");

    def tensorType(self, returns):
        if 'feature' in returns:
            nparray = returns['feature'][0];
        else:
            nparray = returns['tensors'][0];

        return torch.ByteTensor if (str(nparray.dtype) == 'uint8') else torch.Tensor;

    def __getitem__(self, index):
        memmapper, indexWithinMapper = self.indexmap[index];

        try:
            values = memmapper[indexWithinMapper];
        except OSError:
            logging.info("-ERROR- Couldn't access location %s" % indexWithinMapper);
            return None;

        if self.presubsample:
            # Pre-load the read indices to be kept
            subsampleDict = self.presubsampleDict[indexWithinMapper];
            for allele, readsKept in subsampleDict.items():
                if allele == 'origTotalReads':
                    continue;

                assert(allele in values);
                values[allele]['readsKept'] = readsKept;

        self.subsampler(values);
        self.qc(values);

        if not values['isUsable']:
            return None;

        del values['isUsable'];

        alleles = [a for a in values.keys() if a != 'siteLabel'];

        if self.hybrid:
            returns = {
                'feature': tuple(values[a]['feature'] for a in alleles),
                'feature2': tuple(values[a]['feature2'] for a in alleles),
                'label': tuple(values[a]['label'][0] for a in alleles)
            };
        else:
            returns = {
                'feature': tuple(values[a]['feature'] for a in alleles),
                'label': tuple(values[a]['label'][0] for a in alleles)
            };

        if self.hybrid:
            totalReadDepth = (
                sum([values[a]['supportingReadsStrict'][0] for a in alleles]),
                sum([values[a]['supportingReadsStrict2'][0] for a in alleles]),
            );
        else:
            totalReadDepth = sum([values[a]['supportingReadsStrict'][0] for a in alleles]);

        if self.hybrid:
            TensorType = self.tensorType(returns);
            tensors0 = [TensorType(t) for t in returns['feature']];
            tensors1 = [TensorType(t_) for t_ in returns['feature2']];
            labels = list(returns['label']);

            tensors = [
                combiner(transposer(t), transposer(t_)) for t, t_ in zip(tensors0, tensors1)
            ];
        else:
            TensorType = self.tensorType(returns);
            tensors = [TensorType(x) for x in returns['feature']];
            tensors = [transposer(t) for t in tensors];
            labels = list(returns['label']);

        # Check for nan
        for t in tensors:
            if type(t) is tuple:
                for t_ in t:
                    assert(not np.logical_or.reduce(torch.isnan(t_).data.numpy().flatten())), "Location of failure is %s" % str(indexWithinMapper);
                    assert(np.logical_and.reduce(((0 <= t_) * (t_ <= 255)).data.numpy().flatten())), "Location of failure is %s" % str(indexWithinMapper);
            else:
                assert(not np.logical_or.reduce(torch.isnan(t).data.numpy().flatten())), "Location of failure is %s" % str(indexWithinMapper);
                assert(np.logical_and.reduce(((0 <= t) * (t <= 255)).data.numpy().flatten())), "Location of failure is %s" % str(indexWithinMapper);

        for l_ in labels:
            assert(0 <= l_ <= 1), "Location of failure is %s" % str(indexWithinMapper);
            assert(l_ in [0, 0.5, 1]), "Location of failure is %s" % str(indexWithinMapper);

        return tuple(tensors + labels), totalReadDepth;

    def __len__(self):
        return len(self.indexmap);


class Hdf5DatasetFileList(torch.utils.data.Dataset):
    """
    Dataset reader when given a list of files
    """
    def __init__(self, filelist, *args, **kwargs):
        logging.info("Traversing file list to obtain keys");
        self.keymaps = dict();
        with open(filelist, 'r') as filelisthandle:
            for i, file_ in enumerate(filelisthandle):
                file_ = file_.rstrip();
                with h5py.File(file_, 'r') as fhandle:
                    self.keymaps.update({key: file_ for key in fhandle.keys()});

                if (i + 1) % 100 == 0:
                    logging.info("Completed %d files" % (i + 1));

        logging.info("Completed file list traversal");
        self.keylist = list(self.keymaps.keys());

    def __getitem__(self, index):
        key = self.keylist[index];
        filename = self.keymaps[key];
        with h5py.File(filename, 'r') as fhandle:
            value = fhandle[key];
            tensors = [];
            labels = [];

            for key in value.keys():
                if key == 'siteLabel':
                    continue;
                tensor = torch.Tensor(value[key]['feature']);
                tensor = torch.transpose(tensor, 1, 2);
                tensors.append(tensor);
                labels.append(value[key]['label'][0]);

            return tuple(tensors + labels);

    def __len__(self):
        return len(self.keylist);


class Hdf5Dataset(torch.utils.data.Dataset):
    """
    Dataset reader for training searcher
    """
    def __init__(self, data, padlength=0, **kwargs):
        logging.info("Loading data, from %s this may take a while ... " % data);
        self.filename = data;
        data = h5py.File(data, 'r');
        self.keys = list(data.keys());
        logging.info("Completed reading data");
        data.close();
        self.padlength = padlength;

        if 'subtractCoverage' in kwargs:
            self.subtractCoverage = kwargs['subtractCoverage'];
            if self.subtractCoverage:
                logging.info("Local coverage will be subtracted from feature maps");
        else:
            self.subtractCoverage = False;

        if 'avg_cov' in kwargs:
            self.avg_cov = kwargs['avg_cov'];
        else:
            self.avg_cov = 1;

        if 'avg_cov2' in kwargs:
            self.avg_cov2 = kwargs['avg_cov2'];
        else:
            self.avg_cov2 = 1;

        # Currently not used : loses information
        if 'normalize' in kwargs:
            self.normalize = kwargs['normalize'];
        else:
            self.normalize = False;

    def __len__(self):
        return len(self.keys);

    def __getitem__(self, index):
        with h5py.File(self.filename, 'r') as data:
            key = self.keys[index];
            origKey = key;
            value = data[key];
            tensors = [];
            tensors2 = [];
            labels = [];

            logging.debug("Key is %s, value is %s" % (str(key), str(value)));
            totalCoverage = 0;
            totalCoverage2 = 0;

            if self.subtractCoverage:
                for key in value.keys():
                    if key != 'siteLabel':    # Bugfix : used to be "key in 'ACGT'", which doesn't make sense for indels
                        if 'supportingReads' not in value[key]:
                            raise ValueError("Subtract coverage requires supporting reads for all reads");

                        totalCoverage += int(value[key]['supportingReads'][0]);

                        if 'supportingReads2' in value[key]:
                            totalCoverage2 += int(value[key]['supportingReads2'][0]);

            for key in value.keys():
                if key == 'siteLabel':
                    continue;
                logging.debug("Allele is %s" % key);
                tensor = torch.Tensor(value[key]['feature']);  # Original shape: [L, 8]
                # tensor = tensor.view(-1, 8);  # TBD: remove statement, tautology

                if self.subtractCoverage:
                    # assert(totalCoverage > 0);
                    tensor = tensor - totalCoverage;

                # Bug-fix: avg_cov division should be available independent of subtractCoverage
                # This doesn't change previous results as they were all using both
                # Date: 2019/08/12
                tensor = tensor / self.avg_cov;

                tensor = torch.transpose(tensor, 0, 1);  # New shape: [8, L]

                if self.padlength > 0:
                    tensor = sizeTensor(tensor, self.padlength);

                tensors.append(tensor);
                labels.append(value[key]['label'][0]);

                # Perform similar operations for tensors2 if feature2 is present
                if 'feature2' in value[key]:
                    logging.debug("Found feature2, hybrid mode");
                    tensor2 = torch.Tensor(value[key]['feature2']);

                    if self.subtractCoverage:
                        tensor2 = tensor2 - totalCoverage2;

                    tensor2 = tensor2 / self.avg_cov2;
                    tensor2 = torch.transpose(tensor2, 0, 1);  # [8, L]

                    if self.padlength > 0:
                        tensor2 = sizeTensor(tensor2, self.padlength);
                    else:
                        raise ValueError("No padding in hybrid mode. This is an error!");

                    tensors2.append(tensor);
                else:
                    logging.debug("Not in hybrid mode");

        if len(tensors2) > 0:
            newTensors = [];

            for t, t_ in zip(tensors, tensors2):
                newTensors.append(torch.cat((t, t_), dim=0));  # [16, L]

            tensors = newTensors;

        for tensor in tensors:
            assert(tensor.shape[0] == 32), "at key %s" % origKey;

        return tuple(tensors + labels);

    @property
    def keys_(self):
        """
        Return all the keys as stored in an instance of this class
        """
        return self.keys;


class Hdf5DatasetPlain(Hdf5Dataset):
    """
    Dataset for the experimental ReadConvolver class
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs);

    def __getitem__(self, index):
        with h5py.File(self.filename, 'r') as data:
            value = data[self.keys[index]];
            tensors = [];
            labels = [];

            for key in value.keys():
                if key == 'siteLabel':
                    continue;

                tensor = torch.Tensor(value[key]['feature']);
                tensor = torch.transpose(tensor, 1, 2);   # [batch,channels,length]
                tensors.append(tensor);
                labels.append(value[key]['label'][0]);

            return tuple(tensors + labels);


class Hdf5DatasetClassifier(Hdf5Dataset):
    """
    Dataset reader for training classifier
    """
    def __init__(self, data, padlength=100, filterHomozygous=True, **kwargs):
        super().__init__(data);
        # Only use sites with at least two alleles (3 keys)
        data = h5py.File(self.filename, 'r');
        newKeys = [];

        if filterHomozygous:
            logging.info("Filtering out sites with only one allele");

            for key in self.keys:
                # Ensure at least two alleles at the site (data for classifier has no more than two alleles)
                if (len(list(data[key].keys())) >= 3) and (data[key]['siteLabel'][0] in [0, 1]):
                    assert(len(list(data[key].keys())) == 3);
                    newKeys.append(key);

            self.keys = newKeys;

        logging.info("Found %d sites to be classified" % len(self.keys));
        data.close();
        self.padlength = padlength;

    def __getitem__(self, index):
        """
        Mandatory function for indexing. Overriding base class version

        :param index: slice/int
            Index to access data

        :return: tuple
            score, number of supporting reads, allele lengths, label
        """
        with h5py.File(self.filename, 'r') as data:
            key = self.keys[index];
            value = data[key];
            tensors = [];

            logging.debug("Key is %s, value is %s" % (str(key), str(value)));

            for key in value.keys():
                if key == 'siteLabel':
                    continue;
                logging.debug("Allele is %s" % key);
                tensor = torch.Tensor(value[key]['feature']);  # Original shape: [L,8]
                # tensor = tensor.view(-1, 8);
                tensor = torch.transpose(tensor, 0, 1);  # New shape: [8,L]

                if self.padlength > 0:
                    if tensor.shape[-1] < self.padlength:
                        numToPad = self.padlength - tensor.shape[-1]
                        left = numToPad // 2 if numToPad % 2 == 0 else numToPad // 2 + 1;
                        right = numToPad // 2;
                        pad = (left, right);
                        tensor = torch.nn.functional.pad(tensor, pad);

                    elif tensor.shape[-1] != self.padlength:
                        numToChop = tensor.shape[-1] - self.padlength;
                        left = numToChop // 2 if numToChop % 2 == 0 else numToChop // 2 + 1;
                        right = numToChop // 2;
                        tensor = tensor[:, left:];
                        if (right > 0):
                            tensor = tensor[:, :-right];

                tensors.append(tensor);
                label = value['siteLabel'][0];

            # Concatenate tensor channels
            if len(tensors) != 2:
                raise ValueError("Site %s has only one tensor" % str(self.keys[index]));

            tensors = torch.cat(tensors, dim=0);

        return tensors, label;


class Hdf5DatasetClassifierOld(Hdf5Dataset):
    """
    Dataset reader for training classifier
    """
    def __init__(self, data, filterHomozygous=True,  *args, **kwargs):
        super().__init__(data);

        if filterHomozygous:
            logging.info("Filtering out sites with only one allele");
            # Only use sites with at least two alleles (3 keys)
            data = h5py.File(self.filename, 'r');
            newKeys = [];
            for key in self.keys:
                # Ensure at least two alleles at the site
                if (len(list(data[key].keys())) >= 3) and (data[key]['siteLabel'][0] in [0, 1]):
                    newKeys.append(key);
            self.keys = newKeys;
            data.close();

        logging.info("Found %d sites to be classified" % len(self.keys));

    def __getitem__(self, index):
        """
        Mandatory function for indexing. Overriding base class version

        :param index: slice/int
            Index to access data

        :return: tuple
            score, number of supporting reads, allele lengths, label
        """
        with h5py.File(self.filename, 'r') as data:
            key = self.keys[index];
            value = data[key];
            siteLabel = value['siteLabel'][0];
            siteScores = [];
            siteSupportingReads = [];
            siteSupportingReadsStrict = [];
            alleleLengths = [];

            for key in value.keys():
                if key != "siteLabel":
                    siteScores.append(value[key]['score'][0]);
                    siteSupportingReads.append(value[key]['supportingReads'][0]);
                    siteSupportingReadsStrict.append(value[key]['supportingReadsStrict'][0]);
                    alleleLengths.append(len(str(key)));

            if len(siteScores) == 1:
                # Dummy score - 10 less than the site's score, is this right??
                raise ValueError("Two alleles expected per site");
            else:
                sortingInput = zip(siteScores, siteSupportingReads, siteSupportingReadsStrict, alleleLengths);
                sortingOutput = sorted(sortingInput, reverse=True);
                siteScores, siteSupportingReads, siteSupportingReadsStrict, alleleLengths = zip(*sortingOutput);

        return torch.FloatTensor(siteScores), torch.LongTensor(siteSupportingReads), torch.LongTensor(alleleLengths), torch.LongTensor(siteSupportingReadsStrict), siteLabel;


def collate_fn(batch):
    """
    Collate function for combining Hdf5Dataset returns

    :param batch: list
        List of items in a batch

    :return: tuple
        Tuple of items to return
    """
    # batch is a list of items
    numEntries = [];
    allTensors = [];
    allLabels = [];

    for item in batch:
        assert(len(item) % 2 == 0), "Both labels and tensors are expected";
        numEntries.append(len(item) // 2);
        allTensors.extend(item[: len(item) // 2]);
        allLabels.extend(item[len(item) // 2:]);

    # Determine how much to pad each tensor to and pad it; always pad on the right side
    maxLength = max([t.shape[-1] for t in allTensors]);
    newAllTensors = [];
    paddings = [];

    for t in allTensors:
        numTotalPad = maxLength - t.shape[-1];

        if numTotalPad > 0:
            pad = (0, numTotalPad);
            t = torch.nn.functional.pad(t, pad);
            paddings.append(numTotalPad)
        else:
            paddings.append(0);

        newAllTensors.append(t);

    allTensors = torch.stack(newAllTensors, dim=0);
    allLabels = torch.Tensor(allLabels);
    numEntries = torch.LongTensor(numEntries);
    allPaddings = torch.LongTensor(paddings);

    return allTensors, allLabels, allPaddings, numEntries;


# def uncollate_fn(collatedTensors, numEntries):
#     """
#     Uncollate a batch of tensors
#
#     :param collatedTensors: torch.Tensor
#         Collated torch tensor
#
#     :param numEntries: torch.LongTensor
#         Number of entries per site
#
#     :return: list
#         Tensors per site
#     """
#     zeros = numEntries.clone()[0:1];
#     zeros[:] = 0;
#     cumulativeNumEntries = torch.cat(
#         (
#             zeros,
#             torch.cumsum(numEntries, dim=0),
#         ),
#         dim=0
#     ).cpu().tolist();
#
#     uncollatedTensors = [];
#
#     for x, y in zip(cumulativeNumEntries[:-1], cumulativeNumEntries[1:]):
#         uncollatedTensors.append(collatedTensors[x:y]);
#
#     return uncollatedTensors;

def uncollate_fn(collatedTensors, numEntries):
    return torch.split(collatedTensors, split_size_or_sections=numEntries);


def dataLoader(
    numWorkers,
    batchSize,
    hdf5,
    trainPct=0.9,
    overfit=False,
    forClassifier=False,
    padLength=0,
    useOld=False,
    filterHomozygous=True,
    subtractCoverage=False,
    avg_cov=1,
    findMaxQ=True,
    advanced=False,
    useAsMemmap=False,
    useBlockSampler=False,
    blockSize=500,
    pruneHomozygous=False,
    subsampler=None,
    advise=False,
    valData=None,
    presubsample=False,
    presubsampleTag=None,
    fractionalCutoff=0.0,
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

    :param forClassifier: bool
        Whether this dataLoader is for classifier or searcher

    :param padLength: int
        Length to which input tensors are to be padded

    :param useOld: bool
        Use old classifier

    :param filterHomozygous: bool
        Filter homozygous sites for classification

    :param subtractCoverage: bool
        Subtract coverage from feature maps

    :param findMaxQ: bool
        Enable finding the theoretical maximum QLoss

    :param advanced: bool
        Whether we are using the advanced kernel or not

    :param useAsMemmap: bool
        To interpret given data as a list of files

    :param useBlockSampler: bool
        Enable sampling per block rather than global sampling

    :param blockSize: int
        Size of a block of data

    :param pruneHomozygous: bool
        Prune homozygous locations

    :param subsampler: Subsampler
        Subsampler object

    :param advise: bool
        Use madvise to pre-empt the kernel

    :param valData: str
        A separate validation set if necessary

    :param presubsample: bool
        Perform subsampling before training

    :param presubsampleTag: str
        A tag to attach to presubsample files

    :param fractionalCutoff: float
        If allele fraction of a GT allele falls below this cut-off, do not
        use example to train

    :return: tuple
        Two torch.utils.data.DataLoader objects for training and validation
    """
    if advanced:
        if useAsMemmap:
            DataType = Hdf5DatasetMemmap;
        else:
            DataType = Hdf5DatasetPlain;
        collate_function = ReadConvolver.collateFnReadConvolver;
    else:
        DataType = Hdf5Dataset if not forClassifier else (Hdf5DatasetClassifierOld if useOld else Hdf5DatasetClassifier);
        collate_function = collate_fn;

    data = DataType(
        hdf5,
        padlength=padLength,
        filterHomozygous=filterHomozygous,
        subtractCoverage=subtractCoverage,
        avg_cov=avg_cov,
        pruneHomozygous=pruneHomozygous,
        numThreads=numWorkers,
        subsampler=subsampler,
        advise=advise,
        presubsample=presubsample,
        presubsampleTag=presubsampleTag,
        subsampleRates=subsampler.subsampleRates if hasattr(subsampler, 'subsampleRates') else None,
        fractionalCutoff=fractionalCutoff,
    );

    if useAsMemmap:
        if data.hybrid:
            collate_function = ReadConvolver.collateFnReadConvolverHybrid;

    indices = list(range(len(data)));
    random.shuffle(indices);  # Random permutation is performed here, no further permutations will be performed

    if valData is None:
        if not args.overfit:
            trainIndices = indices[: int(len(indices) * trainPct)];
            valIndices = indices[int(len(indices) * trainPct):];
        else:
            trainIndices = indices;
            valIndices = indices;

        logging.info("Received %d training examples, %d validation examples" % (len(trainIndices), len(valIndices)));

        if useBlockSampler:
            logging.info("Using block sampler");
            tSampler = BlockIndexSampler(trainIndices, data, blockSize=blockSize);
        else:
            logging.info("Using index sampler");
            tSampler = IndexSampler(trainIndices);

        vSampler = IndexSampler(valIndices);

        if forClassifier:
            tLoader = torch.utils.data.DataLoader(data, batch_size=batchSize, sampler=tSampler, num_workers=numWorkers);
            vLoader = torch.utils.data.DataLoader(data, batch_size=batchSize, sampler=vSampler, num_workers=numWorkers);
        else:
            tLoader = torch.utils.data.DataLoader(data, batch_size=batchSize, sampler=tSampler, num_workers=numWorkers, collate_fn=collate_function, pin_memory=True);
            vLoader = torch.utils.data.DataLoader(data, batch_size=batchSize, sampler=vSampler, num_workers=numWorkers, collate_fn=collate_function, pin_memory=True);
    else:
        # Note: No additional subsampling is applied to separate validation data
        # Make sure that validation data is pre-subsampled to all the necessary coverage values
        logging.info("Using separate training and validation datasets");
        dataV = DataType(
            valData,
            padlength=padLength,
            filterHomozygous=filterHomozygous,
            subtractCoverage=subtractCoverage,
            avg_cov=avg_cov,
            pruneHomozygous=pruneHomozygous,
            numThreads=numWorkers,
            advise=advise,
        );

        trainIndices = indices;
        valIndices = list(range(len(dataV)));

        logging.info("Received %d training examples, %d validation examples" % (len(trainIndices), len(valIndices)));

        if useBlockSampler:
            tSampler = BlockIndexSampler(trainIndices, data, blockSize=blockSize);
        else:
            tSampler = IndexSampler(trainIndices);

        vSampler = IndexSampler(valIndices);

        tLoader = torch.utils.data.DataLoader(data, batch_size=batchSize, sampler=tSampler, num_workers=numWorkers, collate_fn=collate_function, pin_memory=True);
        vLoader = torch.utils.data.DataLoader(dataV, batch_size=batchSize, sampler=vSampler, num_workers=numWorkers, collate_fn=collate_function, pin_memory=True);

    # (DEPRECATED) Print out the theoretical maximum Q score when training sequence model
    if (not forClassifier) and findMaxQ:
        maxQ = 0;
        return tLoader, vLoader, len(trainIndices), len(valIndices), maxQ;
    else:
        return tLoader, vLoader, len(trainIndices), len(valIndices);


class GraphSearcherWrapper(torch.nn.Module):
    """
    A wrapper for the graph searcher DNN for use in PileupAnalyzer

    :param network0: torch.nn.Module
        Torch module to compute "this" allele's "potential" or feature map

    :param network1: torch.nn.Module
        Torch module to compute "other" allele's "potential" or feature map

    :param network2: torch.nn.Module
        Torch module (linear network) to combine "this" and "other" into potential

    :param padlength: int
        Length to which input features should be padded

    :param useAvg: bool
        Whether the other alleles' features should be added together or averaged.
        Averaging might provide a better feature for learning.

    :param useDiff: bool
        Use difference rather than concatenation at the output of the second stage
    """
    def __init__(self, network0, network1, network2, padlength=0, useAvg=False, useDiff=False):
        super().__init__();
        self.network0, self.network1, self.network2 = network0, network1, network2;
        self.padlength = padlength;
        self.useAvg = useAvg;
        self.useDiff = useDiff;

    def forward(self, featureDict, index=None):
        """
        :param featureDict: dict
            Dictionary representing features for each allele

        :param index: list
            Allows one to perform transfer learning by providing
            intermediate layers' outputs
        """
        # Prepare tensors in the correct format
        tensors = [];
        alleles = [];

        if not hasattr(self, 'useDiff'):
            self.useDiff = False;

        if not hasattr(self, 'useAvg'):
            self.useAvg = False;

        for allele, feature in featureDict.items():
            if (type(feature) is tuple) and (len(feature) == 2):
                # Indicates hybrid mode, where we deal with two features
                # instead of one feature
                feature1, feature2 = feature;
                feature1 = torch.transpose(feature1, 0, 1);
                feature2 = torch.transpose(feature2, 0, 1);

                if self.padlength > 0:
                    feature1 = sizeTensor(feature1, self.padlength);
                    feature2 = sizeTensor(feature2, self.padlength);

                tensors.append(
                    torch.cat((feature1, feature2), dim=0)
                );
                alleles.append(allele);
            else:
                feature = torch.transpose(feature, 0, 1);

                if self.padlength > 0:
                    feature = sizeTensor(feature, self.padlength);

                tensors.append(feature);
                alleles.append(allele);

        tensors = torch.stack(tensors, dim=0);

        # Determine partial features, and combine them to score each allele
        f0, f1 = self.network0(tensors), self.network1(tensors);
        numEntriesPerGivenSite = tensors.shape[0];
        indicesForSum = [list(range(numEntriesPerGivenSite)) for i in range(numEntriesPerGivenSite)];

        for i, indices in enumerate(indicesForSum):
            indices.remove(i);

        if self.useAvg:
            featuresOther = torch.stack([torch.mean(f1[indices], dim=0) for indices in indicesForSum], dim=0);
        else:
            featuresOther = torch.stack([torch.sum(f1[indices], dim=0) for indices in indicesForSum], dim=0);

        if self.useDiff:
            finalFeatureSet = f0 - featuresOther;
        else:
            finalFeatureSet = torch.cat((f0, featuresOther), dim=1);

        if index is None:
            finalResults = self.network2(finalFeatureSet);
            return alleles, torch.squeeze(torch.transpose(finalResults, 0, 1), dim=1);
        else:
            finalResults = self.network2(finalFeatureSet, index=index);
            return alleles, finalResults;


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
                        value = tuple(v.cuda(device=device, non_blocking=True) for v in value);
                    else:
                        # This is a special case for 'multiplierMode'
                        if type(value) is not str:
                            value = value.cuda(device=device, non_blocking=True);
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
        if hasattr(payload, 'numReadsPerAllele%d' % device):
            numReadsPerAllele = getattr(payload, 'numReadsPerAllele%d' % device);
        else:
            numReadsPerAllele = None;

        if hasattr(payload, 'depthMultiplier%d' % device):
            depthMultiplier = getattr(payload, 'depthMultiplier%d' % device);
            if hasattr(payload, 'multiplierMode%d' % device):
                multiplierMode = getattr(payload, 'multiplierMode%d' % device);
                return self.dnn(
                    tensors,
                    numAllelesPerSite,
                    numReadsPerAllele,
                    depthMultiplier=depthMultiplier,
                    multiplierMode=multiplierMode,
                );
            else:
                return self.dnn(
                    tensors,
                    numAllelesPerSite,
                    numReadsPerAllele,
                    depthMultiplier=depthMultiplier,
                );
        else:
            return self.dnn(tensors, numAllelesPerSite, numReadsPerAllele);


class GraphSearcher(torch.nn.Module):
    """
    Graph Searcher DNN for comparing each allele to all others
    """
    def __init__(self, network0, network1, network2, useAvg=False, enableMultiGPU=False, useDiff=False, useOneHot=False):
        """
        :param network0: torch.nn.Module
            Torch module to compute "this" allele's "potential" or feature map

        :param network1: torch.nn.Module
            Torch module to compute "other" allele's "potential" or feature map

        :param network2: torch.nn.Module
            Torch module (linear network) to combine "this" and "other" into potential

        :param useAvg: bool
            Whether the other alternative alleles' features should be averaged instead of
            summed up. This might help learning, but needs to be experimented

        :param enableMultiGPU: bool
            If True, we will use torch.nn.DataParallel to run relevant parts of the code

        :param useDiff: bool
            Use difference between one allele and others (true Graph CNN style)

        :param useOneHot: bool
            Use one-hot encoding at the output instead of softmax encoding
        """
        super().__init__();
        # According to pytorch specification, wrapping in DataParallel shouldn't cause
        # any problems when running on CPU, so it is okay to use the wrapped ones
        wrapper = torch.nn.DataParallel if enableMultiGPU else lambda x : x;
        self.network0 = wrapper(network0);
        self.network1 = wrapper(network1);
        self.network2 = wrapper(network2);
        self.useAvg = useAvg;
        self.useDiff = useDiff;
        self.one = torch.nn.Parameter(torch.Tensor([1]));
        self.useOneHot = useOneHot;
        zero = torch.zeros(1, 1);
        self.register_buffer('zero', zero);  # So it carries to GPU with .cuda() call

    def forward(self, tensors, numEntriesPerSite, *args):
        """
        Note: "padding" seems to be a redundant parameter.
        Will keep it for legacy reasons.
        """
        features0 = self.network0(tensors);  # Removed spurious "padding" argument
        features1 = self.network1(tensors);  # Removed spurious "padding" argument
        perSiteFeatures0 = uncollate_fn(features0, numEntriesPerSite);
        perSiteFeatures1 = uncollate_fn(features1, numEntriesPerSite);
        featuresCombined = [];

        for i, (f0, f1) in enumerate(zip(perSiteFeatures0, perSiteFeatures1)):
            # f0 is [numEntriesPerGivenSite, dim]
            # f1 is [numEntriesPerGivenSite, dim]
            # We want to create numEntriesPerSite sums for f1
            # such that the i-th sum doesn't include element i

            if f0.shape[0] == 1:
                # When there is only one entry, choose that entry, so this is a dummy case
                # Simply create a zero of the same feature size as f0
                if len(f0.shape) == 2:
                    featuresOther = self.zero.expand(f0.shape);
                else:
                    featuresOther = torch.unsqueeze(self.zero, dim=0).expand(f0.shape);
            else:
                # We create the necessary indices below for the summation described above
                # Then we perform the numEntriesPerGivenSite sums
                numEntriesPerGivenSite = f0.shape[0];
                indicesForSum = [list(range(numEntriesPerGivenSite)) for i in range(numEntriesPerGivenSite)];

                for i, indices in enumerate(indicesForSum):
                    indices.remove(i);

                if self.useAvg:
                    featuresOther = torch.stack([torch.mean(f1[indices], dim=0) for indices in indicesForSum], dim=0);
                else:
                    featuresOther = torch.stack([torch.sum(f1[indices], dim=0) for indices in indicesForSum], dim=0);

            if self.useDiff:
                featuresCombined.append(f0 - featuresOther);
            else:
                featuresCombined.append(torch.cat((f0, featuresOther), dim=1));

        # Perform final calls
        featuresCombined = torch.cat(featuresCombined, dim=0);
        resultsCombined = self.network2(featuresCombined);

        if not self.useOneHot:
            returns = [];
            perSiteResults = uncollate_fn(resultsCombined, numEntriesPerSite);

            for i, entriesPerSite in enumerate(perSiteResults):
                # Un-collation rearranges entries as [#entries, 1]
                # Remove spurious dimension
                entriesPerSite = torch.squeeze(entriesPerSite, dim=1);
                if entriesPerSite.shape[-1] == 1:
                    returns.append(torch.sigmoid(self.one));
                        # Do not train NN in this case, but allow training of internal dummy parameter
                        # This is redundant, and may be should be removed
                else:
                    returns.append(torch.nn.functional.softmax(entriesPerSite, dim=0));

            # Concatenation is required for DataParallel
            return torch.cat(returns, dim=0);
        else:
            return resultsCombined;


class Searcher(torch.nn.Module):
    """
    The searcher class uses a network to rank alleles
    """
    def __init__(self, network):
        """
        :param network: Network
            Network object (see above)
        """
        super().__init__();
        self.network = network;
        self.one = torch.nn.Parameter(torch.Tensor([1]));

    # def forward(self, tensors, numEntriesPerSite, labels=None):
    def forward(self, tensors, numEntriesPerSite, padding=None):
        results = self.network(tensors, padding);
        perSiteResults = uncollate_fn(results, numEntriesPerSite);
        returns = [];

        for i, entriesPerSite in enumerate(perSiteResults):
            # print(entriesPerSite.cpu().data.numpy().flatten(), labels[i].cpu().data.numpy());
            # Un-collation rearranges entries as [#entries, 1]
            entriesPerSite = torch.transpose(entriesPerSite, 0, 1);

            if entriesPerSite.shape[-1] == 1:
                # returns.append(torch.sigmoid(entriesPerSite));  # When there is only one item, maximize likelihood
                # print(returns[-1].cpu().data.numpy());
                returns.append(torch.sigmoid(self.one));  # Do not train in this case
            else:
                returns.append(torch.nn.functional.softmax(entriesPerSite, dim=1));

        return returns;


def numCorrectPredictions(predictions, labels):
    """
    Determine the number of correct predictions in a batch

    :param predictions: torch.Tensor
        A tensor of predictions

    :param labels: torch.LongTensor
        A batch of labels

    :return: int
        Number of correct predictions
    """
    predictions = np.argmax(predictions.cpu().data.numpy(), axis=1);
    numCorrect = sum((predictions.flatten() == labels.cpu().data.numpy().flatten()).tolist());
    return numCorrect;


def trainClassifier(
    numEpochs=10,
    numWorkers=12,
    batchSize=64,
    lr=1e-3,
    configFile='classifierConfig',
    data='/root/storage/hotspots/train/data.seq_rank4/alleleSearcher/temp/AlleleSearcherTrainWithLabels.hdf5',
    cuda=True,
    outputPrefix='/tmp/classifier',
    overfit=False,
    padlength=0,
    useOld=False,
    filterHomozygous=True,
    optimizer="Adam",
    momentum=0.9,
    weightDecay=1e-4,
    numEarlyStopIterations=2,
):
    raise NotImplementedError;


@profile
def train(
    numEpochs=10,
    numWorkers=12,
    batchSize=64,
    lr=1e-3,
    configFile='initConfig',
    data='/root/storage/hotspots/train/data.seq_rank4/alleleSearcher/temp/AlleleSearcherTrain.hdf5',
    cuda=True,
    outputPrefix='/tmp/model',
    overfit=False,
    networkType="CNN",
    padlength=0,
    subtractCoverage=False,
    avg_cov=1,
    optimizer="Adam",
    momentum=0.9,
    weightDecay=1e-4,
    findMaxQ=True,
    numEarlyStopIterations=2,
    lrScheduleFactor=-1,
    config2=None,
    useAvg=False,
    config3=None,
    useAsMemmap=False,
    dataloader=None,
    checkpoint=None,
    pretrained=None,
    pretrainedVal=None,
    checkpointArchive=None,
    enableMultiGPU=False,
    minLr=0,
    useDiff=False,
    useReadDepth=False,
    multiplierMode=None,
    keepAll=False,
    useOneHot=False,
    maxLr=1e-2,
    T0=10,
    Tmult=2,
    onlyEval=False,
):
    config = importlib.import_module(configFile).config;
    # Allow use of NetworkL for transfer learning
    NType = Network if (networkType == "DNN") else NetworkL;
    network0, network1, network2, network4 = None, None, None, None;

    if dataloader is None:
        tLoader, vLoader, lTrain, lVal, maxQ = dataLoader(
            numWorkers=numWorkers,
            batchSize=batchSize,
            hdf5=data,
            overfit=overfit,
            padLength=padlength,
            subtractCoverage=subtractCoverage,
            avg_cov=avg_cov,
            findMaxQ=findMaxQ,
            advanced=(config3 is not None),
            useAsMemmap=useAsMemmap,
        );
    else:
        tLoader, vLoader, lTrain, lVal, maxQ = dataloader;

    if (config2 is not None) and (config3 is not None):
        config2 = importlib.import_module(config2).config;
        config3 = importlib.import_module(config3).config;
        network0 = NType(config);
        network1 = NType(config);
        network2 = NType(config2);
        network3 = NType(config3);
        network4 = NType(config3) if tLoader.dataset.hybrid else None;
        # gsearcher = GraphSearcher(network0, network1, network2, useAvg=useAvg, enableMultiGPU=enableMultiGPU, useDiff=useDiff);
        gsearcher = GraphSearcher(network0, network1, network2, useAvg=useAvg, useDiff=useDiff, useOneHot=useOneHot);
        if tLoader.dataset.hybrid:
            # Hybrid DNN for advanced has a different architecture compared to the regular DNN
            # searcher = ReadConvolver.ReadConvolverHybridDNN((network3, network4), gsearcher, enableMultiGPU=enableMultiGPU);
            searcher = ReadConvolver.ReadConvolverHybridDNN((network3, network4), gsearcher);
        else:
            # searcher = ReadConvolver.ReadConvolverDNN(network3, gsearcher, enableMultiGPU=enableMultiGPU);
            searcher = ReadConvolver.ReadConvolverDNN(network3, gsearcher);
    elif (config2 is not None) and (config3 is None):
        config2 = importlib.import_module(config2).config;
        network0 = NType(config);
        network1 = NType(config);
        network2 = NType(config2);
        # searcher = GraphSearcher(network0, network1, network2, useAvg=useAvg, enableMultiGPU=enableMultiGPU, useDiff=useDiff);
        searcher = GraphSearcher(network0, network1, network2, useAvg=useAvg, useDiff=useDiff, useOneHot=useOneHot);
        logging.info("Creating graph searcher instance");
    else:
        raise NotImplementedError("No longer supported!");
        network = NType(config);
        searcher = Searcher(network);
        logging.info("Creating scorer instance");

    if optimizer == "Adam":
        logging.info("Using the Adam optimizer");
        optim = torch.optim.Adam(searcher.parameters(), lr=lr);
    else:
        logging.info("Using the SGD(R) optimizer");

        if optimizer == "SGDR":
            lr = maxLr;

        optim = torch.optim.SGD(searcher.parameters(), lr=lr, momentum=momentum, weight_decay=weightDecay);

    searcher = WrapperForDataParallel(searcher);

    if enableMultiGPU:
        searcher = torch.nn.DataParallel(searcher);

    if cuda:
        searcher.cuda();

    logging.info("Completed preparing data for training/validation iterations");

    prevLoss = float("inf");

    if useOneHot:
        qLossFn = torch.nn.BCEWithLogitsLoss();
    else:
        qLossFn = QLoss();

    totalLoss = 0;
    numIterLossDecrease = 0;
    scheduler = None;

    if lrScheduleFactor > 0:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=lrScheduleFactor, patience=0, cooldown=1, verbose=True, min_lr=minLr);

    if optimizer == "SGDR":
        scheduler = CosineAnnealingWarmRestarts(optim, T_0=T0, T_mult=Tmult, eta_min=minLr);

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
        tSamplerState = checkpoint['tSamplerState'];
        vSamplerState = checkpoint['vSamplerState'];
        tLoader.sampler.load_state_dict(tSamplerState);
        vLoader.sampler.load_state_dict(vSamplerState);
        torch.set_rng_state(seed);
        prevLoss = checkpoint['prevloss'];
        if 'randomState' in checkpoint:
            random.setstate(checkpoint['randomState']);
        if 'numIterLossDecrease' in checkpoint:
            numIterLossDecrease = checkpoint['numIterLossDecrease'];

        # If we are in the training iteration, restore sampler index number
        if itertype == "train":
            tLoader.sampler.nextIdx = batchSize * batchStart;
        elif itertype == "val":
            vLoader.sampler.nextIdx = batchSize * batchStart;

        if 'lr_scheduler_checkpoint' in checkpoint:
            scheduler.load_state_dict(checkpoint['lr_scheduler_checkpoint']);
        else:
            assert(scheduler is None), "Need an lr-scheduler, but none found in checkpoint!";

        searcherRaw = searcher.dnn if not enableMultiGPU else searcher.module.dnn;

        # Restore the individual networks from the checkpoint (so wrapper is created correctly)
        if config3 is not None:
            if tLoader.dataset.hybrid:
                network3, network4 = searcherRaw.network0, searcherRaw.network1;
                graphSearcher = searcherRaw.network2;
                network0, network1, network2 = \
                    graphSearcher.network0, graphSearcher.network1, graphSearcher.network2;
            else:
                readConvolver = searcherRaw.network0;
                graphSearcher = searcherRaw.network1;
                network0, network1, network2 = \
                    graphSearcher.network0, graphSearcher.network1, graphSearcher.network2;
                network3 = readConvolver;
        else:
            network0, network1, network2 = \
                searcherRaw.network0, searcherRaw.network1, searcherRaw.network2;
        logging.info(
            "Resuming %s iteration from the %d-th epoch, %d-th batch" % (itertype, epochStart, batchStart)
        );
    else:
        if pretrained is not None:
            searcher = torch.load(pretrained, map_location='cpu');
            assert(onlyEval), "Pretrained model is accepted only for single run evaluations, not for training!";
            if cuda:
                searcher = searcher.cuda();

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
            'tSamplerState': tLoader.sampler.state_dict(),
            'vSamplerState': vLoader.sampler.state_dict(),
            'randomState': random.getstate(),
            'numIterLossDecrease': numIterLossDecrease,
        };

        if scheduler is not None:
            checkpoint['lr_scheduler_checkpoint'] = scheduler.state_dict();

        torch.save(checkpoint, outputPrefix + ".checkpoint");

        logging.info("Performed checkpointing; epoch: %d, batch: %d, itertype: %s" % (epoch, batch, itertype));

    numParams = countParams(searcher);

    logging.info("Starting training of model with %d parameters" % numParams);

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

        for iterType in itertypeList:
            totalLoss = 0;
            totalQ = 0;
            loader = tLoader if iterType == "train" else vLoader;
            searcher.train(iterType == "train");

            # At the start of each training iteration, shuffle the indices
            # Currently, we do not shuffle the indices for the validation iteration
            # Note, this shuffling happens only if we are starting at iteration 0, because
            # in this case, it means either we didn't restore from a checkpoint, or
            # we simply checkpointed at the start of the last validation iteration (and not
            # during the training iteration. This means that the indices have not been shuffled for
            # the training iteration in this epoch)
            if (iterType == "train") and (batchStart == 0):
                tLoader.sampler.shuffle();

            # Checkpoint after every validation iteration
            if (iterType == "val"):
                performCheckpoint(j, 0, "val", prevLoss);

            i_ = 0;

            def numLocationsInBatch(batch):
                tensors = batch[0];
                labels = batch[1];
                numReadsPerAllele = batch[2];
                numAllelesPerSite = batch[3];
                return numAllelesPerSite.shape[0];

            def numAllelesInBatch(batch):
                numAllelesPerSite = batch[3];
                return sum(numAllelesPerSite.cpu().data.numpy().flatten().tolist());

            loaderIter = iter(loader);

            numCorrect = 0;
            numLabels = 0;

            while True:
                collectedBatches = [];
                lastIteration = False;
                dummyPadLength = 0;
                dummyPadLengthAlleles = 0;

                try:
                    for _ in range(len(devices)):
                        collectedBatches.append(next(loaderIter));
                        i_ += 1;
                except StopIteration:
                    logging.info("Completed epoch");
                    if len(collectedBatches) == 0:
                        break;
                    lastIteration = True;
                    if len(collectedBatches) < len(devices):
                        for i in range(len(devices) - len(collectedBatches)):
                            dummyPadLength += numLocationsInBatch(collectedBatches[-1]);
                            dummyPadLengthAlleles += numAllelesInBatch(collectedBatches[-1]);
                            collectedBatches += [collectedBatches[-1]];

                i = i_ + batchStart;

                assert(cuda);

                batches = [];
                labels = [];
                numAllelesPerSiteAll = [];

                for batch in collectedBatches:
                    tensors = batch[0];
                    labels.append(batch[1]);
                    numReadsPerAllele = batch[2];
                    numAllelesPerSite = batch[3];

                    batchDict = {
                        'tensors': tensors,
                        'numReadsPerAllele': numReadsPerAllele,
                        'numAllelesPerSite': numAllelesPerSite,
                    };

                    if useReadDepth:
                        numReadsPerSite = batch[4];
                        batchDict['depthMultiplier'] = numReadsPerSite;

                    if multiplierMode is not None:
                        batchDict['multiplierMode'] = multiplierMode;

                    batches.append(batchDict);
                    numAllelesPerSiteAll.append(numAllelesPerSite);

                numAllelesPerSiteAll = torch.cat(numAllelesPerSiteAll, dim=0).cpu().data.tolist();
                payload = Payload(devices, batches, listTypes=['numReadsPerAllele', 'numAllelesPerSite']);
                labels = torch.cat(labels, dim=0);
                labels = labels.cuda(non_blocking=True);
                labelsPerSite = uncollate_fn(labels, numAllelesPerSiteAll);

                if dummyPadLength > 0:
                    labelsPerSite = labelsPerSite[:-dummyPadLength];

                if iterType == "train":
                    # SGDR cosine annealing is called after each batch
                    if optimizer == "SGDR":
                        scheduler.step(j + i / len(tLoader));

                    # Test code for checking whether SGDR really works
                    # for param_group in optim.param_groups:
                    #     print("Learning rate = %f", param_group['lr']);

                    results = searcher(payload);

                    if not useOneHot:
                        resultsPerSite = uncollate_fn(results, numAllelesPerSiteAll);

                        if dummyPadLength > 0:
                            resultsPerSite = resultsPerSite[:-dummyPadLength];

                        losses = qLossFn(resultsPerSite, labelsPerSite);
                    else:
                        if dummyPadLengthAlleles > 0:
                            labels = labels[:-dummyPadLengthAlleles];
                            results = results[:-dummyPadLengthAlleles];

                        labels = (labels > 0.4).float();
                        losses = qLossFn(torch.squeeze(results, dim=1), labels);
                        numCorrect += countNumCorrect(labels, torch.squeeze(results, dim=1));
                        numLabels = labels.shape[0];

                    optim.zero_grad();
                    losses.backward();
                    optim.step();
                else:
                    with torch.no_grad():
                        results = searcher(payload);

                        if not useOneHot:
                            resultsPerSite = uncollate_fn(results, numAllelesPerSiteAll);

                            if dummyPadLength > 0:
                                resultsPerSite = resultsPerSite[:-dummyPadLength];

                            losses = qLossFn(resultsPerSite, labelsPerSite);
                            tlosses = qLossFn(labelsPerSite, labelsPerSite);  # Always compute this, to act as a check against checkpointing issues
                        else:
                            if dummyPadLengthAlleles > 0:
                                labels = labels[:-dummyPadLengthAlleles];
                                results = results[:-dummyPadLengthAlleles];

                            labels = (labels > 0.4).float();
                            losses = qLossFn(torch.squeeze(results, dim=1), labels);
                            tlosses = torch.Tensor([0]);
                            numCorrect += countNumCorrect(labels, torch.squeeze(results, dim=1));
                            numLabels += labels.shape[0];

                floss = float(losses.cpu().data.numpy().flatten()[0]);

                if not useOneHot:
                    totalLoss += floss * len(labelsPerSite);
                else:
                    totalLoss += floss * labels.shape[0];

                # Perform checkpoint every CHECKPOINT_FREQ-th (TRAIN) iteration
                if iterType == "train":
                    if (i > 0) and (i % CHECKPOINT_FREQ == 0):
                        performCheckpoint(j, i, iterType, prevLoss);

                if (iterType == 'val'):
                    tloss = float(tlosses.cpu().data.numpy().flatten()[0]);

                    if not useOneHot:
                        totalQ += tloss * len(labelsPerSite);
                    else:
                        totalQ += 0;  # tloss is 0 when one-hot encoding is used

                if i % TRAIN_MESSAGE_INTERVAL == 0:
                    logging.info("Completed %d-th %s iteration, loss = %f" % (i, iterType, floss));

                if lastIteration:
                    break;

            if not useOneHot:
                totalLoss /= lTrain if iterType == "train" else lVal;
            else:
                totalLoss /= numLabels;
                pctCorrect = numCorrect / numLabels;

            # Reset batch start
            batchStart = 0;

            if iterType == 'val':
                if maxQ is not None:
                    assert(abs(maxQ - totalQ / lVal) < 1e-6), \
                        "Possible checkpointing problem, original = %f, new = %f" % (maxQ, totalQ / lVal);
                else:
                    maxQ = totalQ / lVal;

        # Set to None, so that checkpoint's effects are gone
        itertype = None;

        # learning-rate schedule (this is after the validation iteration)
        if scheduler is not None:
            scheduler.step(totalLoss);

        if prevLoss > totalLoss:
            logging.info("Model improves in iteration (%d), saving model; total loss = %f, best loss = %f" % (j, totalLoss, maxQ));
            prevLoss = totalLoss;

            if not onlyEval:
                if config2 is None:
                    raise NotImplementedError("Not longer supported");
                    scoringWrapper = ScoringWrapper(network, padlength=padlength);
                elif config3 is None:
                    scoringWrapper = GraphSearcherWrapper(
                        network0, network1, network2,
                        padlength=padlength, useAvg=useAvg, useDiff=useDiff,
                    );
                    network = searcher;  # Allow saving of scorer, not network in this case
                else:
                    if tLoader.dataset.hybrid:
                        graphSearcherWrapper = GraphSearcherWrapper(
                            network0, network1, network2, useAvg=useAvg, padlength=-1, useDiff=useDiff,
                        );
                        scoringWrapper = ReadConvolver.ReadConvolverHybridWrapper((network3, network4), graphSearcherWrapper);
                        network = searcher;
                    else:
                        graphSearcherWrapper = GraphSearcherWrapper(
                            network0, network1, network2, useAvg=useAvg, padlength=-1, useDiff=useDiff,
                        );
                        scoringWrapper = ReadConvolver.ReadConvolverWrapper(network3, graphSearcherWrapper);
                        network = searcher;

                if not keepAll:
                    # Overwrite previous model in this case
                    torch.save(scoringWrapper, os.path.abspath(outputPrefix + ".wrapper.dnn"));
                    torch.save(network, os.path.abspath(outputPrefix + ".dnn"));
                else:
                    # Write a new model (using epoch number as tag)
                    torch.save(scoringWrapper, os.path.abspath(outputPrefix + ".%d.wrapper.dnn" % j));
                    torch.save(network, os.path.abspath(outputPrefix + ".%d.dnn" % j));

                numIterLossDecrease = 0;
        else:
            logging.info("Model fails to improve in curent iteration (%d); total loss = %f" % (j, totalLoss));
            numIterLossDecrease += 1;
            if (not overfit) and (numIterLossDecrease >= numEarlyStopIterations):
                logging.info("Ending training");
                break;

    logging.info("Theoretical Max Q in validation set = %f, achieved best validation Q = %f" % (maxQ, prevLoss));

    if onlyEval and useOneHot:
        logging.info("Accuracy = %f" % pctCorrect);


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
        "--useAsMemmap",
        help="Indicates that provided --data should be used as memmap index file",
        default=False,
        action="store_true",
    );

    parser.add_argument(
        "--config",
        help="Config for DNN",
        default="initConfig",
        required=False,
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
        default=False,
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
        "--forClassification",
        help="Train model for classification",
        action="store_true",
        default=False,
    );

    parser.add_argument(
        "--networkType",
        help="Type of network to use",
        choices=["DNN", "DNNL"],
        default="DNN",
    );

    parser.add_argument(
        "--padlength",
        help="Length to which feature maps should be padded",
        default=0,
        type=int,
    );

    parser.add_argument(
        "--useOld",
        action="store_true",
        default=False,
        help="Use old classifier (MLP)",
    );

    parser.add_argument(
        "--homozygousFiltered",
        help="Homozygous sites have been filtered in dataset",
        action="store_true",
        default=False,
    );

    parser.add_argument(
        "--subtractCoverage",
        help="Subtract local coverage from feature maps for training",
        action="store_true",
        default=False,
    );

    parser.add_argument(
        "--avg_cov",
        type=int,
        default=1,
        help="Provide avg cov to normalize features with respect to this (DNNs train better with values in [0, 1])",
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
        "--config2",
        help="Second config for Graph-type searcher",
        default=None,
    );

    parser.add_argument(
        "--config3",
        help="Apply a read convolver to input features",
        default=None,
    );

    parser.add_argument(
        "--useAvg",
        help="Use average for other alleles' features for Graph-type CNN",
        default=False,
        action="store_true",
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
        "--pretrained",
        help="Load pretrained model",
        default=None,
    );

    parser.add_argument(
        "--pretrainedVal",
        help="Validation accuracy of pretrained model",
        type=float,
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
        "--useDiff",
        help="Use difference instead of concatenation across alleles",
        action="store_true",
        default=False,
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
        "--useReadDepth",
        help="Whether read depth normalization should be used",
        default=False,
        action="store_true",
    );

    parser.add_argument(
        "--subsampleFile",
        help="File containing subsample statistics",
        default=None,
    );

    parser.add_argument(
        "--advise",
        help="Use madvise to pre-empt the kernel",
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
        "--presubsample",
        help="Perform subsampling statically",
        default=False,
        action="store_true",
    );

    parser.add_argument(
        "--presubsampleTag",
        required=False,
        help="Provide a tag for pre-subsampling"
    );

    parser.add_argument(
        "--fractionalCutoff",
        help="If GT any allele does not pass this fractional cutoff, we do not use the example for training",
        default=0.0,
        type=float,
    );

    parser.add_argument(
        "--useOneHot",
        help="Use one-hot encoded outputs (instead of softmax-encoded)",
        default=False,
        action="store_true"
    );

    parser.add_argument(
        "--onlyEval",
        help="Only perform evaluation",
        default=False,
        action="store_true",
    );

    args = parser.parse_args();

    if args.onlyEval:
        args.numEpochs = 1;
        assert(args.pretrained is not None), "Provide a pretrained model for evaluation";

    if args.useOneHot:
        args.pruneHomozygousIndices = False;  # Disable pruning homozygous indices, when one-hot is used

    if args.presubsample:
        assert(args.presubsampleTag is not None), "Provide pressubsampleTag";

    logging.basicConfig(level=(logging.INFO if not args.debug else logging.DEBUG), format='%(asctime)-15s %(message)s');

    if args.test:
        deterministicBackEnd();
        TRAIN_MESSAGE_INTERVAL = 1;

    if args.subsampleFile is not None:
        fhandle = open(args.subsampleFile, 'r');
        subsampleRates = ast.literal_eval(' '.join(fhandle.readlines()));
        subsampler = Subsampler(subsampleRates);
    else:
        subsampler = None;

    CHECKPOINT_FREQ = args.checkPointFreq;

    if args.forClassification:
        trainClassifier(
            numEpochs=args.numEpochs,
            numWorkers=args.numWorkers,
            batchSize=args.batchSize,
            lr=args.lr,
            configFile=args.config,
            data=args.data,
            cuda=args.cuda,
            outputPrefix=args.outputPrefix,
            overfit=args.overfit,
            padlength=args.padlength,
            useOld=args.useOld,
            filterHomozygous=(not args.homozygousFiltered),
            optimizer=args.optimizer,
            momentum=args.momentum,
            weightDecay=args.weightDecay,
            numEarlyStopIterations=args.numEarlyStopIterations,
        );
    else:
        dataloader = dataLoader(
            numWorkers=args.numWorkers,
            batchSize=args.batchSize,
            hdf5=args.data,
            overfit=args.overfit,
            padLength=args.padlength,
            subtractCoverage=args.subtractCoverage,
            avg_cov=args.avg_cov,
            findMaxQ=True,
            advanced=(args.config3 is not None),
            useAsMemmap=args.useAsMemmap,
            useBlockSampler=args.useBlockSampler,
            blockSize=args.blockSize,
            pruneHomozygous=args.pruneHomozygous,
            subsampler=subsampler,
            advise=args.advise,
            valData=args.valData,
            presubsample=args.presubsample,
            presubsampleTag=args.presubsampleTag,
            fractionalCutoff=args.fractionalCutoff,
        );
        train(
            numEpochs=args.numEpochs,
            numWorkers=args.numWorkers,
            batchSize=args.batchSize,
            lr=args.lr,
            configFile=args.config,
            data=args.data,
            cuda=args.cuda,
            outputPrefix=args.outputPrefix,
            overfit=args.overfit,
            networkType=args.networkType,
            padlength=args.padlength,
            subtractCoverage=args.subtractCoverage,
            avg_cov=args.avg_cov,
            optimizer=args.optimizer,
            momentum=args.momentum,
            weightDecay=args.weightDecay,
            numEarlyStopIterations=args.numEarlyStopIterations,
            lrScheduleFactor=args.lrFactor,
            config2=args.config2,
            useAvg=args.useAvg,
            config3=args.config3,
            useAsMemmap=args.useAsMemmap,
            dataloader=dataloader,
            checkpoint=args.checkpoint,
            pretrained=args.pretrained,
            pretrainedVal=args.pretrainedVal,
            checkpointArchive=args.checkpointArchive,
            enableMultiGPU=args.useMultiGPU,
            minLr=args.minLr,
            useDiff=args.useDiff,
            useReadDepth=args.useReadDepth,
            multiplierMode=args.depthMode,
            keepAll=args.keepAll,
            useOneHot=args.useOneHot,
            maxLr=args.maxLr,
            T0=args.T0,
            Tmult=args.Tmult,
            onlyEval=args.onlyEval,
        );
