import torch
import itertools
import ReadConvolver
import AlleleSearcherDNN
import importlib
import logging
import collections
import math
import copy
import numpy as np
from functools import reduce


def reduceFrames(frames):
    return torch.stack(
        list(
            torch.sum(frame, dim=0) for frame in frames
        ),
        dim=0
    );


def reduceSlots(d, slots):
    if type(slots) is list:
        slots = torch.LongTensor(slots);
        if d.is_cuda:
            slots = slots.cuda(device=d.get_device());

    results = torch.cumsum(d, dim=0);
    indices = torch.cumsum(slots, dim=0) - 1;
    zeroPad = torch.unsqueeze(torch.zeros_like(d[0]), dim=0);
    resultsSelected = results[indices];
    paddedSelections = torch.cat((zeroPad, resultsSelected[:-1]), dim=0);
    return resultsSelected - paddedSelections;


class Inception(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__();
        branches = kwargs['branches'];
        self.numBranches = len(branches);

        for i, branch in enumerate(branches):
            branchNetwork = AlleleSearcherDNN.Network(branch);
            setattr(self, 'branch%d' % i, branchNetwork);

    def forward(self, tensor):
        branchResults = [];

        for i in range(self.numBranches):
            branch = getattr(self, 'branch%d' % i);
            branchResults.append(branch(tensor));

        return torch.cat(branchResults, dim=1);


class ConvCombiner(torch.nn.Module):
    def __init__(self, config):
        super().__init__();
        self.network = AlleleSearcherDNN.Network(config);

    def forward(self, conv1, conv2):
        convInput = torch.cat((conv1, conv2), dim=1);
        return self.network(convInput);


torch.nn.Inception = Inception;


class DummyGraphNetwork(torch.nn.Module):
    """
    A dummy graph tensor that always returns a negative label
    for any input
    """
    def __init__(self):
        super().__init__();
        self.register_buffer(
            'baseTensor',
            torch.Tensor([-20])
        );

    def forward(self, *args, **kwargs):
        tensors = args[0];
        return torch.unsqueeze(self.baseTensor.repeat(tensors.shape[0]), dim=1);


class MoEMergedAdvanced(torch.nn.Module):
    """
    Mixture-of-experts DNN with read convolution parameter sharing
    """
    def __init__(
        self,
        readConv0,
        readConv1,
        alleleConv0,
        alleleConv1,
        experts,
        meta,
        detachMeta=False,
        detachExperts=False,
        useSeparateMeta=False,
        useAdditive=False,
        alleleConvCombiner=None,
        siteConvCombiner=None,
    ):
        """
        :param readConv0: AlleleSearcherDNN.Network
            A read convolver module (tech0)

        :param readConv1: AlleleSearcherDNN.Network
            A read convolver module (tech1)

        :param alleleConv0: AlleleSearcherDNN.Network
            Allele-level convolver (tech0)

        :param alleleConv1: AlleleSearcherDNN.Network
            Allele-level convolver (tech1)

        :param experts: tuple
            Graph-searcher instances for each expert

        :param meta: torch.Module
            Meta-expert (not MetaExpert class above, but MetaExpert.network2)

        :param detachMeta: bool
            Whether meta expert should be detached from training the downstream features

        :param detachExperts: bool
            Remove experts' parameters from training

        :param useSeparateMeta: bool
            Use separate read convolvers for meta-expert

        :param useAdditive: bool
            Features are combined using addition/subtraction instead of
            concatenation

        :param alleleConvCombiner: torch.nn.Module
            Combine two convolvers into a single one

        :param siteConvCombiner: torch.nn.Module
            Combine two site-level convolvers together
        """
        super().__init__();
        self.readConv0 = readConv0;
        self.readConv1 = readConv1;
        self.alleleConv0 = alleleConv0;
        self.alleleConv1 = alleleConv1;
        self.expert0 = experts[0];
        self.expert1 = experts[1];
        self.expert2 = experts[2];
        self.meta = meta;
        self.detachMeta = detachMeta;
        self.detachExperts = detachExperts;
        self.useAdditive = useAdditive;
        self.alleleConvCombiner = alleleConvCombiner;
        self.siteConvCombiner = siteConvCombiner;

        if useSeparateMeta:
            logging.info("Will use separate read convolvers for the meta-expert");
            self.readConv0Meta = copy.deepcopy(readConv0);
            self.readConv1Meta = copy.deepcopy(readConv1);

    def performAlleleConv(self, readLevelConv0, readLevelConv1, numReadsPerAllele):
        reducedFramesPerAllele0 = self.alleleConv0(
            reduceSlots(readLevelConv0, numReadsPerAllele[0])
        );
        reducedFramesPerAllele1 = self.alleleConv1(
            reduceSlots(readLevelConv1, numReadsPerAllele[1])
        );
        return reducedFramesPerAllele0, reducedFramesPerAllele1;

    def preparePerSiteFramesFromReads(self, conv0, conv1, numAllelesPerSite, numReadsPerAllele):
        perAlleleFrames0 = torch.split(
            conv0,
            split_size_or_sections=numReadsPerAllele[0]
        );
        perAlleleFrames1 = torch.split(
            conv1,
            split_size_or_sections=numReadsPerAllele[1]
        );

        def reduceArrays(array):
            return torch.cat(array, dim=0);

        endBoundaries = np.cumsum(numAllelesPerSite).tolist();
        beginBoundaries = [0] + endBoundaries[:-1];

        perSiteFrames0 = [
            reduceArrays(perAlleleFrames0[x: y]) for x, y in zip(beginBoundaries, endBoundaries)
        ];
        perSiteFrames1 = [
            reduceArrays(perAlleleFrames1[x: y]) for x, y in zip(beginBoundaries, endBoundaries)
        ];

        return reduceFrames(perSiteFrames0), reduceFrames(perSiteFrames1);

    def preparePerSiteFrames(self, postAlleleLevelConv, numAllelesPerSite):
        return reduceSlots(postAlleleLevelConv, numAllelesPerSite);

    def expertComputations(self, alleleLevelConv, perSiteFrames, numAllelesPerSiteTensor, expertIdx):
        def computations():
            remainingAllelesSignal = torch.repeat_interleave(
                perSiteFrames, numAllelesPerSiteTensor, dim=0
            ) - alleleLevelConv;

            if hasattr(self, 'useAdditive') and self.useAdditive:
                expertInput = alleleLevelConv - remainingAllelesSignal;
            else:
                expertInput = torch.cat((alleleLevelConv, remainingAllelesSignal), dim=1);

            return getattr(self, 'expert%d' % expertIdx)(expertInput);

        if hasattr(self, 'detachExperts') and self.detachExperts:
            with torch.no_grad():
                return computations();
        else:
            return computations();

    def metaExpertComputations(self, perSiteFrames):
        if hasattr(self, 'detachMeta') and self.detachMeta:
            perSiteFrames_ = perSiteFrames.clone().detach();
            return self.meta(perSiteFrames_);
        else:
            return self.meta(perSiteFrames);

    def forward(self, tensors, numAllelesPerSite, numReadsPerAllele, *args, **kwargs):
        # 1. Perform read-level convolutions
        readLevelConv0 = self.readConv0(tensors[0].float());
        readLevelConv1 = self.readConv1(tensors[1].float());

        # 2. Perform allele-level convolutions
        alleleLevelConv0, alleleLevelConv1 = self.performAlleleConv(
            readLevelConv0, readLevelConv1, numReadsPerAllele
        );

        if hasattr(self, 'useAdditive') and self.useAdditive:
            if self.alleleConvCombiner is None:
                alleleLevelConv2 = alleleLevelConv0 + alleleLevelConv1;
            else:
                alleleLevelConv2 = self.alleleConvCombiner(
                    alleleLevelConv0, alleleLevelConv1
                );
        else:
            alleleLevelConv2 = torch.cat(
                (alleleLevelConv0, alleleLevelConv1), dim=1
            );

        # 3. Determine per-site frames
        perSiteFrame0 = self.preparePerSiteFrames(alleleLevelConv0, numAllelesPerSite);
        perSiteFrame1 = self.preparePerSiteFrames(alleleLevelConv1, numAllelesPerSite);

        if hasattr(self, 'useAdditive') and self.useAdditive:
            if self.siteConvCombiner is None:
                # perSiteFrame2 = perSiteFrame0 + perSiteFrame1;
                # Note: This is conceptually better than the previous solution
                # This makes all three experts use the same type of computations,
                # A_j - \sum_{l != j} A_l, to produce input feature maps
                perSiteFrame2 = self.preparePerSiteFrames(alleleLevelConv2, numAllelesPerSite);
            else:
                perSiteFrame2 = self.siteConvCombiner(
                    perSiteFrame0, perSiteFrame1,
                );
        else:
            perSiteFrame2 = torch.cat(
                (perSiteFrame0, perSiteFrame1),
                dim=1
            );

        if hasattr(self, 'readConv0Meta'):
            readLevelConv0Meta = self.readConv0Meta(tensors[0].float());
            readLevelConv1Meta = self.readConv1Meta(tensors[1].float());
            perSiteFrames0Meta, perSiteFrames1Meta = self.preparePerSiteFramesFromReads(
                readLevelConv0Meta,
                readLevelConv1Meta,
                numAllelesPerSite,
                numReadsPerAllele
            );
            if hasattr(self, 'useAdditive') and self.useAdditive:
                if self.siteConvCombiner is None:
                    perSiteFrameMeta = perSiteFrames0Meta + perSiteFrames1Meta;
                else:
                    perSiteFrameMeta = self.siteConvCombiner(
                        perSiteFrames0Meta, perSiteFrames1Meta
                    );
            else:
                perSiteFrameMeta = torch.cat(
                    (perSiteFrames0Meta, perSiteFrames1Meta),
                    dim=1
                );
        else:
            perSiteFrameMeta = perSiteFrame2;

        # 4. Prepare alleles per site tensor
        numAllelesPerSiteTensor = torch.LongTensor(numAllelesPerSite);
        if perSiteFrame0.is_cuda:
            numAllelesPerSiteTensor = numAllelesPerSiteTensor.cuda(device=perSiteFrame0.get_device());

        # 5. Perform expert computations
        ngsPredictions = self.expertComputations(alleleLevelConv0, perSiteFrame0, numAllelesPerSiteTensor, 0);
        tgsPredictions = self.expertComputations(alleleLevelConv1, perSiteFrame1, numAllelesPerSiteTensor, 1);
        hybPredictions = self.expertComputations(alleleLevelConv2, perSiteFrame2, numAllelesPerSiteTensor, 2);

        # 6. Perform meta-expert computations
        meta = torch.nn.functional.softmax(self.meta(perSiteFrameMeta), dim=1);

        return [ngsPredictions, tgsPredictions, hybPredictions], meta;


class MoEMergedWrapperAdvanced(torch.nn.Module):
    def __init__(self, moeMerged, providePredictions=False):
        super().__init__();
        self.moeMerged = moeMerged;
        self.providePredictions = providePredictions;

    def _singleFeatureDictData(self, featureDict):
        # Convert items in single feature dict into format for MoEMergedAdvanced
        alleles = list(featureDict.keys());

        numAllelesPerSite = [len(alleles)];
        numReadsPerAllele0 = [featureDict[key][0].shape[0] for key in alleles];
        numReadsPerAllele1 = [featureDict[key][1].shape[0] for key in alleles];
        tensors0 = torch.cat(
            [torch.transpose(featureDict[key][0], 1, 2) for key in alleles], dim=0
        );
        tensors1 = torch.cat(
            [torch.transpose(featureDict[key][1], 1, 2) for key in alleles], dim=0
        );

        return alleles, (tensors0, tensors1), numAllelesPerSite, (numReadsPerAllele0, numReadsPerAllele1);

    def forward(self, featureDict):
        nnInputs = self._singleFeatureDictData(featureDict);
        alleles = nnInputs[0];
        results = self.moeMerged(*nnInputs[1:]);
        experts, meta = results;
        meta = meta[0];  # Remove unnecessary batch dimension
        experts = [torch.squeeze(torch.sigmoid(e), dim=1) for e in experts];
        alleleNumbers = dict({a: i for i, a in enumerate(alleles)});

        def invert(pairing):
            return (pairing[1], pairing[0]);

        def expertProbability(prediction, target):
            return torch.exp(
                torch.sum(
                    torch.log(prediction * target + (1 - prediction) * (1 - target) + 1e-10)
                )
            );

        def siteLogProb(e_, m_):
            m_ = torch.log(m_ + 1e-10);
            joint = m_ + e_;
            maxVal = torch.max(joint);
            jointNormed = joint - maxVal;
            results = torch.log(torch.sum(torch.exp(jointNormed))) + maxVal;
            maxProb = results.clone().fill_(math.log(1 - 1e-8));
            return torch.min(results, maxProb);

        def getExpertPredictions(expertId):
            predictionQualities = dict();

            for allelePairing in itertools.product(alleles, alleles):
                if (allelePairing in predictionQualities) or (invert(allelePairing) in predictionQualities):
                    continue;
                targetTensor = torch.zeros(len(alleles));
                targetTensor[alleleNumbers[allelePairing[0]]] = 1;
                targetTensor[alleleNumbers[allelePairing[1]]] = 1;
                expertProb = expertProbability(experts[expertId], targetTensor);
                predictionQualities[allelePairing] = expertProb;

            return predictionQualities;

        def getPredictionDictionary():
            predictionQualities = dict();
            expert0Prediction = getExpertPredictions(0);
            expert1Prediction = getExpertPredictions(1);
            expert2Prediction = getExpertPredictions(2);

            for allelePairing in expert0Prediction:
                predictionQualities[allelePairing] = (
                    meta[0] * expert0Prediction[allelePairing] + meta[1] * expert1Prediction[allelePairing] + meta[2] * expert2Prediction[allelePairing]
                );

            return predictionQualities, expert0Prediction, expert1Prediction, expert2Prediction;

        if self.providePredictions:
            return tuple(list(getPredictionDictionary()) + [meta]);
        else:
            return getPredictionDictionary()[0];


def createMoEFullMergedAdvancedModel(configDict, useSeparateMeta=False):
    ngsConvolver = AlleleSearcherDNN.Network(importlib.import_module(configDict['readConv']).config);
    tgsConvolver = AlleleSearcherDNN.Network(importlib.import_module(configDict['readConv']).config);
    meta = AlleleSearcherDNN.Network(importlib.import_module(configDict['meta']).config);

    alleleConvNgs = AlleleSearcherDNN.Network(importlib.import_module(configDict['alleleConvSingle']).config);
    alleleConvTgs = AlleleSearcherDNN.Network(importlib.import_module(configDict['alleleConvSingle']).config);
    graphConvNgs = AlleleSearcherDNN.Network(importlib.import_module(configDict['graphConvSingle']).config);
    graphConvTgs = AlleleSearcherDNN.Network(importlib.import_module(configDict['graphConvSingle']).config);

    # It is possible to avoid using graph convolver hybrid; in
    # this case, use a dummy module that always outputs a small value
    graphConvHybConf = importlib.import_module(configDict['graphConvHybrid']).config;

    if len(graphConvHybConf) > 0:
        graphConvHyb = AlleleSearcherDNN.Network(graphConvHybConf);
    else:
        graphConvHyb = DummyGraphNetwork();

    if 'convCombiner' in configDict:
        alleleConvCombiner = ConvCombiner(importlib.import_module(configDict['convCombiner']).config);

        # New code to force useAdditive only on alleleLevelCombiner; 2020/02/12
        if ('noSiteLevelCombiner' in configDict) and configDict['noSiteLevelCombiner']:
            siteConvCombiner = None;
        else:
            siteConvCombiner = ConvCombiner(importlib.import_module(configDict['convCombiner']).config);
        # 2020/02/12 over
    else:
        alleleConvCombiner = None;
        siteConvCombiner = None;

    if 'kwargs' in configDict:
        kwargs = configDict['kwargs'];
    else:
        kwargs = dict();

    moe = MoEMergedAdvanced(
        ngsConvolver,
        tgsConvolver,
        alleleConvNgs,
        alleleConvTgs,
        (graphConvNgs, graphConvTgs, graphConvHyb),
        meta,
        useSeparateMeta=useSeparateMeta,
        alleleConvCombiner=alleleConvCombiner,
        siteConvCombiner=siteConvCombiner,
        **kwargs
    );

    return moe;


def createMoEFullMergedAdvancedModelWrapper(moe):
    return MoEMergedWrapperAdvanced(moe);
