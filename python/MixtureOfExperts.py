import torch
import itertools
import ReadConvolver
import AlleleSearcherDNN
import importlib
import logging
import collections
import math
import numpy as np
import sys

PREDICTIVE_OFFSET = math.log(1e-8 / (1 - 1e-8));
np.set_printoptions(threshold=sys.maxsize);

try:
    profile
except Exception:
    def profile(x):
        return x;

ExpertBundle = collections.namedtuple('ExpertBundle', ['ngs', 'tgs', 'hyb']);


def reduceFrames(frames):
    return torch.stack(
        list(
            torch.sum(frame, dim=0) for frame in frames
        ),
        dim=0
    );


class ConditionalNetwork(AlleleSearcherDNN.Network):
    def __init__(self, config, offset=0):
        super().__init__(config);
        self.offset = offset;

    def forward(self, tensors):
        """
        Conditionally returns a zero tensor (with offset) if an input
        tensor is zero. Otherwise returns the result of convolution.
        """
        # Determine input tensors that are zero-padded
        conditioner = (tensors == 0);

        for i in range(1, len(tensors.shape)):
            conditioner = torch.prod(
                conditioner,
                dim=i, keepdim=True,
            );  # [batch, 1, 1, ... #dims - 1];

        # Compute network output, and create dummy zeros
        results = self.network(tensors);
        zeros = results.new_zeros(results.shape, requires_grad=False) + self.offset;

        # Prune the conditioner to fit the number of dimensions in results
        numPrune = len(conditioner.shape) - len(zeros.shape);

        for i in range(numPrune):
            conditioner = torch.squeeze(conditioner, dim=-1);

        # Multiplex the correct output based on whether an input
        # tensor has zeros or not
        return torch.where(
            conditioner.byte(),
            zeros,
            results,
        );


class IlluminaExpert(torch.nn.Module):
    """
    Wrapper class for ReadConvolver that takes only the left half of all inputs
    """
    def __init__(self, network):
        super().__init__();
        self.network = network;

    def forward(self, tensors, numAllelesPerSite, numReadsPerAllele):
        return self.network(
            tensors[0],
            numAllelesPerSite,
            numReadsPerAllele[0]
        );


class PacBioExpert(torch.nn.Module):
    """
    Wrapper class for ReadConvolver that takes only the right half of all inputs
    """
    def __init__(self, network):
        super().__init__();
        self.network = network;

    def forward(self, tensors, numAllelesPerSite, numReadsPerAllele):
        return self.network(
            tensors[1],
            numAllelesPerSite,
            numReadsPerAllele[1]
        );


class MetaExpertWrapper(torch.nn.Module):
    """
    Wrapper for meta-expert network for inference
    """
    def __init__(self, network0, network1, network2):
        super().__init__();
        self.network0 = network0;
        self.network1 = network1;
        self.network2 = network2;

    def forward(self, featureDict):
        features0 = [];
        features1 = [];

        for allele, feature in featureDict.items():
            f0, f1 = feature;
            features0.append(f0);
            features1.append(f1);

        features0 = torch.transpose(torch.cat(features0, dim=0), 1, 2);
        features1 = torch.transpose(torch.cat(features1, dim=0), 1, 2);
        readLevelConv0 = self.network0(features0);
        readLevelConv1 = self.network1(features1);
        reducedFrames0 = torch.sum(readLevelConv0, dim=0, keepdim=True);
        reducedFrames1 = torch.sum(readLevelConv1, dim=0, keepdim=True);

        return torch.nn.functional.softmax(
            self.network2(
                torch.cat((reducedFrames0, readucedFrames1), dim=1)
            ),
            dim=1
        );


class MetaExpert(torch.nn.Module):
    """
    Meta-Expert class for Mixture-of-Experts hybrid model
    """
    def __init__(self, network0, network1, network2):
        """
        :param network0: AlleleSearcherDNN.Network
            Read-level convolutions

        :param network1: AlleleSearcherDNN.Network
            Read-level convolutions

        :param network2: AlleleSearcherDNN.Network
            Site-level combiner convolutions
        """
        super().__init__();
        self.network0 = network0;
        self.network1 = network1;
        self.network2 = network2;

    def forward(self, tensors, numReadsPerSite):
        assert((type(tensors) is tuple) or (type(tensors) is list));
        assert(len(tensors) == 2);
        readLevelConv0 = self.network0(tensors[0].float());
        readLevelConv1 = self.network1(tensors[1].float());

        # Per site results
        perSiteFrames0 = torch.split(
            readLevelConv0,
            split_size_or_sections=numReadsPerSite[0]
        );
        perSiteFrames1 = torch.split(
            readLevelConv1,
            split_size_or_sections=numReadsPerSite[1]
        );

        # Sum up per-site results
        reducedFrames0 = reduceFrames(perSiteFrames0);
        reducedFrames1 = reduceFrames(perSiteFrames1);

        reducedFrames = torch.cat(
            (reducedFrames0, reducedFrames1),
            dim=1
        );

        # Pass through the final stage neural network
        # Per-site predictions of which expert to pick
        results = torch.nn.functional.softmax(
            self.network2(reducedFrames), dim=1
        );

        logging.debug("Exiting Meta Expert");

        return results;


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

        return self.dnn(
            tensors,
            numAllelesPerSite,
            numReadsPerAllele,
            numReadsPerSite,
        );


class GraphSearcherMergedWrapper(torch.nn.Module):
    """
    A wrapper for the graph searcher DNN (merged version)

    :param network0: AlleleSearcherDNN.Network
        Determine allele-level features

    :param network1: AlleleSearcherDNN.Network
        Determine allele-vs-allele scoring
    """
    def __init__(self, network0, network1):
        super().__init__();
        self.network0 = network0;
        self.network1 = network1;

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

        for allele, features in featureDict.items():
            if type(features) is tuple:
                f = torch.cat(
                    (
                        torch.transpose(features[0], 0, 1),
                        torch.transpose(features[1], 0, 1)
                    ),
                    dim=0
                );
            else:
                f = torch.transpose(features, 0, 1);

            tensors.append(f);
            alleles.append(allele);

        tensors = torch.stack(tensors, dim=0);
        frames = self.network0(tensors);
        total = torch.sum(frames, dim=0, keepdim=True);
        framesOther = total - frames;
        siteLevelFrames = torch.cat((frames, framesOther), dim=1);
        # This is weird - but for the sake of compatibility
        finalResults = torch.transpose(self.network1(siteLevelFrames), 0, 1);

        return alleles, finalResults;


class GraphSearcherMerged(torch.nn.Module):
    """
    Graph Searcher DNN for comparing each allele to all others. Unlike AlleleSearcherDNN.GraphSearcher,
    here, two sets of DNNs aren't used for analyzing an allele
    """
    def __init__(self, network0, network1):
        """
        :param network0: AlleleSearcherDNN.Network
            Torch module to compute an allele's "potential" or feature map

        :param network1: AlleleSearcherDNN.Network
            Compares an allele to all other alleles
        """
        super().__init__();
        self.network0 = network0;
        self.network1 = network1;

    def forward(self, tensors, numAllelesPerSite):
        features = self.network0(tensors);
        perSiteFeatures = torch.split(
            features, split_size_or_sections=numAllelesPerSite
        );
        featuresCombined = [];

        # Compute feature sums at each site
        combinedFeaturesAtSite = torch.stack(
            [torch.sum(f, dim=0) for f in perSiteFeatures],
            dim=0
        );

        # Repeat feature sums for each allele at a site
        repeatNumbers = torch.LongTensor(numAllelesPerSite);

        if features.is_cuda:
            repeatNumbers = repeatNumbers.cuda(features.get_device());

        combinedFeaturesExpanded = torch.repeat_interleave(
            combinedFeaturesAtSite, repeatNumbers, dim=0
        );

        # Subtract individual allele's features to obtain sum of features
        # of "other" alleles
        featuresOther = combinedFeaturesExpanded - features;
        featuresCombined = torch.cat((features, featuresOther), dim=1);
        resultsCombined = self.network1(featuresCombined);

        return resultsCombined;


class MoE(torch.nn.Module):
    """
    Hybrid mixture-of-experts DNN
    """
    def __init__(self, experts, meta):
        super().__init__();
        self.numExperts = len(experts);

        for i, expert in enumerate(experts):
            setattr(self, 'expert%d' % i, expert);

        self.meta = meta;

    def forward(
        self,
        tensors,
        numAllelesPerSite,
        numReadsPerAllele,
        numReadsPerSite,
        *args,
        **kwargs,
    ):
        # Return expert and meta-expert predictions
        return [
            getattr(self, 'expert%d' % i)(tensors, numAllelesPerSite, numReadsPerAllele)
            for i in range(self.numExperts)
        ], self.meta(tensors, numReadsPerSite);


class MoEMergedWrapper(torch.nn.Module):
    def __init__(self, readConv0, readConv1, experts, meta):
        """
        :param readConv0: AlleleSearcherDNN.Network
            Convolve NGS reads

        :param readConv1: AlleleSearcherDNN.Network
            Convolve TGS reads

        :param experts: ExpertBundle
            GraphConvolverWrapper instances

        :param meta: AlleleSearcherDNN.Network
            Meta-expert network
        """
        super().__init__();
        self.readConv0 = readConv0;
        self.readConv1 = readConv1;
        self.expert0 = experts.ngs;
        self.expert1 = experts.tgs;
        self.expert2 = experts.hyb;
        self.meta = meta;

    def prepareMetaInputs(self, readLevelConv0, readLevelConv1):
        perSiteFrame0 = torch.sum(readLevelConv0, dim=0, keepdim=True);
        perSiteFrame1 = torch.sum(readLevelConv1, dim=0, keepdim=True);
        metaExpertInput = torch.cat(
            (perSiteFrame0, perSiteFrame1),
            dim=1,
        );
        return metaExpertInput;

    def prepareExpertInputs(self, readLevelConv0, readLevelConv1, numReadsPerAllele):
        perAlleleFrames0 = torch.split(
            readLevelConv0,
            split_size_or_sections=numReadsPerAllele[0],
        );
        perAlleleFrames1 = torch.split(
            readLevelConv1,
            split_size_or_sections=numReadsPerAllele[1],
        );
        reducedFrames0 = [
            torch.transpose(torch.sum(frame, dim=0), 0, 1) for frame in perAlleleFrames0
        ];
        reducedFrames1 = [
            torch.transpose(torch.sum(frame, dim=0), 0, 1) for frame in perAlleleFrames1
        ];
        return reducedFrames0, reducedFrames1;

    def forward(self, featureDict):
        alleles = [];
        features0 = [];
        features1 = [];
        numReadsPerAllele0 = [];
        numReadsPerAllele1 = [];

        for allele in featureDict:
            alleles.append(allele);
            features0.append(featureDict[allele][0]);
            features1.append(featureDict[allele][1]);
            numReadsPerAllele0.append(features0[-1].shape[0]);
            numReadsPerAllele1.append(features1[-1].shape[0]);

        features0 = torch.transpose(torch.cat(features0, dim=0), 1, 2);
        features1 = torch.transpose(torch.cat(features1, dim=0), 1, 2);
        readLevelConv0 = self.readConv0(features0);
        readLevelConv1 = self.readConv1(features1);
        metaPredictions = torch.squeeze(
            torch.nn.functional.softmax(
                self.meta(
                    self.prepareMetaInputs(readLevelConv0, readLevelConv1)
                ), dim=1
            ),
            dim=0
        );

        reducedFrames0, reducedFrames1 = self.prepareExpertInputs(
            readLevelConv0,
            readLevelConv1,
            (numReadsPerAllele0, numReadsPerAllele1)
        );

        featureDictNgs = dict();
        featureDictTgs = dict();
        featureDictHyb = dict();

        for allele, r0, r1 in zip(alleles, reducedFrames0, reducedFrames1):
            featureDictNgs[allele] = r0;
            featureDictTgs[allele] = r1;
            featureDictHyb[allele] = (r0, r1);

        def dictify(items):
            x, y = items;
            y = torch.squeeze(y, dim=0);
            if len(x) == 1:
                return {x[0]: y};
            else:
                return dict(zip(x, y));

        ngsPrediction = dictify(self.expert0(featureDictNgs));
        tgsPrediction = dictify(self.expert1(featureDictTgs));
        hybPrediction = dictify(self.expert2(featureDictHyb));

        expertPredictions = [];

        expertPredictions.append(
            [torch.sigmoid(ngsPrediction[a]) for a in alleles]
        );

        expertPredictions.append(
            [torch.sigmoid(tgsPrediction[a]) for a in alleles]
        );

        expertPredictions.append(
            [torch.sigmoid(hybPrediction[a]) for a in alleles]
        );

        alleleNumbers = dict({a: i for i, a in enumerate(alleles)});

        predictionQualities = dict();

        def invertPairing(pairing):
            return (pairing[1], pairing[0]);

        for allelePairing in itertools.product(alleles, alleles):
            if (
                (allelePairing in predictionQualities) or
                (invertPairing(allelePairing) in predictionQualities)
            ):
                continue;

            targetTensor = torch.zeros(len(alleles));
            targetTensor[alleleNumbers[allelePairing[0]]] = 1;
            targetTensor[alleleNumbers[allelePairing[1]]] = 1;
            expertLogProbs = torch.Tensor(
                [getExpertLogProb(torch.Tensor(e), targetTensor) for e in expertPredictions]
            );
            predictionQualities[allelePairing] = perSiteLogProb(expertLogProbs, metaPredictions);

        return predictionQualities;


class MoEMerged(torch.nn.Module):
    """
    Mixture-of-experts DNN with read convolution parameter sharing
    """
    def __init__(self, readConv0, readConv1, experts, meta):
        """
        :param readConv0: torch.Module
            A read convolver module (tech0)

        :param readConv1: torch.Module
            A read convolver module (tech1)

        :param experts: ExpertBundle
            Graph-searcher instances for each expert

        :param meta: torch.Module
            Meta-expert (not MetaExpert class above, but MetaExpert.network2)
        """
        super().__init__();
        self.readConv0 = readConv0;
        self.readConv1 = readConv1;
        self.meta = meta;
        self.expert0 = experts.ngs;
        self.expert1 = experts.tgs;
        self.expert2 = experts.hyb;

    def prepareMetaExpertInputs(self, readLevelConv0, readLevelConv1, numReadsPerSite):
        # Per site results
        perSiteFrames0 = torch.split(
            readLevelConv0,
            split_size_or_sections=numReadsPerSite[0]
        );
        perSiteFrames1 = torch.split(
            readLevelConv1,
            split_size_or_sections=numReadsPerSite[1]
        );

        # Sum up per-site results
        reducedFrames0 = reduceFrames(perSiteFrames0);
        reducedFrames1 = reduceFrames(perSiteFrames1);
        reducedFrames = torch.cat(
            (reducedFrames0, reducedFrames1), dim=1
        );

        return reducedFrames;

    def prepareExpertInputs(self, readLevelConv0, readLevelConv1, numReadsPerAllele):
        perAlleleFrames0 = torch.split(
            readLevelConv0,
            split_size_or_sections=numReadsPerAllele[0]
        );
        perAlleleFrames1 = torch.split(
            readLevelConv1,
            split_size_or_sections=numReadsPerAllele[1]
        );
        reducedFramesPerAllele0 = reduceFrames(perAlleleFrames0);
        reducedFramesPerAllele1 = reduceFrames(perAlleleFrames1);
        reducedFramesPerAlleleH = torch.cat(
            (reducedFramesPerAllele0, reducedFramesPerAllele1),
            dim=1
        );
        return reducedFramesPerAllele0, reducedFramesPerAllele1, reducedFramesPerAlleleH;

    def forward(self, tensors, numAllelesPerSite, numReadsPerAllele, numReadsPerSite):
        readLevelConv0 = self.readConv0(tensors[0].float());
        readLevelConv1 = self.readConv1(tensors[1].float());

        # Meta expert results
        meta = torch.nn.functional.softmax(
            self.meta(
                self.prepareMetaExpertInputs(readLevelConv0, readLevelConv1, numReadsPerSite)
            ), dim=1
        );

        # Expert inputs
        expertInputsNgs, expertInputsTgs, expertInputsHyb = self.prepareExpertInputs(
            readLevelConv0,
            readLevelConv1,
            numReadsPerAllele,
        );

        # Expert predictions
        ngsPredictions = self.expert0(expertInputsNgs, numAllelesPerSite);
        tgsPredictions = self.expert1(expertInputsTgs, numAllelesPerSite);
        hybPredictions = self.expert2(expertInputsHyb, numAllelesPerSite);

        # Return everything in original format
        return [ngsPredictions, tgsPredictions, hybPredictions], meta;


def initMetaToUniform(moe):
    """
    Utility function to initialize meta expert to choose all experts
    with approximately equal probability

    :param moe: MoEMergedWrapper
        Mixture of experts model in which to make initialization
    """
    meta = moe.meta;

    # See torch.nn.Sequential/torch.nn.Module
    # Pick the last item in ordered dictionary for torch.nn.Sequential
    # This is an iterator, hence cannot index directoy
    for _ in meta.network._modules.values():
        lastLayer = _;

    if hasattr(lastLayer, 'weight') and hasattr(lastLayer, 'bias'):
        logging.info("Initializing meta-expert's final layer");
        lastLayer.weight.data.normal_(0.0, 1e-2);  # Small weights
        lastLayer.bias.data.fill_(0.33);  # Relatively larger biases
    else:
        logging.info("Cannot initialize meta-expert since last layer doesn't have bias (or weight)");


def getExpertLogProb(predictions, targets):
    """
    Get log probability for a single expert for a given site target

    :param predictions: torch.Tensor
        Predictions for an expert for a single site

    :param targets: torch.Tensor
        Tensor object indicating target label for a single site

    :return: torch.Tensor (singleton)
        Singleton torch tensor which provides the log prob
        for a given expert for a given target, given the
        expert's predictions (outputs)
    """
    results = torch.sum(
        torch.log(
            torch.where(
                targets > 0, predictions, 1 - predictions
            ) + 1e-10
        )
    );
    return results;


def perSiteLogProb(expertLogProbs, metaExpertPredictions):
    """
    Get probability of a target given the probability
    of each expert for that target, and meta-expert weights

    :param expertLogProbs: torch.Tensor
        Log-probability of target from each expert

    :param metaExpertPredictions: torch.Tensor
        Meta-expert predictions (weights) for each expert

    :return: torch.Tensor (singleton)
        Singleton torch tensor which provides the log prob
        for a given expert for a given target
    """
    metaExpertLogProb = torch.log(metaExpertPredictions + 1e-10);
    # Joint (expert, switch)
    summed = metaExpertLogProb + expertLogProbs;
    # Sum over switch values/meta-expert values
    # Perform log(sum(exp(value - max))) trick
    maxValue = torch.max(summed);
    summedNormalized = summed - maxValue;
    results = torch.log(torch.sum(torch.exp(summedNormalized))) + maxValue;
    maxVal = results.clone().fill_(math.log(1 - 1e-8));
    results = torch.min(results, maxVal);  # Cap at quality score 80
    return results;


def getPosteriorProb(expertPredictions, metaExpertPredictions, targets):
    """
    Obtain posterior log probabilities

    :param expertPredictions: torch.Tensor
        Tensor representing predictions of each expert [batch, #experts]

    :param metaExpertPredictions: torch.Tensor
        Tensor representing weights for each expert

    :param targets: torch.Tensor
        Targets or labels
    """
    expertProbs = targets * expertPredictions + (1 - targets) * (1 - expertPredictions);
    jointProbs = expertProbs * metaExpertPredictions + 1e-10;  # For numerical stability
    probTargets = torch.sum(jointProbs, dim=1, keepdim=True);
    posterior = jointProbs / probTargets;
    return posterior;


class MoELoss(torch.nn.Module):
    """
    Loss function for EM-algorithm for mixture-of-experts model
    The output of MoE goes directly here
    """
    def __init__(self, regularizer=0, decay=0.5, provideIndividualLoss=False, weights=[1, 1], smoothing=0, aux_loss=0):
        super().__init__();
        # self.regularizer = regularizer;
        self.register_buffer(
            'regularizer', torch.Tensor([regularizer])
        );
        self.register_buffer(
            'decay', torch.Tensor([decay])
        );
        assert(0 <= decay <= 1), "Decay should be between 0 and 1";
        self.provideIndividualLoss = provideIndividualLoss;
        self.register_buffer(
            'weights', torch.Tensor(weights)
        );
        self.register_buffer(
            'smoothing', torch.Tensor([smoothing])
        );
        self.register_buffer(
            'aux_loss', torch.Tensor([aux_loss])
        );

    def preparePredictions(self, predictions, targets, numAllelesPerSite):
        """
        Prepare results from DNN into predictions for use in self.forward

        :param predictions: tuple
            Direct output of MoE object

        :param targets: torch.Tensor
            Flattened label tensor

        :param numAllelesPerSite: list
            List of number of alleles per site
        """
        expertPredictions, metaExpertPredictions = predictions;
        targets = torch.unsqueeze(targets.float(), dim=1);

        # Repeat meta-expert predictions the required number of times for each site
        # Loss is computed allele-level instead of site-level (math is equivalent; it's okay even if not)
        # This is much more computationally efficient
        repeatNumbers = torch.LongTensor(numAllelesPerSite);

        if metaExpertPredictions.is_cuda:
            repeatNumbers = repeatNumbers.cuda(metaExpertPredictions.get_device());

        # Compute entropy
        metaEntropy = -torch.sum(
            metaExpertPredictions * torch.log(metaExpertPredictions + 1e-10)
        ) * self.regularizer;
        self.regularizer = self.regularizer * self.decay;

        metaExpertPredictions = torch.repeat_interleave(
            metaExpertPredictions, repeatNumbers, dim=0
        );

        # Note: meta-expert predictions are already soft-maxed
        # Individual experts aren't sigmoid-ed
        expertPredictions = torch.sigmoid(
            torch.squeeze(
                torch.stack(
                    expertPredictions, dim=1
                ),
                dim=2
            )
        );

        return targets, metaExpertPredictions, expertPredictions, metaEntropy;

    @profile
    def forward(self, predictions, targets, numAllelesPerSite):
        """
        :param predictions: tuple
            Direct output of MoE object

        :param targets: torch.Tensor
            Flattened label tensor

        :param numAllelesPerSite: list
            List of number of alleles per site
        """
        # Obtain target weights
        targetWeights = torch.unsqueeze(self.weights[targets.long()], dim=1);

        targets, metaExpertPredictions, expertPredictions, entropy = \
            self.preparePredictions(predictions, targets, numAllelesPerSite);

        # Perform label smoothing
        if self.training:
            targets += targets * (-self.smoothing) + (1 - targets) * self.smoothing;

        with torch.no_grad():
            posterior = getPosteriorProb(expertPredictions, metaExpertPredictions, targets);

        expertProbs = targets * expertPredictions + (1 - targets) * (1 - expertPredictions);
        totalLoss = torch.sum(
            posterior * (torch.log(expertProbs + 1e-10) + torch.log(metaExpertPredictions + 1e-10)) * targetWeights
        );

        # We want to maximize likelihood, but also entropy
        if self.training:
            totalLoss = totalLoss + entropy;

        if self.provideIndividualLoss:
            individualLoss = torch.sum(
                torch.log(expertProbs + 1e-10),
                dim=0
            );

            if self.training:
                totalLoss = totalLoss + torch.sum(self.aux_loss * torch.sum(individualLoss) / 3);

            return -totalLoss, -individualLoss, posterior;
        else:
            return -totalLoss;


class PredictionLoss(MoELoss):
    """
    Loss function giving prediction loss
    """
    def __init__(self):
        super().__init__();

    def forward(self, predictions, targets, numAllelesPerSite):
        targets, metaExpertPredictions, expertPredictions, entropy = \
            self.preparePredictions(predictions, targets, numAllelesPerSite);

        expertProbs = targets * expertPredictions + (1 - targets) * (1 - expertPredictions);
        systemProbs = torch.sum(expertProbs * metaExpertPredictions, dim=1);
        totalLoss = torch.sum(torch.log(systemProbs + 1e-10));

        return -totalLoss;


class Accuracy(MoELoss):
    def __init__(self):
        super().__init__();

    def forward(self, predictions, targets, numAllelesPerSite):
        _, metaExpertPredictions, expertPredictions, entropy = \
            self.preparePredictions(predictions, targets, numAllelesPerSite);

        meanPrediction = torch.sum(expertPredictions * metaExpertPredictions, dim=1);
        predictedLabels = meanPrediction > 0.5;
        numCorrect = torch.sum(predictedLabels == targets);

        return numCorrect;


class IlluminaExpertWrapper(torch.nn.Module):
    """
    Wrapper for ReadConvolverWrapper class that uses only left half of features
    """
    def __init__(self, network):
        super().__init__();
        self.network = network;

    def forward(self, featureDict):
        featureDict_ = {};

        for key in featureDict:
            featureDict_[key] = featureDict[key][0];

        return self.network(featureDict_);


class PacBioExpertWrapper(torch.nn.Module):
    """
    Wrapper for ReadConvolverWrapper class that uses only right half of features
    """
    def __init__(self, network):
        super().__init__();
        self.network = network;

    def forward(self, featureDict):
        featureDict_ = {};

        for key in featureDict:
            featureDict_[key] = featureDict[key][1];

        return self.network(featureDict_);


class MoEWrapper(torch.nn.Module):
    """
    Wrapper for mixture of experts
    """
    def __init__(self, experts, meta):
        super().__init__();
        self.experts = experts;
        self.meta = meta;

        # If this is not done .cuda() calls won't work
        for i, e in enumerate(experts):
            setattr(self, 'expert%d' % i, e);

    def forward(self, featureDict):
        expertPredictions = dict();

        # Obtain meta expert predictions
        metaExpertPredictions = self.meta(featureDict);

        # Obtain expert predictions
        for expert in self.experts:
            alleles, predictions = expert(featureDict);

            for a, p in zip(alleles, predictions):
                if a not in expertPredictions:
                    expertPredictions[a] = [];

                expertPredictions[a].append(torch.sigmoid(p));

        allelesOrdered = tuple(expertPredictions.keys());
        expertPredictions = tuple(
            zip(
                *(expertPredictions[a] for a in allelesOrdered)
            )
        );

        # Obtain all targets, and evaluate their qualities
        alleleNumbers = dict({a: i for i, a in enumerate(allelesOrdered)});

        predictionQualities = dict();

        def invertPairing(pairing):
            return (pairing[1], pairing[0]);

        for allelePairing in itertools.product(allelesOrdered, allelesOrdered):
            if (
                (allelePairing in predictionQualities) or
                (invertPairing(allelePairing) in predictionQualities)
            ):
                continue;

            targetTensor = torch.zeros(len(allelesOrdered));
            targetTensor[alleleNumbers[allelePairing[0]]] = 1;
            targetTensor[alleleNumbers[allelePairing[1]]] = 1;
            expertLogProbs = torch.Tensor(
                [getExpertLogProb(torch.Tensor(e), targetTensor) for e in expertPredictions]
            );
            predictionQualities[allelePairing] = perSiteLogProb(expertLogProbs, metaExpertPredictions);

        return predictionQualities;


def createExpert(config):
    """
    Creates a single expert and wraps it as necessary
    """
    readConv = AlleleSearcherDNN.Network(importlib.import_module(config['readConv']).config);
    alleleConv0 = AlleleSearcherDNN.Network(importlib.import_module(config['allele']).config);
    alleleConv1 = AlleleSearcherDNN.Network(importlib.import_module(config['allele']).config);
    graphConv = AlleleSearcherDNN.Network(importlib.import_module(config['graph']).config);
    graphSearcher = AlleleSearcherDNN.GraphSearcher(alleleConv0, alleleConv1, graphConv, useOneHot=True);
    readConvolver = ReadConvolver.ReadConvolverDNN(readConv, graphSearcher);
    graphSearcher.name = "SingleGraph";
    readConvolver.name = "SingleTopLevel";
    return readConvolver;


def createMetaExpert(config):
    """
    Creates a single meta-expert
    """
    readConv0 = AlleleSearcherDNN.Network(importlib.import_module(config['readConv']).config);
    readConv1 = AlleleSearcherDNN.Network(importlib.import_module(config['readConv']).config);
    combiner = AlleleSearcherDNN.Network(importlib.import_module(config['combiner']).config);
    metaExpert = MetaExpert(readConv0, readConv1, combiner);
    return metaExpert;


def createExpertHybrid(config):
    """
    Creates hybrid DNN
    """
    readConv0 = AlleleSearcherDNN.Network(importlib.import_module(config['readConv']).config);
    readConv1 = AlleleSearcherDNN.Network(importlib.import_module(config['readConv']).config);
    alleleConv0 = AlleleSearcherDNN.Network(importlib.import_module(config['allele']).config);
    alleleConv1 = AlleleSearcherDNN.Network(importlib.import_module(config['allele']).config);
    graphConv = AlleleSearcherDNN.Network(importlib.import_module(config['graph']).config);
    graphSearcher = AlleleSearcherDNN.GraphSearcher(alleleConv0, alleleConv1, graphConv, useOneHot=True);
    readConvolver = ReadConvolver.ReadConvolverHybridDNN((readConv0, readConv1), graphSearcher);
    graphSearcher.name = "HybridGraph";
    readConvolver.name = "HybridTopLevel";
    return readConvolver;


def graphConvWrapper(graphConv):
    """
    Provides a graph convolver wrapper from a graph convolver
    """
    graphConvWrapper = AlleleSearcherDNN.GraphSearcherWrapper(
        graphConv.network0,
        graphConv.network1,
        graphConv.network2,
    );
    return graphConvWrapper;


def graphConvMergedWrapper(graphConv):
    """
    Provides a graph convolver wrapper from a graph convolver
    """
    graphConvWrapper = GraphSearcherMergedWrapper(
        graphConv.network0,
        graphConv.network1,
    );
    return graphConvWrapper;


def getWrappedDNN(moe):
    """
    Get a wrapped MoE DNN from MoE used for training
    """
    # This is a place-holder; put a function call here
    if type(moe) is MoEMerged:
        return MoEMergedWrapper();

    # NGS wrapper
    readConvNGS = moe.expert0.network.network0;
    graphConvNGS = moe.expert0.network.network1;
    ngsWrapper = IlluminaExpertWrapper(
        ReadConvolver.ReadConvolverWrapper(
            readConvNGS, graphConvWrapper(graphConvNGS)
        )
    );

    # TGS Wrapper
    readConvTGS = moe.expert1.network.network0;
    graphConvTGS = moe.expert1.network.network1;
    tgsWrapper = PacBioExpertWrapper(
        ReadConvolver.ReadConvolverWrapper(
            readConvTGS, graphConvWrapper(graphConvTGS)
        )
    );

    # Hybrid DNN wrapper
    readConvHyb0 = moe.expert2.network0;
    readConvHyb1 = moe.expert2.network1;
    graphConvHyb = moe.expert2.network2;
    hybWrapper = ReadConvolver.ReadConvolverHybridWrapper(
        (readConvHyb0, readConvHyb1), graphConvWrapper(graphConvHyb)
    );

    # Meta-expert Wrapper
    metaWrapper = MetaExpertWrapper(
        moe.meta.network0,
        moe.meta.network1,
        moe.meta.network2,
    );

    # MoE wrapper
    moeWrapper = MoEWrapper(
        [ngsWrapper, tgsWrapper, hybWrapper], metaWrapper,
    );

    return moeWrapper;


def createMoEModel(configDict):
    """
    Provides a mixture of experts model based on a config dictionary
    """
    ngsDNN = createExpert(configDict['ngs']);
    tgsDNN = createExpert(configDict['tgs']);
    hybDNN = createExpertHybrid(configDict['hybrid']);
    meta = createMetaExpert(configDict['meta']);
    moe = MoE([
        IlluminaExpert(ngsDNN),
        PacBioExpert(tgsDNN),
        hybDNN,
    ], meta);

    return moe;


def createMoEMergedWrapper(moe):
    readConv0 = moe.readConv0;
    readConv1 = moe.readConv1;
    meta = moe.meta;
    ngsWrapper = graphConvWrapper(moe.expert0);
    tgsWrapper = graphConvWrapper(moe.expert1);
    hybWrapper = graphConvWrapper(moe.expert2);
    experts = ExpertBundle(
        ngs=ngsWrapper,
        tgs=tgsWrapper,
        hyb=hybWrapper,
    );
    return MoEMergedWrapper(readConv0, readConv1, experts, meta);


def createMoEFullMergedWrapper(moe):
    readConv0 = moe.readConv0;
    readConv1 = moe.readConv1;
    meta = moe.meta;
    ngsWrapper = graphConvMergedWrapper(moe.expert0);
    tgsWrapper = graphConvMergedWrapper(moe.expert1);
    hybWrapper = graphConvMergedWrapper(moe.expert2);
    experts = ExpertBundle(
        ngs=ngsWrapper,
        tgs=tgsWrapper,
        hyb=hybWrapper,
    );
    return MoEMergedWrapper(readConv0, readConv1, experts, meta);


def createMoEMergedModel(configDict):
    # Prepare read convolvers
    ngsConvolver = AlleleSearcherDNN.Network(importlib.import_module(configDict['readConv']).config);
    tgsConvolver = AlleleSearcherDNN.Network(importlib.import_module(configDict['readConv']).config);
    meta = AlleleSearcherDNN.Network(importlib.import_module(configDict['meta']).config);

    alleleConvNgs0 = AlleleSearcherDNN.Network(importlib.import_module(configDict['alleleConvSingle']).config);
    alleleConvNgs1 = AlleleSearcherDNN.Network(importlib.import_module(configDict['alleleConvSingle']).config);
    alleleConvTgs0 = AlleleSearcherDNN.Network(importlib.import_module(configDict['alleleConvSingle']).config);
    alleleConvTgs1 = AlleleSearcherDNN.Network(importlib.import_module(configDict['alleleConvSingle']).config);
    alleleConvHyb0 = AlleleSearcherDNN.Network(importlib.import_module(configDict['alleleConvHybrid']).config);
    alleleConvHyb1 = AlleleSearcherDNN.Network(importlib.import_module(configDict['alleleConvHybrid']).config);
    graphConvNgs = AlleleSearcherDNN.Network(importlib.import_module(configDict['graphConvSingle']).config);
    graphConvTgs = AlleleSearcherDNN.Network(importlib.import_module(configDict['graphConvSingle']).config);
    graphConvHyb = AlleleSearcherDNN.Network(importlib.import_module(configDict['graphConvHybrid']).config);

    ngsExpert = AlleleSearcherDNN.GraphSearcher(alleleConvNgs0, alleleConvNgs1, graphConvNgs, useOneHot=True);
    tgsExpert = AlleleSearcherDNN.GraphSearcher(alleleConvTgs0, alleleConvTgs1, graphConvTgs, useOneHot=True);
    hybExpert = AlleleSearcherDNN.GraphSearcher(alleleConvHyb0, alleleConvHyb1, graphConvHyb, useOneHot=True);

    moe = MoEMerged(
        ngsConvolver,
        tgsConvolver,
        ExpertBundle(ngs=ngsExpert, tgs=tgsExpert, hyb=hybExpert),
        meta,
    );

    return moe;


def createMoEFullMergedModel(configDict):
    # Prepare a fully-merged Mixture of Experts model
    ngsConvolver = AlleleSearcherDNN.Network(importlib.import_module(configDict['readConv']).config);
    tgsConvolver = AlleleSearcherDNN.Network(importlib.import_module(configDict['readConv']).config);
    meta = AlleleSearcherDNN.Network(importlib.import_module(configDict['meta']).config);

    alleleConvNgs = AlleleSearcherDNN.Network(importlib.import_module(configDict['alleleConvSingle']).config);
    alleleConvTgs = AlleleSearcherDNN.Network(importlib.import_module(configDict['alleleConvSingle']).config);
    alleleConvHyb = AlleleSearcherDNN.Network(importlib.import_module(configDict['alleleConvHybrid']).config);
    graphConvNgs = AlleleSearcherDNN.Network(importlib.import_module(configDict['graphConvSingle']).config);
    graphConvTgs = AlleleSearcherDNN.Network(importlib.import_module(configDict['graphConvSingle']).config);
    graphConvHyb = AlleleSearcherDNN.Network(importlib.import_module(configDict['graphConvHybrid']).config);

    ngsExpert = GraphSearcherMerged(alleleConvNgs, graphConvNgs);
    tgsExpert = GraphSearcherMerged(alleleConvTgs, graphConvTgs);
    hybExpert = GraphSearcherMerged(alleleConvHyb, graphConvHyb);

    moe = MoEMerged(
        ngsConvolver,
        tgsConvolver,
        ExpertBundle(ngs=ngsExpert, tgs=tgsExpert, hyb=hybExpert),
        meta,
    );

    return moe;


def addGradientsToTensorBoard(moeFullMergedModel, logWriter, step):
    """
    Add gradient histogram to a tensorboard writer for a MoEFullMerged model

    :param moeFullMergedModel: MoEMerged
        Neural network for which we determine gradient histograms

    :param logWriter: tensorboard.SummaryWriter
        tensorboard writer object

    :param step: int
        Time-step at which gradient is calculated
    """
    if type(moeFullMergedModel) is not MoEMerged:
        return;

    def writeNetworkGrads(network, prefix):
        histograms = [];

        for parameter in network.parameters():
            histograms += parameter.grad.cpu().data.numpy().flatten().tolist();

        logWriter.add_histogram(prefix + "_step%d" % step, torch.Tensor(histograms), step);

    # Write histograms of weights for read convolvers
    writeNetworkGrads(moeFullMergedModel.readConv0, 'readConv0');
    writeNetworkGrads(moeFullMergedModel.readConv1, 'readConv1');
    writeNetworkGrads(moeFullMergedModel.meta, 'meta');
    writeNetworkGrads(moeFullMergedModel.expert0, 'expert0');
    writeNetworkGrads(moeFullMergedModel.expert1, 'expert1');
    writeNetworkGrads(moeFullMergedModel.expert2, 'expert2');


def createMoEFullMergedConditionalModel(configDict):
    # Prepare a fully-merged Mixture of Experts model with conditional networks at the input

    # Convolvers that force zero-feed-through for reads
    ngsConvolver = ConditionalNetwork(importlib.import_module(configDict['readConv']).config);
    tgsConvolver = ConditionalNetwork(importlib.import_module(configDict['readConv']).config);

    # Convolvers that force zero-feed-through for alleles
    alleleConvNgs = ConditionalNetwork(importlib.import_module(configDict['alleleConvSingle']).config);
    alleleConvTgs = ConditionalNetwork(importlib.import_module(configDict['alleleConvSingle']).config);
    alleleConvHyb = ConditionalNetwork(importlib.import_module(configDict['alleleConvHybrid']).config);

    # Convolvers that predict low probabilities for zero-valued inputs
    graphConvNgs = ConditionalNetwork(importlib.import_module(configDict['graphConvSingle']).config, offset=PREDICTIVE_OFFSET);
    graphConvTgs = ConditionalNetwork(importlib.import_module(configDict['graphConvSingle']).config, offset=PREDICTIVE_OFFSET);
    graphConvHyb = ConditionalNetwork(importlib.import_module(configDict['graphConvHybrid']).config, offset=PREDICTIVE_OFFSET);

    # Meta-expert
    meta = AlleleSearcherDNN.Network(importlib.import_module(configDict['meta']).config);

    # Individual experts
    ngsExpert = GraphSearcherMerged(alleleConvNgs, graphConvNgs);
    tgsExpert = GraphSearcherMerged(alleleConvTgs, graphConvTgs);
    hybExpert = GraphSearcherMerged(alleleConvHyb, graphConvHyb);

    moe = MoEMerged(
        ngsConvolver,
        tgsConvolver,
        ExpertBundle(ngs=ngsExpert, tgs=tgsExpert, hyb=hybExpert),
        meta,
    );

    return moe;
