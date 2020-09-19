import torch
import itertools
import NNTools
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
    )


def reduceSlots(d, slots):
    if type(slots) is list:
        slots = torch.LongTensor(slots)
        if d.is_cuda:
            slots = slots.cuda(device=d.get_device())

    results = torch.cumsum(d, dim=0)
    indices = torch.cumsum(slots, dim=0) - 1
    zeroPad = torch.unsqueeze(torch.zeros_like(d[0]), dim=0)
    resultsSelected = results[indices]
    paddedSelections = torch.cat((zeroPad, resultsSelected[:-1]), dim=0)
    return resultsSelected - paddedSelections


class ConvCombiner(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.network = NNTools.Network(config)

    def forward(self, conv1, conv2):
        convInput = torch.cat((conv1, conv2), dim=1)
        return self.network(convInput)


class DummyGraphNetwork(torch.nn.Module):
    """
    A dummy graph tensor that always returns a negative label
    for any input
    """
    def __init__(self):
        super().__init__()
        self.register_buffer(
            'baseTensor',
            torch.Tensor([-20])
        )

    def forward(self, *args, **kwargs):
        tensors = args[0]
        return torch.unsqueeze(self.baseTensor.repeat(tensors.shape[0]), dim=1)


def batch_dot_product(a, b):
    return torch.matmul(
        a.view(a.shape[0], 1, a.shape[1]),
        b.view(b.shape[0], b.shape[1], 1)
    )


class MoEAttention(torch.nn.Module):
    def __init__(
        self,
        read_convolver0,
        read_convolver1,
        compressor0,
        compressor1,
        compressor2,
        xattn0,
        xattn1,
        xattn2,
        combiner0,
        combiner1,
        meta,
    ):
        """
        Attention-based Mixture-of-Experts model

        :param read_convolver?: NNTools.Network
            Convolutional modules for convolving reads of different types

        :param compressor?: NNTools.Network
            Dilated convolutional modules for compressing read features along the length-dimension

        :param xattn?: NNTools.Network
            Cross-attention modules for comparing allele-level to site-level features

        :param combiner?: NNTools.Network
            Combiner modules for combining higher-level features from multiple platforms

        :param meta: NNTools.Network
            Meta-expert DNN. Neural network works off-of compressed site-level features
        """
        super().__init__()
        self.read_convolver0 = read_convolver0
        self.read_convolver1 = read_convolver1
        self.compressor0 = compressor0
        self.compressor1 = compressor1
        self.compressor2 = compressor2
        self.xattn0 = xattn0
        self.xattn1 = xattn1
        self.xattn2 = xattn2
        self.meta = meta
        self.combiner0 = combiner0
        self.combiner1 = combiner1

    def compress_and_predict(
        self,
        reduced_frames_for_allele,
        numAllelesPerSite,
        compressor_index
    ):
        # Compress inputs (allele-level)
        compressor = getattr(self, 'compressor%d' % compressor_index)
        compressed_features_allele = compressor(reduced_frames_for_allele)

        # Prepare tensor version
        num_alleles_per_site_tensor = torch.LongTensor(numAllelesPerSite)
        if reduced_frames_for_allele.is_cuda:
            num_alleles_per_site_tensor = num_alleles_per_site_tensor.cuda(device=reduced_frames_for_allele.get_device())

        # Prepare per-site compressed features from read-level features
        reduced_frames_for_site = reduceSlots(
            reduced_frames_for_allele, num_alleles_per_site_tensor
        )
        compressed_features_site0 = compressor(reduced_frames_for_site)
        expanded_frames_for_site0 = torch.repeat_interleave(
            compressed_features_site0, num_alleles_per_site_tensor, dim=0
        )

        # Prepare per-site compressed features from allele-level compressed features
        compressed_features_site1 = reduceSlots(
            compressed_features_allele, num_alleles_per_site_tensor
        )
        expanded_frames_for_site1 = torch.repeat_interleave(
            compressed_features_site1, num_alleles_per_site_tensor, dim=0
        )

        # Attention computation for prediction
        if hasattr(self, 'xattn%d' % compressor_index):
            attn_network = getattr(self, 'xattn%d' % compressor_index)

            expert_predictions = attn_network(
                (compressed_features_allele, (expanded_frames_for_site0, expanded_frames_for_site1))
            )
        else:
            expert_predictions = None

        return expert_predictions, (compressed_features_site0, compressed_features_site1), compressed_features_allele

    def forward(self, tensors, numAllelesPerSite, numReadsPerAllele, reference_segments, *args, **kwargs):
        read_conv0 = self.read_convolver0(tensors[0].float())
        reduced_frames0_for_allele = reduceSlots(read_conv0, numReadsPerAllele[0])
        expert0_predictions, f0, compressed_features_allele0 = self.compress_and_predict(
            reduced_frames0_for_allele,
            numAllelesPerSite,
            0
        )

        # Note that for all hybrid modes, we need read_convolver1
        # Conversely, if read_convolver1 is configured, then this is a hybrid DNN
        if self.read_convolver1 is not None:
            read_conv1 = self.read_convolver1(tensors[1].float())
            reduced_frames1_for_allele = reduceSlots(read_conv1, numReadsPerAllele[1])
            expert1_predictions, f1, compressed_features_allele1 = self.compress_and_predict(
                reduced_frames1_for_allele,
                numAllelesPerSite,
                1
            )

            if hasattr(self, 'compressor2') and self.compressor2:
                # We can produce hybrid features either directly from read-level features
                # using a hybrid compressor, or ...
                assert(hasattr(self, 'xattn2') and self.xattn2), "xattn2 is needed with compressor2"
                reduced_frames2_for_allele = reduced_frames0_for_allele + reduced_frames1_for_allele
                expert2_predictions, f2, compressed_features_allele1 = self.compress_and_predict(
                    reduced_frames2_for_allele,
                    numAllelesPerSite,
                    compressor_index=2
                )
                # Meta expert uses site frames from scratch
                site_frames_for_meta = f2[0]
            elif hasattr(self, 'xattn2') and self.xattn2:
                # ... alternatively, we can use combiner networks to combine the features at the
                # allele and site-level and then compare the two using xattn2
                assert(hasattr(self, 'combiner0') and self.combiner0), "Combiner network expected"
                assert(hasattr(self, 'combiner1') and self.combiner1), "Combiner network expected"

                # Create num alleles per site tensor
                if read_conv0.is_cuda:
                    num_alleles_per_site_tensor = torch.LongTensor(numAllelesPerSite).cuda(read_conv0.get_device())
                else:
                    num_alleles_per_site_tensor = torch.LongTensor(numAllelesPerSite)

                # Create combined allele and site-level features
                compressed_features_allele2 = self.combiner0((compressed_features_allele0, compressed_features_allele1))
                compressed_features_site2 = self.combiner1((f0[1], f1[1]))
                expert2_predictions = self.xattn2((
                    compressed_features_allele2,
                    (
                        None,
                        torch.repeat_interleave(
                            compressed_features_site2, repeats=num_alleles_per_site_tensor, dim=0
                        ),
                    )
                ))

                # Meta expert uses site-level features used by xattn2
                site_frames_for_meta = compressed_features_site2
            else:
                expert2_predictions = None

                # Meta expert uses pooled resources from the two experts
                site_frames_for_meta = reduceSlots(
                    reduced_frames0_for_allele + reduced_frames1_for_allele,
                    numAllelesPerSite
                )

            if self.meta is not None:
                meta_predictions = torch.softmax(
                    self.meta((site_frames_for_meta, reference_segments.float())), dim=-1
                )
            else:
                self.meta = None

            # One of three situations are possible
            if (expert0_predictions is None) and (expert1_predictions is None):
                # Case 1. Neither expert0 nor expert1, but only expert2 that produces a binary classification
                assert(expert2_predictions is not None), "No expert prediction is valid"
                return expert2_predictions
            elif expert2_predictions is None:
                # Case 2. No expert2, and expert0 and expert1 work with the meta-expert
                assert((expert0_predictions is not None) and (expert1_predictions is not None)), "Hybrid data provided, but only single tech prediction is available"
                expert2_predictions = torch.zeros_like(expert0_predictions)
                return [expert0_predictions, expert1_predictions, expert2_predictions], meta_predictions
            elif (expert0_predictions is not None) and (expert1_predictions is not None) and (expert2_predictions is not None):
                # Case 3. All experts are present and we are off to the races
                return [expert0_predictions, expert1_predictions, expert2_predictions], meta_predictions
            else:
                raise ValueError("Unknown prediction type for hybrid data")
        else:
            return expert0_predictions


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
        :param readConv0: NNTools.Network
            A read convolver module (tech0)

        :param readConv1: NNTools.Network
            A read convolver module (tech1)

        :param alleleConv0: NNTools.Network
            Allele-level convolver (tech0)

        :param alleleConv1: NNTools.Network
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
        super().__init__()
        self.readConv0 = readConv0
        self.readConv1 = readConv1
        self.alleleConv0 = alleleConv0
        self.alleleConv1 = alleleConv1
        self.expert0 = experts[0]
        self.expert1 = experts[1]
        self.expert2 = experts[2]
        self.meta = meta
        self.detachMeta = detachMeta
        self.detachExperts = detachExperts
        self.useAdditive = useAdditive
        self.alleleConvCombiner = alleleConvCombiner
        self.siteConvCombiner = siteConvCombiner

        if useSeparateMeta:
            logging.info("Will use separate read convolvers for the meta-expert")
            self.readConv0Meta = copy.deepcopy(readConv0)
            self.readConv1Meta = copy.deepcopy(readConv1)

    def performAlleleConv(self, readLevelConv0, readLevelConv1, numReadsPerAllele):
        reducedFramesPerAllele0 = self.alleleConv0(
            reduceSlots(readLevelConv0, numReadsPerAllele[0])
        )
        if (self.alleleConv1 is not None) and (readLevelConv1 is not None):
            reducedFramesPerAllele1 = self.alleleConv1(
                reduceSlots(readLevelConv1, numReadsPerAllele[1])
            )
        else:
            reducedFramesPerAllele1 = None
        return reducedFramesPerAllele0, reducedFramesPerAllele1

    def preparePerSiteFramesFromReads(self, conv0, conv1, numAllelesPerSite, numReadsPerAllele):
        perAlleleFrames0 = torch.split(
            conv0,
            split_size_or_sections=numReadsPerAllele[0]
        )
        perAlleleFrames1 = torch.split(
            conv1,
            split_size_or_sections=numReadsPerAllele[1]
        )

        def reduceArrays(array):
            return torch.cat(array, dim=0)

        endBoundaries = np.cumsum(numAllelesPerSite).tolist()
        beginBoundaries = [0] + endBoundaries[:-1]

        perSiteFrames0 = [
            reduceArrays(perAlleleFrames0[x: y]) for x, y in zip(beginBoundaries, endBoundaries)
        ]
        perSiteFrames1 = [
            reduceArrays(perAlleleFrames1[x: y]) for x, y in zip(beginBoundaries, endBoundaries)
        ]

        return reduceFrames(perSiteFrames0), reduceFrames(perSiteFrames1)

    def preparePerSiteFrames(self, postAlleleLevelConv, numAllelesPerSite):
        return reduceSlots(postAlleleLevelConv, numAllelesPerSite)

    def expertComputations(self, alleleLevelConv, perSiteFrames, numAllelesPerSiteTensor, expertIdx):
        def computations():
            remainingAllelesSignal = torch.repeat_interleave(
                perSiteFrames, numAllelesPerSiteTensor, dim=0
            ) - alleleLevelConv

            if hasattr(self, 'useAdditive') and self.useAdditive:
                expertInput = alleleLevelConv - remainingAllelesSignal
            else:
                expertInput = torch.cat((alleleLevelConv, remainingAllelesSignal), dim=1)

            return getattr(self, 'expert%d' % expertIdx)(expertInput)

        if hasattr(self, 'detachExperts') and self.detachExperts:
            with torch.no_grad():
                return computations()
        else:
            return computations()

    def metaExpertComputations(self, perSiteFrames):
        if hasattr(self, 'detachMeta') and self.detachMeta:
            perSiteFrames_ = perSiteFrames.clone().detach()
            return self.meta(perSiteFrames_)
        else:
            return self.meta(perSiteFrames)

    def forward(self, tensors, numAllelesPerSite, numReadsPerAllele, *args, **kwargs):
        # 1. Perform read-level convolutions
        readLevelConv0 = self.readConv0(tensors[0].float())
        readLevelConv1 = self.readConv1(tensors[1].float()) if (self.readConv1 is not None) else None

        # 2. Perform allele-level convolutions
        alleleLevelConv0, alleleLevelConv1 = self.performAlleleConv(
            readLevelConv0, readLevelConv1, numReadsPerAllele
        )

        if alleleLevelConv1 is not None:
            if hasattr(self, 'useAdditive') and self.useAdditive:
                if self.alleleConvCombiner is None:
                    alleleLevelConv2 = alleleLevelConv0 + alleleLevelConv1
                else:
                    alleleLevelConv2 = self.alleleConvCombiner(
                        alleleLevelConv0, alleleLevelConv1
                    )
            else:
                alleleLevelConv2 = torch.cat(
                    (alleleLevelConv0, alleleLevelConv1), dim=1
                )

        # 3. Determine per-site frames
        perSiteFrame0 = self.preparePerSiteFrames(alleleLevelConv0, numAllelesPerSite)
        perSiteFrame1 = self.preparePerSiteFrames(alleleLevelConv1, numAllelesPerSite) if (alleleLevelConv1 is not None) else None

        if perSiteFrame1 is not None:
            if hasattr(self, 'useAdditive') and self.useAdditive:
                if self.siteConvCombiner is None:
                    # perSiteFrame2 = perSiteFrame0 + perSiteFrame1
                    # Note: This is conceptually better than the previous solution
                    # This makes all three experts use the same type of computations,
                    # A_j - \sum_{l != j} A_l, to produce input feature maps
                    perSiteFrame2 = self.preparePerSiteFrames(alleleLevelConv2, numAllelesPerSite)
                else:
                    perSiteFrame2 = self.siteConvCombiner(
                        perSiteFrame0, perSiteFrame1,
                    )
            else:
                perSiteFrame2 = torch.cat(
                    (perSiteFrame0, perSiteFrame1),
                    dim=1
                ) if perSiteFrame1 else None

            if hasattr(self, 'readConv0Meta'):
                readLevelConv0Meta = self.readConv0Meta(tensors[0].float())
                readLevelConv1Meta = self.readConv1Meta(tensors[1].float())
                perSiteFrames0Meta, perSiteFrames1Meta = self.preparePerSiteFramesFromReads(
                    readLevelConv0Meta,
                    readLevelConv1Meta,
                    numAllelesPerSite,
                    numReadsPerAllele
                )
                if hasattr(self, 'useAdditive') and self.useAdditive:
                    if self.siteConvCombiner is None:
                        perSiteFrameMeta = perSiteFrames0Meta + perSiteFrames1Meta
                    else:
                        perSiteFrameMeta = self.siteConvCombiner(
                            perSiteFrames0Meta, perSiteFrames1Meta
                        )
                else:
                    perSiteFrameMeta = torch.cat(
                        (perSiteFrames0Meta, perSiteFrames1Meta),
                        dim=1
                    )
            else:
                perSiteFrameMeta = perSiteFrame2

        # 4. Prepare alleles per site tensor
        numAllelesPerSiteTensor = torch.LongTensor(numAllelesPerSite)
        if perSiteFrame0.is_cuda:
            numAllelesPerSiteTensor = numAllelesPerSiteTensor.cuda(device=perSiteFrame0.get_device())

        # 5. Perform expert computations
        ngsPredictions = self.expertComputations(alleleLevelConv0, perSiteFrame0, numAllelesPerSiteTensor, 0)

        if perSiteFrame1 is not None:
            tgsPredictions = self.expertComputations(alleleLevelConv1, perSiteFrame1, numAllelesPerSiteTensor, 1)
            hybPredictions = self.expertComputations(alleleLevelConv2, perSiteFrame2, numAllelesPerSiteTensor, 2)

            # 6. Perform meta-expert computations
            meta = torch.nn.functional.softmax(self.meta(perSiteFrameMeta), dim=1)

            return [ngsPredictions, tgsPredictions, hybPredictions], meta
        else:
            return ngsPredictions


class MoEMergedWrapperAdvanced(torch.nn.Module):
    def __init__(self, moeMerged, providePredictions=False):
        super().__init__()
        self.moeMerged = moeMerged
        self.providePredictions = providePredictions

    def _singleFeatureDictData(self, featureDict):
        # Convert items in single feature dict into format for MoEMergedAdvanced
        alleles = list(featureDict.keys())

        numAllelesPerSite = [len(alleles)]
        numReadsPerAllele0 = [featureDict[key][0].shape[0] for key in alleles]
        numReadsPerAllele1 = [
            featureDict[key][1].shape[0] if (featureDict[key][1] is not None) else None \
            for key in alleles
        ]
        tensors0 = torch.cat(
            [torch.transpose(featureDict[key][0], 1, 2) for key in alleles], dim=0
        )

        # Note: This is not really a reflection of whether this is a hybrid variant calling mode, but
        # rather whether we need to combine predictions from multiple experts or not. This is unfortunately named
        hybrid = self.moeMerged.meta is not None

        if None in numReadsPerAllele1:
            tensors1 = None
        else:
            tensors1 = torch.cat(
                [torch.transpose(featureDict[key][1], 1, 2) for key in alleles], dim=0
            )

        return alleles, (tensors0, tensors1), numAllelesPerSite, (numReadsPerAllele0, numReadsPerAllele1), hybrid

    def forward(self, featureDict, segment):
        nnInputs = self._singleFeatureDictData(featureDict)
        alleles = nnInputs[0]
        # Modifications for accepting reference segment input
        args = list(nnInputs[1: -1]) + [segment]
        results = self.moeMerged(*args)
        ####
        hybrid = nnInputs[-1]
        alleleNumbers = dict({a: i for i, a in enumerate(alleles)})

        if hybrid:
            experts, meta = results
            meta = meta[0];  # Remove unnecessary batch dimension
            experts = [torch.squeeze(torch.sigmoid(e), dim=1) for e in experts]
        else:
            expert0 = torch.squeeze(torch.sigmoid(results), dim=1)
            experts = [expert0, torch.zeros_like(expert0), torch.zeros_like(expert0)]
            meta = torch.zeros(3)
            meta[0] = 1

        def invert(pairing):
            return (pairing[1], pairing[0])

        def expertProbability(prediction, target):
            return torch.exp(
                torch.sum(
                    torch.log(prediction * target + (1 - prediction) * (1 - target) + 1e-10)
                )
            )

        def siteLogProb(e_, m_):
            m_ = torch.log(m_ + 1e-10)
            joint = m_ + e_
            maxVal = torch.max(joint)
            jointNormed = joint - maxVal
            results = torch.log(torch.sum(torch.exp(jointNormed))) + maxVal
            maxProb = results.clone().fill_(math.log(1 - 1e-8))
            return torch.min(results, maxProb)

        def getExpertPredictions(expertId):
            predictionQualities = dict()

            for allelePairing in itertools.product(alleles, alleles):
                if (allelePairing in predictionQualities) or (invert(allelePairing) in predictionQualities):
                    continue
                targetTensor = torch.zeros(len(alleles))
                targetTensor[alleleNumbers[allelePairing[0]]] = 1
                targetTensor[alleleNumbers[allelePairing[1]]] = 1
                expertProb = expertProbability(experts[expertId], targetTensor)
                predictionQualities[allelePairing] = expertProb

            return predictionQualities

        def getPredictionDictionary():
            predictionQualities = dict()
            expert0Prediction = getExpertPredictions(0)
            expert1Prediction = getExpertPredictions(1)
            expert2Prediction = getExpertPredictions(2)

            for allelePairing in expert0Prediction:
                predictionQualities[allelePairing] = (
                    meta[0] * expert0Prediction[allelePairing] + meta[1] * expert1Prediction[allelePairing] + meta[2] * expert2Prediction[allelePairing]
                )

            return predictionQualities, expert0Prediction, expert1Prediction, expert2Prediction

        if self.providePredictions:
            return tuple(list(getPredictionDictionary()) + [meta])
        else:
            return getPredictionDictionary()[0]


def make_network(configDict, name, module_to_use=NNTools.Network, use_weight_norm=False):
    if name in configDict and (configDict[name] is not None):
        if type(configDict[name]) is str:
            module = importlib.import_module(configDict[name])

            # Regenerate config with batch-normalization if necessary
            if hasattr(module, 'weight_norm') and use_weight_norm:
                module.weight_norm = True
                module.gen_config()
                logging.info("Enabling weight-normalization for layer")

            configuration = module.config
        elif type(configDict[name]) is list:
            configuration = configDict[name]
        else:
            raise AttributeError("configurations should point to a module or should be a list")

        return module_to_use(configuration)
    else:
        return None


def createMoEFullMergedAdvancedModel(configDict, useSeparateMeta=False):
    use_weight_norm = 'weight_norm' in configDict and configDict['weight_norm']

    if use_weight_norm:
        logging.info("Using weight-normalization network-wide")

    ngsConvolver = make_network(configDict, "readConvNGS", use_weight_norm=use_weight_norm)
    tgsConvolver = make_network(configDict, "readConvTGS", use_weight_norm=use_weight_norm)
    meta = make_network(configDict, "meta")

    alleleConvNgs = make_network(configDict, "alleleConvSingleNGS", use_weight_norm=use_weight_norm)
    alleleConvTgs = make_network(configDict, "alleleConvSingleTGS", use_weight_norm=use_weight_norm)
    graphConvNgs = make_network(configDict, "graphConvSingleNGS", use_weight_norm=use_weight_norm)
    graphConvTgs = make_network(configDict, "graphConvSingleTGS", use_weight_norm=use_weight_norm)

    # It is possible to avoid using graph convolver hybrid; in
    # this case, use a dummy module that always outputs a small value
    graphConvHyb = make_network(configDict, "graphConvHybrid", use_weight_norm=use_weight_norm)

    alleleConvCombiner = make_network(configDict, "alleleConvCombiner", module_to_use=ConvCombiner, use_weight_norm=use_weight_norm)
    siteConvCombiner = make_network(configDict, "siteConvCombiner", module_to_use=ConvCombiner, use_weight_norm=use_weight_norm)

    if 'kwargs' in configDict:
        kwargs = configDict['kwargs']
    else:
        kwargs = dict()

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
    )

    return moe


def create_moe_attention_model(config_dict, *args, **kwargs):
    use_weight_norm = 'weight_norm' in config_dict and config_dict['weight_norm']

    if use_weight_norm:
        logging.info("Using weight-normalization network-wide")

    read_convolver0 = make_network(config_dict, "read_conv0", use_weight_norm=use_weight_norm)
    read_convolver1 = make_network(config_dict, "read_conv1", use_weight_norm=use_weight_norm)

    compressor0 = make_network(config_dict, "compressor0", use_weight_norm=use_weight_norm)
    compressor1 = make_network(config_dict, "compressor1", use_weight_norm=use_weight_norm)
    compressor2 = make_network(config_dict, "compressor2", use_weight_norm=use_weight_norm)

    xattn0 = make_network(config_dict, "xattn0", use_weight_norm=use_weight_norm)
    xattn1 = make_network(config_dict, "xattn1", use_weight_norm=use_weight_norm)
    xattn2 = make_network(config_dict, "xattn2", use_weight_norm=use_weight_norm)

    combiner0 = make_network(config_dict, "combiner0", use_weight_norm=use_weight_norm)
    combiner1 = make_network(config_dict, "combiner1", use_weight_norm=use_weight_norm)

    meta = make_network(config_dict, "meta", use_weight_norm=use_weight_norm)

    moe = MoEAttention(
        read_convolver0=read_convolver0,
        read_convolver1=read_convolver1,
        compressor0=compressor0,
        compressor1=compressor1,
        compressor2=compressor2,
        xattn0=xattn0,
        xattn1=xattn1,
        xattn2=xattn2,
        combiner0=combiner0,
        combiner1=combiner1,
        meta=meta,
    )

    # Remove unnecessary components (patch)
    # This is because the check for xattn0-1 is simply
    # based on hasattr. For xattn2 we also (correctly) check
    # whether it is None. I do not want to change the code at
    # this stage
    if not moe.xattn0:
        del moe.xattn0
    if not moe.xattn1:
        del moe.xattn1

    return moe


def createMoEFullMergedAdvancedModelWrapper(moe):
    return MoEMergedWrapperAdvanced(moe)
