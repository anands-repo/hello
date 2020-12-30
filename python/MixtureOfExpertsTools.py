# © 2019 University of Illinois Board of Trustees.  All rights reserved
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

PREDICTIVE_OFFSET = math.log(1e-8 / (1 - 1e-8))
np.set_printoptions(threshold=sys.maxsize)

try:
    profile
except Exception:
    def profile(x):
        return x


def cappedLog(tensor):
    cushionedTensor = tensor + 1e-10

    # Make sure that cushioning for log is not applied for
    # large values. Otherwise log may go positive, and cause
    # problems in loss.backward() calls
    return torch.log(
        torch.where(
            cushionedTensor > 1 - 1e-10,
            tensor,
            cushionedTensor,
        )
    )


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
        cappedLog(
            torch.where(
                targets > 0, predictions, 1 - predictions
            )
        )
    )
    return results


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
    metaExpertLogProb = cappedLog(metaExpertPredictions)
    # Joint (expert, switch)
    summed = metaExpertLogProb + expertLogProbs
    # Sum over switch values/meta-expert values
    # Perform log(sum(exp(value - max))) trick
    maxValue = torch.max(summed)
    summedNormalized = summed - maxValue
    results = torch.log(torch.sum(torch.exp(summedNormalized))) + maxValue
    maxVal = results.clone().fill_(math.log(1 - 1e-8))
    results = torch.min(results, maxVal);  # Cap at quality score 80
    return results


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
    expertProbs = targets * expertPredictions + (1 - targets) * (1 - expertPredictions)
    jointProbs = expertProbs * metaExpertPredictions + 1e-10;  # For numerical stability
    probTargets = torch.sum(jointProbs, dim=1, keepdim=True)
    posterior = jointProbs / probTargets
    return posterior


class MoELoss(torch.nn.Module):
    """
    Loss function for EM-algorithm for mixture-of-experts model
    The output of MoE goes directly here
    """
    def __init__(self, regularizer=0, decay=0.5, provideIndividualLoss=False, weights=[1, 1], smoothing=0, aux_loss=0):
        super().__init__()
        # self.regularizer = regularizer
        self.register_buffer(
            'regularizer', torch.Tensor([regularizer])
        )
        self.register_buffer(
            'decay', torch.Tensor([decay])
        )
        assert(0 <= decay <= 1), "Decay should be between 0 and 1"
        self.provideIndividualLoss = provideIndividualLoss
        self.register_buffer(
            'weights', torch.Tensor(weights)
        )
        self.register_buffer(
            'smoothing', torch.Tensor([smoothing])
        )
        self.register_buffer(
            'aux_loss', torch.Tensor([aux_loss])
        )

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
        expertPredictions, metaExpertPredictions = predictions
        targets = torch.unsqueeze(targets.float(), dim=1)

        # Repeat meta-expert predictions the required number of times for each site
        # Loss is computed allele-level instead of site-level (math is equivalent; it's okay even if not)
        # This is much more computationally efficient
        repeatNumbers = torch.LongTensor(numAllelesPerSite)

        if metaExpertPredictions.is_cuda:
            repeatNumbers = repeatNumbers.cuda(metaExpertPredictions.get_device())

        # Compute entropy
        metaEntropy = -torch.sum(
            metaExpertPredictions * cappedLog(metaExpertPredictions)
        ) * self.regularizer
        self.regularizer = self.regularizer * self.decay

        metaExpertPredictions = torch.repeat_interleave(
            metaExpertPredictions, repeatNumbers, dim=0
        )

        # Note: meta-expert predictions are already soft-maxed
        # Individual experts aren't sigmoid-ed
        expertPredictions = torch.sigmoid(
            torch.squeeze(
                torch.stack(
                    expertPredictions, dim=1
                ),
                dim=2
            )
        )

        return targets, metaExpertPredictions, expertPredictions, metaEntropy

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
        targetWeights = torch.unsqueeze(self.weights[targets.long()], dim=1)

        targets, metaExpertPredictions, expertPredictions, entropy = \
            self.preparePredictions(predictions, targets, numAllelesPerSite)

        # Perform label smoothing
        if self.training:
            targets += targets * (-self.smoothing) + (1 - targets) * self.smoothing

        with torch.no_grad():
            posterior = getPosteriorProb(expertPredictions, metaExpertPredictions, targets)

        expertProbs = targets * expertPredictions + (1 - targets) * (1 - expertPredictions)
        totalLoss = torch.sum(
            posterior * (cappedLog(expertProbs) + cappedLog(metaExpertPredictions)) * targetWeights
        )

        # We want to maximize likelihood, but also entropy
        if self.training:
            totalLoss = totalLoss + entropy

        if self.provideIndividualLoss:
            individualLoss = torch.sum(
                cappedLog(expertProbs),
                dim=0
            )

            if self.training:
                totalLoss = totalLoss + torch.sum(self.aux_loss * torch.sum(individualLoss) / 3)

            return -totalLoss, -individualLoss, posterior
        else:
            return -totalLoss


class PredictionLoss(MoELoss):
    """
    Loss function giving prediction loss
    """
    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets, numAllelesPerSite):
        targets, metaExpertPredictions, expertPredictions, entropy = \
            self.preparePredictions(predictions, targets, numAllelesPerSite)

        expertProbs = targets * expertPredictions + (1 - targets) * (1 - expertPredictions)
        systemProbs = torch.sum(expertProbs * metaExpertPredictions, dim=1)
        totalLoss = torch.sum(cappedLog(systemProbs))

        return -totalLoss


class Accuracy(MoELoss):
    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets, numAllelesPerSite):
        _, metaExpertPredictions, expertPredictions, entropy = \
            self.preparePredictions(predictions, targets, numAllelesPerSite)

        meanPrediction = torch.sum(expertPredictions * metaExpertPredictions, dim=1)
        predictedLabels = meanPrediction > 0.5
        numCorrect = torch.sum(predictedLabels == targets)

        return numCorrect
