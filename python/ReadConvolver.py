"""
This file contains experimental code for a prefix-DNN to the DNNs in AlleleSearcherDNN. The prefix DNN performs
convolutions on input read features. These features are then combined using a simple summation. The use of a DNN
allows the values to be added together without losing individual read-level features. The file also includes
additional tools needed for this purpose.
"""
import torch
import torch.utils.data
import h5py
import logging

try:
    profile
except Exception:
    def profile(x):
        return x;


@profile
def collateFnReadConvolverHybrid(batch):
    """
    Collate function for hybrid read convolver

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

        # Replicate depth for each allele
        numReadsPerSite0.extend(
            [depth[0] if (depth[0] > 0) else 1 for i in range(numAlleles)]
        );
        numReadsPerSite1.extend(
            [depth[1] if (depth[1] > 0) else 1 for i in range(numAlleles)]
        );

    labels = torch.cat(labelsPerSite, dim=0);

    return (
        (torch.cat(allTensors0, dim=0), torch.cat(allTensors1, dim=0)),
        labels,
        (torch.LongTensor(numReadsPerAllele0), torch.LongTensor(numReadsPerAllele1)),
        torch.LongTensor(numAllelesPerSite),
        (torch.Tensor(numReadsPerSite0) / 2, torch.Tensor(numReadsPerSite1) / 2),
    );


def customCat(allTensors):
    """
    This customCat works the same way as default_collate does
    """
    numel = sum([x.numel() for x in allTensors])
    storage = allTensors[0].storage()._new_shared(numel)
    out = allTensors[0].new(storage)
    return torch.cat(allTensors, 0, out=out)


@profile
def collateFnReadConvolver(batch):
    """
    Collate function for read convolver

    :param batch: list
        Batch of tensors

    :return: tuple
        - torch.Tensor: [total#reads in batch, featureLength, 18]
        - list: list of torch.Tensor objects indicating labels per site
        - list: number of reads per each allele in the batch
        - list: number of alleles per site
    """
    numReadsPerAllele = [];
    numAllelesPerSite = [];
    allTensors = [];
    labelsPerSite = [];
    numReadsPerSite = [];

    for items_ in batch:
        # NoneType items can come from subsampling failures
        if items_ is None:
            continue;

        items = items_[0];
        depth = items_[1];
        assert(len(items) % 2 == 0);
        numAlleles = len(items) // 2;
        numAllelesPerSite.append(numAlleles);
        tensors = items[:numAlleles];
        labels_ = items[numAlleles:];

        for tensor in tensors:
            numReadsPerAllele.append(tensor.shape[0]);
            allTensors.append(tensor);

        labelsPerSite.append(torch.Tensor(labels_));

        numReadsPerSite.extend(
            [depth if (depth > 0) else 1 for i in range(numAllelesPerSite[-1])]
        );

    labels = torch.cat(labelsPerSite, dim=0);

    # return torch.cat(allTensors, dim=0), labels, torch.LongTensor(numReadsPerAllele), torch.LongTensor(numAllelesPerSite);
    return (
        customCat(allTensors),
        labels,
        torch.LongTensor(numReadsPerAllele),
        torch.LongTensor(numAllelesPerSite),
        torch.Tensor(numReadsPerSite) / 2
    );


# # The uncollate function is redefined here, even though it is the same as in AlleleSearcherDNN
# # This is in order to prevent circular imports.
# def uncollateFnReadConvolver(collatedTensors, numEntries):
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

def uncollateFnReadConvolver(collatedTensors, numEntries):
    return torch.split(collatedTensors, split_size_or_sections=numEntries);


def normalize(result, depthMultiplier):
    if depthMultiplier is None:
        return result;

    depthMultiplier = torch.unsqueeze(torch.unsqueeze(depthMultiplier, dim=1), dim=1);
    return result / depthMultiplier;


class ReadConvolverWrapper(torch.nn.Module):
    """
    Wrapper class for read convolver
    """
    def __init__(self, network0, network1):
        """
        :param network0: AlleleSearcherDNN.Network
            Performs read convolution

        :param network1: AlleleSearcherDNN.GraphSearcherWrapper
            Wrapper for AlleleSearcherDNN.GraphSearcher
        """
        super().__init__();
        self.network0 = network0;
        self.network1 = network1;

    def forward(self, featureDict, index=None, **kwargs):
        """
        :param featureDict: dict
            Dictionary representing features for each allele

        :param index: list
            Enables transfer learning by providing intermediate
            layers' output for the final GraphSearcher

        :param depthMultiplier: kwargs entry
            Provide a depth normalization factor as a kwargs entry
        """
        # Perform read convolution for each allele
        for allele, feature in featureDict.items():
            # Note: Currently hybrid mode is not supported
            feature = torch.transpose(feature, 1, 2);
            convResult = torch.sum(self.network0(feature), dim=0);

            if 'depthMultiplier' in kwargs:
                logging.debug("Found depth multiplier, normalizing");
                # convResult = normalize(convResult, kwargs['depthMultiplier']);
                convResult = convResult / kwargs['depthMultiplier'];

            featureDict[allele] = torch.transpose(
                convResult, 0, 1
            );

        # Pass on modified features to GraphSearcher
        return self.network1(featureDict, index=index);


class ReadConvolverDNN(torch.nn.Module):
    """
    The read convolver DNN is a combination of two DNNs
    The first is a read-by-read 1d convolutional network. This is followed by an accumulation operation
    The accumulated results are passed through an AlleleSearcherDNN.GraphSearcher
    """
    def __init__(self, network0, network1, enableMultiGPU=False):
        """
        :param network0: AlleleSearcherDNN.Network
            Performs read-wise convolutions

        :param network1: AlleleSearcherDNN.GraphSearcher
            Performs site-specific convolutions

        :param enableMultiGPU: bool
            Whether we should enable Multi-GPU training or not
        """
        super().__init__();
        wrapper = torch.nn.DataParallel if enableMultiGPU else lambda x : x;
        self.network0 = wrapper(network0);
        self.network1 = network1;

    def forward(self, tensors, numAllelesPerSite, numReadsPerAllele, *args, **kwargs):
        """
        :param tensors: torch.Tensor
            A flattened tensor of all reads at a site

        :param numAllelesPerSite: torch.LongTensor
            Number of alleles per site

        :param numReadsPerAllele: torch.LongTensor
            Number of reads per allele

        :return: list
            List of allelic predictions per site
        """
        # Perform read-by-read convolution
        frames = self.network0(tensors.float());

        # Uncollate the results of read-by-read convolution, and reduce
        perAlleleFrames = uncollateFnReadConvolver(frames, numReadsPerAllele);

        # Sum up the results for each allele
        reducedFrames = list(torch.sum(frame, dim=0) for frame in perAlleleFrames);
        reducedFrames = torch.stack(reducedFrames, dim=0);

        # Normalize if depth multiplier is provided
        if 'depthMultiplier' in kwargs:
            logging.debug("Found depth multiplier");

            # How to incorporate depth multiplier - normalize, or add feature dimension
            if 'multiplierMode' in kwargs:
                multiplierMode = kwargs['multiplierMode'];
                logging.debug("Received multiplier mode %s" % multiplierMode);
            else:
                logging.debug("Did not receive multiplier mode; setting to %s" % multiplierMode);
                multiplierMode = 'normalize';

            if multiplierMode == 'normalize':
                # In this case, simply normalize using depthMultiplier
                reducedFrames = normalize(reducedFrames, kwargs['depthMultiplier']);
            else:
                # Simply attach a feature dimension to the output of reducedFrames indicating read depth
                # reducedFrames is [#batch, #channels, length]
                # Adjust addendum to be [#batch, 1, length]
                addendum = torch.unsqueeze(torch.unsqueeze(kwargs['depthMultiplier'], dim=1), dim=1);  # [batch, 1, 1]
                addendum = addendum.expand(reducedFrames.shape[0], 1, reducedFrames.shape[2]);  # [batch, 1, length]
                # Add a single item to the channel dimension
                reducedFrames = torch.cat((addendum, reducedFrames), dim=1);  # [batch, #channels + 1, length]
        else:
            logging.debug("Not using depth multiplier normalization");

        # Send this through GraphSearcherDNN
        return self.network1(reducedFrames, numAllelesPerSite);


class ReadConvolverHybridWrapper(torch.nn.Module):
    """
    Wrapper class for read convolver
    """
    def __init__(self, networks0, network1):
        """
        :param networks0: tuple
            (AlleleSearcherDNN.Network, AlleleSearcherDNN.Network) representing
            DNNs for performing read convolutions

        :param network1: AlleleSearcherDNN.GraphSearcherWrapper
            Wrapper for AlleleSearcherDNN.GraphSearcher
        """
        super().__init__();
        self.network0, self.network1 = networks0;
        self.network2 = network1;

    def forward(self, featureDict, *args, **kwargs):
        """
        :param featureDict: dict
            Dictionary representing features for each allele
        """
        # Perform read convolutions for each allele
        for allele, feature in featureDict.items():
            f0, f1 = feature;
            f0 = torch.transpose(f0, 1, 2);
            f1 = torch.transpose(f1, 1, 2);
            convResult0 = torch.sum(self.network0(f0), dim=0);
            convResult1 = torch.sum(self.network1(f1), dim=0);

            if 'depthMultiplier' in kwargs:
                logging.debug("Found depth multiplier, normalizing");
                # convResult0 = normalize(convResult0, kwargs['depthMultiplier'][0]);
                # convResult1 = normalize(convResult1, kwargs['depthMultiplier'][1]);
                convResult0 = convResult0 / kwargs['depthMultiplier'][0];
                convResult1 = convResult1 / kwargs['depthMultiplier'][1];

            featureDict[allele] = (
                torch.transpose(
                    convResult0, 0, 1
                ),
                torch.transpose(
                    convResult1, 0, 1
                ),
            );

        # Pass on modified features to GraphSearcher
        return self.network2(featureDict);


class ReadConvolverHybridDNN(torch.nn.Module):
    """
    The read convolver hybrid DNN uses two separate sets of inputs for
    two types of sequencing data. It uses two sets of network for processing
    individual reads in the data, then a single GraphSearcher to analyze the
    results
    """
    def __init__(self, networks0, network1, enableMultiGPU=False):
        """
        :param networks0: tuple
            (AlleleSearcherDNN.Network, AlleleSearcherDNN.Network) representing
            DNNs for analyzing two types of data

        :param network1: AlleleSearcherDNN.GraphSearcher
            Performs site-specific convolutions

        :param enableMultiGPU: bool
            Whether to enable multi-GPU training or not
        """
        super().__init__();
        wrapper = torch.nn.DataParallel if enableMultiGPU else lambda x : x;
        self.network0 = wrapper(networks0[0]);
        self.network1 = wrapper(networks0[1]);
        self.network2 = network1;

    def forward(
        self,
        tensors,
        numAllelesPerSite,
        numReadsPerAllele,
        *args,
        **kwargs,
    ):
        """
        :param tensors: tuple
            (tensors0, tensors1) representing two types of data

        :param numAllelesPerSite: torch.LongTensor
            Number of alleles per site

        :param numReadsPerAllele: (torch.LongTensor, torch.LongTensor)
            Number of reads per allele in each data source

        :return: list
            List of allelic predictions per site
        """
        # Read-by-read convolution
        frames0 = self.network0(tensors[0].float());
        frames1 = self.network1(tensors[1].float());

        # Uncollate results of read-by-read convolution
        perAlleleFrames0 = uncollateFnReadConvolver(frames0, numReadsPerAllele[0]);
        perAlleleFrames1 = uncollateFnReadConvolver(frames1, numReadsPerAllele[1]);

        # Sum up the read convolution results to obtain per-allele results
        reducedFrames0 = list(torch.sum(frame, dim=0) for frame in perAlleleFrames0);
        reducedFrames1 = list(torch.sum(frame, dim=0) for frame in perAlleleFrames1);
        reducedFrames0 = torch.stack(reducedFrames0, dim=0);
        reducedFrames1 = torch.stack(reducedFrames1, dim=0);

        if 'depthMultiplier' in kwargs:
            logging.debug("Found depth multiplier, normalizing");
            depthMultiplier = kwargs['depthMultiplier'];
            reducedFrames0 = normalize(reducedFrames0, depthMultiplier[0]);
            reducedFrames1 = normalize(reducedFrames1, depthMultiplier[1]);

        # Concatenate the results and send it through GraphSearcher
        # Concatenation across the channel dimension
        reducedFrames = torch.cat(
            (
                reducedFrames0,
                reducedFrames1,
            ), dim=1,
        );

        return self.network2(reducedFrames, numAllelesPerSite);
