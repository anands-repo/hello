# © 2019 University of Illinois Board of Trustees.  All rights reserved
import MixtureOfExpertsAdvanced
from MixtureOfExpertsDNNFast import WrapperForDataParallel
import MixtureOfExpertsDNNFast
import torch
import sys
import argparse
import torch.distributed
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a model wrapper and data parallel model")

    parser.add_argument(
        "--ddp",
        help="Distributed data parallel model",
        required=True,
    )

    parser.add_argument(
        "--output_prefix",
        help="Prefix of the output file",
        required=True,
    )

    args = parser.parse_args()

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "9413"

    torch.distributed.init_process_group(backend='gloo', init_method='env://', world_size=1, rank=0)

    model = torch.load(args.ddp, map_location='cpu')
    base_model = model.module
    wrapper = MixtureOfExpertsAdvanced.createMoEFullMergedAdvancedModelWrapper(base_model)
    data_parallel = torch.nn.DataParallel(
        MixtureOfExpertsDNNFast.WrapperForDataParallel(base_model)
    )

    torch.save(
        wrapper, args.output_prefix + ".wrapper.dnn"
    )

    torch.save(
        data_parallel, args.output_prefix + ".dp.dnn"
    )
