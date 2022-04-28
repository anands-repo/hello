from MixtureOfExpertsDNNFastDistributed import *
import torch
import argparse
import os


if __name__ == "__main__":
	parser = argparse.ArgumentParser("Read distributed data parallel model and save it")

	parser.add_argument(
		"--input",
		help="DDP model",
		required=True,
	)

	parser.add_argument(
		"--output",
		help="DDP model saved as DataParallel model",
		required=True,
	)

	args = parser.parse_args()

	os.environ["MASTER_ADDR"] = "localhost"
	os.environ["MASTER_PORT"] = "8080"

	assert(args.output != args.input), "Same name for input and output not allowed"

	torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=1, rank=0)

	model = torch.nn.DataParallel(WrapperForDataParallel(torch.load(args.input, map_location="cpu").module)).cuda()

	torch.save(model, args.output)
