import pysam


class PySamFastaWrapper:
    def __init__(self, database, chrom=None):
        self.fasta = database;
        self.chrom = chrom;
        self.handle = pysam.FastaFile(database);

    def __len__(self):
        if self.chrom is not None:
            return self.handle.get_reference_length(self.chrom);

        raise ValueError("Chromosome is not set");

    def __getitem__(self, index):
        assert(self.chrom is not None), "Chromosome is not set";

        if type(index) is slice:
            assert(
                (index.start is not None) and (index.stop is not None)
            ), "Provide both start and stop indices when slicing";

            bases = list(self.handle.fetch(self.chrom, index.start, index.stop));
        else:
            bases = self.handle.fetch(self.chrom, index, index + 1);

        return bases;
