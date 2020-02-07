from Bio import SeqIO
import os
import warnings
import ast
from multiprocessing import Pool
import argparse


class ReferenceCache(object):
    def __init__(self, fasta=None, database=None, chrom=None, positions_per_file=10000):
        """
        Splits the reference sequence into multiple files and stores them for easy access

        :rtype: reference_cache object

        :param fasta: str
            The reference fasta file

        :param database: str
            The directory root where the reference resides in file format. If fasta is provided, but database is not provided, then reference cache will be dumped into the current directory

        :param chrom: str
            If provided, the reference cache will create only one chromosome on disk

        :param positions_per_file: int
            Number of positions per file
        """
        assert ((fasta is not None) or (database is not None)), "One of fasta, or database should be given";

        # Remove the "chr" prefix from chromosomes
        if (chrom is not None) and (chrom[:3] == 'chr'):
            chrom = chrom[3:];

        if fasta is not None:
            if database is None:
                warnings.warn("Dumping reference cache into current directory");
                database = "reference_cache";

            print("NOTE: Creating reference_cache database. This may take a few minutes ... ");

            # Open reference sequence for access
            fhandle = open(fasta, 'r');
            reference = SeqIO.parse(fhandle, 'fasta');

            # Iterate through contig to dump reference_chunks
            for record in reference:
                chromosome = record.id;
                directory = os.path.join(database, str(chromosome));

                # If the chrom argument is provided, work with only that
                if chrom is not None:
                    if chromosome != chrom: continue;

                # Make sub-directory for chromosome
                if not os.path.exists(directory):
                    os.makedirs(directory);

                # Go through the sequence and dump into files, use iterators so memory isn't clogged
                seqiter = iter(record.seq);
                EOS = False;
                chunk = 0;

                while not EOS:
                    sub = [];

                    for i in range(positions_per_file):
                        try:
                            sub.append(next(seqiter));
                        except StopIteration:
                            EOS = True;
                            break;

                    chunk_name = os.path.join(directory, 'reference_chunk%d' % chunk);

                    with open(chunk_name, 'w') as chandle:
                        chandle.write(str(sub));

                    chunk += 1;

            fhandle.close();

        self.fasta = fasta;
        self.database = database;
        self.positions_per_file = positions_per_file;
        self.__chrom = chrom;
        self.__last_filename = None;
        self.__last_contents = None;

    def __reduce__(self):
        return ReferenceCache, (self.fasta, self.database, self.chrom, self.positions_per_file);

    @property
    def chrom(self):
        return self.__chrom;

    @chrom.setter
    def chrom(self, _chrom):
        if _chrom[:3] == 'chr':
            _chrom = _chrom[3:];

        self.__chrom = _chrom;

        # Delete length, since it is not valid any more
        if hasattr(self, '_total_len'):
            del self._total_len;

    # @profile
    def slice(self, chrom, start, end):
        """
        Provide a slice from the reference sequence

        :param chrom: str
            Chromosome id from which reference is to be sliced

        :param start: int
            Start position of the slice

        :param end: int
            End position of the slice

        :return: list
            List containing reference sequence from start to end in chrom
        """
        directory = os.path.join(self.database, str(chrom));

        if not os.path.exists(directory):
            raise ValueError("Cannot find chromosome %s" % str(chrom));

        file_start = start // self.positions_per_file;
        file_end = end // self.positions_per_file;

        files_to_open = [os.path.join(directory, 'reference_chunk%d' % i) for i in range(file_start, file_end + 1)];

        sequence = [];

        # Caching file contents to prevent repeated file opening and closing
        for f in files_to_open:
            if (self.__last_filename is not None) and (self.__last_filename == f):
                sequence += self.__last_contents;
            else:
                with open(f, 'r') as fhandle:
                    self.__last_filename = f;
                    self.__last_contents = ast.literal_eval(fhandle.read());
                    sequence += self.__last_contents;

        seq_start = start - file_start * self.positions_per_file;
        seq_end = seq_start + (end - start);
        sequence = sequence[seq_start:seq_end]

        return sequence;

    def __len__(self):
        if self.chrom is None:
            raise ValueError("Cannot perform len operation without chromosome being set");

        if hasattr(self, '_total_len'):
            return self._total_len;

        directory = os.path.join(self.database, str(self.chrom));
        filelist = os.listdir(directory);
        filenums = [int(f[15:]) for f in filelist];
        lastnum = sorted(filenums)[-1];
        lastfile = os.path.join(directory, 'reference_chunk' + str(lastnum));

        # Open last file
        sequence = [];

        with open(lastfile, 'r') as fhandle:
            sequence += ast.literal_eval(fhandle.read());

        total_len = lastnum * self.positions_per_file + len(sequence);

        self._total_len = total_len;

        return total_len;

    def __getitem__(self, index):
        if self.chrom is None:
            raise ValueError("Set chrom before using __getitem__");

        if type(index) is slice:
            assert ((index.start is not None) and (
                        index.stop is not None)), "Provide both start and stop indices for the indexing operation";

            slice_ = self.slice(self.chrom, index.start, index.stop);

            if index.step is not None:
                newslice = slice(0, len(slice_), index.step);
                slice_ = slice_[newslice];
        else:
            slice_ = self.slice(self.chrom, index, index + 1)[0];

        return slice_;


def create_reference_cache(args):
    fasta, directory, chrom = args;

    cache = ReferenceCache(fasta, directory, chrom);
    del cache;


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Convert a reference sequence into cached reference on disk");

    parser.add_argument("--fasta",
                        help="Fasta file carrying the reference",
                        action="store",
                        required=True,
                        );

    parser.add_argument("--output_directory",
                        help="Output directory name",
                        action="store",
                        required=True,
                        );

    parser.add_argument("--num_threads",
                        help="Number of CPU threads to use",
                        action="store",
                        type=int,
                        default=4,
                        );

    args = parser.parse_args();

    # First, open reference and obtain all contig ids
    contigs = [];

    with open(args.fasta, 'r') as fhandle:
        reference = SeqIO.parse(fhandle, 'fasta');

        for record in reference:
            contigs.append(record.id);
            print("Found contig %s" % record.id);

    if args.num_threads > 1:
        arguments = list(zip(
            [args.fasta] * len(contigs),
            [args.output_directory] * len(contigs),
            contigs,
        ));

        workers = Pool(args.num_threads);
        workers.map(create_reference_cache, arguments);
    else:
        cache = ReferenceCache(args.fasta, args.output_directory);
        del cache;
