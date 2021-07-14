import argparse
import glob
from os.path import join

from tokenizers import ByteLevelBPETokenizer

parser = argparse.ArgumentParser()
parser.add_argument(
    "--files",
    metavar="path",
    type=str,
    default="python_data/train.txt",
    help="The files to use as training; accept '**/*.txt' type of patterns \
                          if enclosed in quotes",
)
parser.add_argument(
    "--out",
    default="trained_models/roberta_python_tokenizer",
    type=str,
    help="Path to the output directory, where the files will be saved",
)
parser.add_argument(
    "--name", default="bpe-bytelevel", type=str, help="The name of the output vocab files"
)
args = parser.parse_args()

files = glob.glob(args.files)
if not files:
    print(f"File does not exist: {args.files}")
    exit(1)


# Initialize an empty tokenizer
tokenizer = ByteLevelBPETokenizer(add_prefix_space=True)

# And then train
tokenizer.train(
    files,
    vocab_size=50265,
    min_frequency=2,
    show_progress=True,
    special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
)

# Save the files
os.makedirs(args.out, exist_ok=True)
tokenizer.save_model(args.out, args.name)

# Restoring model from learned vocab/merges
tokenizer = ByteLevelBPETokenizer(
    join(args.out, "{}-vocab.json".format(args.name)),
    join(args.out, "{}-merges.txt".format(args.name)),
    add_prefix_space=True,
)

# Test encoding
print(tokenizer.encode("Training ByteLevel BPE is very easy").tokens)
