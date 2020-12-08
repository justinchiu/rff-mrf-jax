

import numpy as np

from data.lm import PennTreebank, WikiText2
from data.data import BucketIterator, BPTTIterator
from data.field import Field

from args import get_args

# factor out args later
args = get_args()
print(args)

TEXT = Field(batch_first = True)

if args.dataset == "ptb":
    Dataset = PennTreebank
elif args.dataset == "wikitext2":
    Dataset = WikiText2

# get raw data as strings
train, valid, test = Dataset.splits(
    TEXT,
    newline_eos = True,
)

# build vocab
TEXT.build_vocab(train)
V = TEXT.vocab

# build iterators
def batch_size_tokens(new, count, sofar):
    return max(len(new.text), sofar)
def batch_size_sents(new, count, sofar):
    return count

if args.iterator == "bucket":
    train_iter, valid_iter, test_iter = BucketIterator.splits(
        (train, valid, test),
        batch_sizes = [args.bsz, args.eval_bsz, args.eval_bsz],
        sort_key = lambda x: len(x.text),
        batch_size_fn = batch_size_tokens if args.bsz_fn == "tokens" else batch_size_sents,
    )
elif args.iterator == "bptt":
    train_iter, valid_iter, test_iter = BPTTIterator.splits(
        (train, valid, test),
        batch_sizes = [args.bsz, args.eval_bsz, args.eval_bsz],
        bptt_len = args.bptt,
        sort = False,
    )
else:
    raise ValueError(f"Invalid iterator {args.iterator}")

if args.no_shuffle_train:
    train_iter.shuffle = False

first_batch = next(iter(train_iter))
print(first_batch)
import pdb; pdb.set_trace()
