

import numpy as np

import jax
from jax import random
import jax.numpy as jnp
import flax.linen as nn

from data.lm import PennTreebank, WikiText2
from data.data import BucketIterator, BPTTIterator
from data.field import Field

from models.ff import FfLm

from args import get_args

# factor out args later
args = get_args()
print(args)

TEXT = Field(batch_first = True, include_lengths = True)

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


# begin hacking
 
model = FfLm(
    V = len(V),
    emb_dim = args.emb_dim,
    hidden_dim = args.hidden_dim,
    order = args.order,
    num_layers = args.num_layers,
)

first_batch = next(iter(train_iter))
print(first_batch)
text, lengths = first_batch.text
state = model.init_state(len(first_batch.text[0]))

key1, key2, key3 = random.split(random.PRNGKey(0), 3)

variables = model.init({"params": key1, "dropout": key2}, text, state)

F = jax.vmap(model.apply, (None, 0, 0), 0)
output = F(variables, text, state)

import pdb; pdb.set_trace()

