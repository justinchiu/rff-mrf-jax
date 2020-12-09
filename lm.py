

import numpy as np

import jax
from jax import random
import jax.numpy as jnp
import flax.linen as nn
import flax.optim as optim

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
# batchify module
FfLm = nn.vmap(
    FfLm,
    in_axes = 0,
    out_axes = 0,
    variable_axes = {"params": None},
    split_rngs = {"params": False, "dropout": True},
) 
model = FfLm(
    V = len(V),
    emb_dim = args.emb_dim,
    hidden_dim = args.hidden_dim,
    order = args.order,
    num_layers = args.num_layers,
)

optimizer = optim.Adam(
    learning_rate = args.lr,
)

first_batch = next(iter(train_iter))
print(first_batch)
key = random.PRNGKey(0)

def process_batch(batch, key, model):
    text, lengths = batch.text
    N, T = text.shape
    state = model.init_state(N)

    key, key1, key2, key3 = random.split(, 4)

    # raises error without dropout key
    variables = model.init({"params": key1, "dropout": key2}, text, state)

    output = model.apply(variables, text[:,:-1], state, rngs={"dropout": key3})

    Nr = jnp.arange(N)
    Tr = jnp.arange(T)

    mask = Tr.tile((N, 1)) < lengths[:,None]
    loss = output[
        Nr[:,None,None],
        Tr[None,:,None],
        text[:,:,None],
    ][:,:,0][mask].sum()
    nwords = mask.sum()

    return loss, nwords, key

loss, nwords, key = process_batch(first_batch, key, model)
import pdb; pdb.set_trace()
