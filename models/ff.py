
import functools
from typing import Any, Callable

import numpy as np

import jax
from jax import lax, random, numpy as jnp
from jax.scipy.special import logsumexp as lse

import flax
import flax.linen as nn
from flax.nn import dropout # deprecated or not?

class FfLm(nn.Module):
    #V: Any # torchtext.Vocab or int?
    V: int
    emb_dim: int
    hidden_dim: int
    num_layers: int
    order: int
    dropout: float = 0.3
    tie_weights: bool = True
    linear_init: Callable = nn.initializers.lecun_normal()
    training: bool = True

    def setup(self):
        V = self.V + self.order - 1
        self.emb = self.param("emb", self.linear_init, (V, self.emb_dim))
        self.proj = self.param("proj", self.linear_init, (V, self.emb_dim))
        self.conv = nn.Conv(
            features = self.hidden_dim,
            kernel_size = self.order,
        )
        self.conv_dropout = nn.Dropout(self.dropout)
        self.linears = [
            nn.Dense(self.hidden_dim) for _ in range(self.num_layers)
        ]
        self.dropouts = [
            nn.Dropout(self.dropout) for _ in range(self.num_layers)
        ]

    def init_state(self, batch_size=1):
        return jnp.arange(self.V, self.V+self.order-1).tile((batch_size, 1))

    # p(x_t \mid x_<t)
    # NO BOS
    def __call__(self, x, state):
        # concat state (in this case prefix) to text x
        x = jnp.concatenate((state, x), axis=-1)
        emb = self.emb[x]
        y = self.conv_dropout(nn.relu(self.conv(emb)), not self.training)
        for lin, dropout in zip(self.linears, self.dropouts):
            y = dropout(nn.relu(lin(y)), not self.training)
        logits = y @ (self.emb if self.tie_weights else self.proj).T
        log_probs = logits - lse(logits, axis=-1, keepdims=True)
        return log_probs

