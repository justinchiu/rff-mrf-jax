
from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset",
        choices=["ptb", "wikitext2"], default="ptb")
    parser.add_argument("--iterator", choices=["bucket", "bptt"], default="bucket")
    parser.add_argument("--max_len", default=0, type=int,)
    # learning args
    parser.add_argument("--bsz", default=256, type=int,)
    parser.add_argument("--eval_bsz", default=256, type=int,)
    parser.add_argument("--bsz_fn", choices=["tokens", "sentences"], default="tokens",)
    parser.add_argument("--lr", default=1e-3, type=float,)
    parser.add_argument("--clip", default=5, type=float,)
    parser.add_argument("--beta1", default=0.9, type=float,)
    parser.add_argument("--beta2", default=0.999, type=float,)
    parser.add_argument("--wd", default=0.000, type=float,)
    parser.add_argument("--decay", default=4, type=float,)
    parser.add_argument("--optimizer",
        choices=["adamw", "sgd"],
        default="adamw",)
    parser.add_argument("--no_shuffle_train", action="store_true")

    # model args
    # 
    parser.add_argument("--num_layers", default=1, type=int,)
    parser.add_argument("--emb_dim", default=256, type=int,)
    parser.add_argument("--hidden_dim", default=256, type=int,)
    parser.add_argument("--dropout", default=0, type=float,)
    parser.add_argument("--tie_weights", default=0, type=int,)
    parser.add_argument("--order", default=5, type=int,)
    return parser.parse_args()

