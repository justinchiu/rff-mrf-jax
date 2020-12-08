
from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset",
        choices=["ptb", "wikitext2"], default="ptb")
    parser.add_argument("--iterator", choices=["bucket", "bptt"], default="bucket")
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
    return parser.parse_args()

