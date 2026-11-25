"""Train one model from a YAML config.

    python scripts/train.py configs/duffing.yaml
    python scripts/train.py configs/duffing.yaml --set model.nx=4 model.nz=40 regularization.input.weight=1e-3
"""

import argparse
import logging

from regstab.config import load_config
from regstab.trainer import run


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("config")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="override config entries, e.g. train.lr=1e-3")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config, args.set)
    result = run(cfg)
    print(f"saved to {result['out_dir']}")


if __name__ == "__main__":
    main()
