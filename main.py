from train import Trainer
from option import args

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def main():
    if args.test_only:
        t = Trainer()
        t.validation(args.ep)
    else:
        t = Trainer()
        t.train()


if __name__ == '__main__':
    main()
