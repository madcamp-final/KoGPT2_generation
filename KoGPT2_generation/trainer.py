import argparse
import logging

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from model import KoGPT2IdeaModel

class KoGPT2IdeaTrainer():
    def __init__(self, args) -> None:
        self.args = args
        pass

    def train(self):
        checkpoint_callback = ModelCheckpoint(
            dirpath='model_chp',
            filename='{epoch:02d}-{train_loss:.2f}',
            verbose=True,
            save_last=True,
            monitor='train_loss',
            mode='min',
            prefix='model_'
        )
        # python train_torch.py --train --gpus 1 --max_epochs 3
        model = KoGPT2IdeaModel(self.args)
        model.train()
        trainer = Trainer.from_argparse_args(
            self.args,
            checkpoint_callback=checkpoint_callback, gradient_clip_val=1.0)
        trainer.fit(model)
        logging.info('best model path {}'.format(checkpoint_callback.best_model_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Idea based on KoGPT-2')
    parser = KoGPT2IdeaModel.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    ideaTrainer = KoGPT2IdeaTrainer(args)
    ideaTrainer.train()

#  python trainer.py --gpus 1 --max_epochs 15 --max-len 64 --batch-size 32

