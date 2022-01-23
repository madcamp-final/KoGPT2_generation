import torch
import os
import argparse
import random
from pytorch_lightning import Trainer
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint

# from model import KoGPT2IdeaModel
from idea_generation.model import KoGPT2IdeaModel

os.environ["CUDA_VISIBLE_DEVICES"]= "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device:', device)
print('Current cuda device:', torch.cuda.current_device())
print('Count of using GPUs:', torch.cuda.device_count())


class KoGPT2IdeaGenerator(LightningModule):
    def __init__(self, hparams, **kwargs):
        super(KoGPT2IdeaGenerator, self).__init__()
        self.hparams = hparams
        self.neg = -1e18

        model = KoGPT2IdeaModel(self.hparams)
        self.model = model.load_from_checkpoint(self.hparams.model_params)
        self.model.eval()
        trainer = Trainer(gpus=1)
        trainer.test(self.model)

    def generate(self, category_content):
        result = self.model.idea_maker(category_content)
        return result

    def generate_nbest_ideas(self, category_content):
        result = self.model.nbest_ideas_maker(category_content)
        final_result = random.choice(result)
        return final_result

