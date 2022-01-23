import torch
import os
import argparse
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
    # def __init__(self, args):
    #     checkpoint_callback = ModelCheckpoint(
    #         dirpath='model_chp',
    #         filename='{epoch:02d}-{train_loss:.2f}',
    #         verbose=True,
    #         save_last=True,
    #         monitor='train_loss',
    #         mode='min',
    #         prefix='model_'
    #     )

    #     model = KoGPT2IdeaModel(args)
    #     self.model = model.load_from_checkpoint(args.model_params)
    #     self.model.eval()
    #     trainer = Trainer.from_argparse_args(
    #         args,
    #         checkpoint_callback=checkpoint_callback, gradient_clip_val=1.0)
    #     trainer.test(self.model)



    def __init__(self, hparams, **kwargs):
        super(KoGPT2IdeaGenerator, self).__init__()
        self.hparams = hparams
        self.neg = -1e18

        checkpoint_callback = ModelCheckpoint(
            dirpath='model_chp',
            filename='{epoch:02d}-{train_loss:.2f}',
            verbose=True,
            save_last=True,
            monitor='train_loss',
            mode='min',
            prefix='model_'
        )

        model = KoGPT2IdeaModel(self.hparams)
        self.model = model.load_from_checkpoint(self.hparams.model_params)
        self.model.eval()
        trainer = Trainer(gpus=1)
        # trainer = Trainer.from_argparse_args(
        #     self.hparams,
        #     checkpoint_callback=checkpoint_callback, gradient_clip_val=1.0, gpus=1)
        trainer.test(self.model)
        # trainer = Trainer(gpus=1)
        # trainer.test(self.model)
        # self.device = torch.device("cuda")
        # self.model.to(self.device)

    # def __init__(self, args) -> None:
    #     # model = KoGPT2IdeaModel(args)
    #     # parser = argparse.ArgumentParser(description='Idea based on KoGPT-2')
    #     # parser = KoGPT2IdeaModel.add_model_specific_args(args)
    #     # parser = Trainer.add_argparse_args(parser)
    #     # args = parser.parse_args()
    #     # parser = Trainer.add_argparse_args(args.gpus)
    #     model = KoGPT2IdeaModel(args)
    #     self.model = model.load_from_checkpoint(args.model_params)
    #     # self.model.to(device)
    #     # self.model.eval()
    #     # self.model = KoGPT2IdeaModel.load_from_checkpoint(args.model_params)

    def generate(self, category_content):
        result = self.model.idea_maker(category_content)
        return result
        