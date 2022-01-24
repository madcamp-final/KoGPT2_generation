import torch
import os
import argparse
import random
import pandas as pd
import csv
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
        # f=open('./skt_dataset.csv', encoding='utf8')
        # reader = csv.reader(f)
        # csv_list = []
        # for l in reader:
        #     csv_list.append(l)
        # f.close()
        # data = pd.DataFrame(csv_list)
        # # data = pd.read_csv('/root/week4/KoGPT2_generation/idea_generation/skt_dataset.csv', encoding='utf-8')
        # print(data)

        # find_row = []
        # for k in range(len(data)):
        #     if(data[0][k]!=category_content):
        #         find_row.append(k)
        #     # find_row = data.index[(data['Q'][k]!=category_content)].tolist()
        # print(find_row)
        # for i in range(len(result)):
        #     for j in find_row:
        #         if(result[i]==data[1][j]):
        #             del result[i]
        # print(result)
        final_result = random.choice(result)
        return final_result
    
    def generate_temp_ideas(self, category_content):
        result = self.model.temperature_idea_maker(category_content)
        return result
        
