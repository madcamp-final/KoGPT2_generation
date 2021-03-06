import argparse
import logging
from unittest.util import _MAX_LENGTH
import gc
import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.lightning import LightningModule
from torch.utils.data import DataLoader, Dataset
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
import tensorflow as tf

# from dataset import IdeaDataset
from idea_generation.dataset import IdeaDataset

gc.collect()
torch.cuda.empty_cache()

U_TKN = '<usr>'
S_TKN = '<sys>'
BOS = '</s>'
EOS = '</s>'
MASK = '<unused0>'
SENT = '<unused1>'
PAD = '<pad>'

class KoGPT2IdeaModel(LightningModule):
    def __init__(self, hparams, **kwargs):
        super(KoGPT2IdeaModel, self).__init__()
        self.hparams = hparams
        self.neg = -1e18

        self.model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')
        # self.device = torch.device("cuda")
        # self.model.to(self.device)

        self.tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
            bos_token=BOS, eos_token=EOS, unk_token='<unk>',
            pad_token=PAD, mask_token=MASK)
        self.loss_function = torch.nn.CrossEntropyLoss(reduction='none')

    @staticmethod
    def add_model_specific_args(parent_parser):
        # add model specific args
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--max-len',
                            type=int,
                            default=128,
                            help='max sentence length on input (default: 128)')

        parser.add_argument('--batch-size',
                            type=int,
                            default=64,
                            help='batch size for training (default: 64)')
        parser.add_argument('--lr',
                            type=float,
                            default=5e-5,
                            help='The initial learning rate')
        parser.add_argument('--warmup_ratio',
                            type=float,
                            default=0.1,
                            help='warmup ratio')
        return parser

    def forward(self, inputs):
        # (batch, seq_len, hiddens)
        # inputs = inputs.to(self.device)
        output = self.model(inputs, return_dict=True)
        return output.logits

    def training_step(self, batch, batch_idx):
        token_ids, mask, label = batch
        out = self(token_ids)
        mask_3d = mask.unsqueeze(dim=2).repeat_interleave(repeats=out.shape[2], dim=2)
        mask_out = torch.where(mask_3d == 1, out, self.neg * torch.ones_like(out))
        loss = self.loss_function(mask_out.transpose(2, 1), label)
        loss_avg = loss.sum() / mask.sum()
        self.log('train_loss', loss_avg)
        return loss_avg

    def configure_optimizers(self):
        # Prepare optimizer
        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparams.lr, correct_bias=False)
        # warm up lr
        num_train_steps = len(self.train_dataloader()) * self.hparams.max_epochs
        num_warmup_steps = int(num_train_steps * self.hparams.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)
        lr_scheduler = {'scheduler': scheduler, 'name': 'cosine_schedule_with_warmup',
                        'monitor': 'loss', 'interval': 'step',
                        'frequency': 1}
        return [optimizer], [lr_scheduler]

    def _collate_fn(self, batch):
        data = [item[0] for item in batch]
        mask = [item[1] for item in batch]
        label = [item[2] for item in batch]
        return torch.LongTensor(data), torch.LongTensor(mask), torch.LongTensor(label)

    def train_dataloader(self):
        # data = pd.read_csv('./skt_dataset.csv')
        data = pd.read_csv('./shuffled_dataset.csv')
        self.train_set = IdeaDataset(data, max_len=self.hparams.max_len)
        train_dataloader = DataLoader(
            self.train_set, batch_size=self.hparams.batch_size, num_workers=3,
            shuffle=True, collate_fn=self._collate_fn)
        return train_dataloader

    def idea_maker(self, category):
        sent='0'
        tok = self.tokenizer
        sent_tokens = tok.tokenize(sent)
        with torch.no_grad():
            a = ''
            while 1:
                input_ids = torch.LongTensor(tok.encode(U_TKN + category + SENT + sent + S_TKN + a)).unsqueeze(dim=0).cuda()
                pred = self(input_ids).cuda()
                gen = tok.convert_ids_to_tokens(
                    torch.argmax(
                        pred,
                        dim=-1).squeeze().cpu().numpy().tolist())[-1]
                if gen == EOS:
                    break
                a += gen.replace('???', ' ')

            a.replace(' ', '')
            return a

    def nbest_ideas_maker(self, category):
        # input_ids = self.tokenizer.encode(category, return_tensors='tf')
        input_ids = torch.LongTensor(self.tokenizer.encode(BOS + category + EOS)).unsqueeze(dim=0).cuda()
        beam_outputs = self.model.generate(
            input_ids, 
            max_length=128, 
            num_beams=10, 
            no_repeat_ngram_size=2, 
            num_return_sequences=10,
            early_stopping=True
        )

        print("Output:nbest\n" + 100 * '-')
        for i, beam_output in enumerate(beam_outputs):
            print("{}: {}".format(i, self.tokenizer.decode(beam_output, skip_special_tokens=True)))        
        # result = self.tokenizer.decode(beam_output[0], skip_special_tokens=True)

        result = []
        for i in range(0, 10):
            a = self.tokenizer.batch_decode(beam_outputs.tolist(), skip_special_tokens=True)[i]
            idea = a.replace(category+' ', '')
            result.append(idea)
        return result

    def temperature_idea_maker(self, category):
        input_ids = torch.LongTensor(self.tokenizer.encode(BOS + category + EOS)).unsqueeze(dim=0).cuda()
        temp_outputs = self.model.generate(
            input_ids,
            do_sample=True,
            max_length = 128,
            top_k=0,
            temperature=0.8
        )
        result = self.tokenizer.decode(temp_outputs[0], skip_special_tokens = True)
        final_result = result.replace(category+' ', '')
        print("Output:temperature\n" + 100 * '-')
        print(final_result)

        return final_result
