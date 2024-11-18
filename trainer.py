import os

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchmetrics import SpearmanCorrCoef, F1Score
from transformers import get_linear_schedule_with_warmup, AutoTokenizer, \
    AutoModelForSeq2SeqLM, BartModel
from sentence_transformers import SentenceTransformer
import argparse 
from model.EmpathicSummaryModel import EmpathicSummaryModel
from omegaconf import OmegaConf


import sys
sys.path.append("./")

class EmpathicSimilarityModel(pl.LightningModule):
    def __init__(self, model="BART", pooling="CLS", bin=True, losses="MSE", use_pretrained=False):
        super(EmpathicSimilarityModel, self).__init__()
        self.base_model = model
        self.pooling = pooling
        self.bin = bin
        self.losses = losses
        self.use_pretrained = use_pretrained
        # Load pre-trained model weights and initialize corresponding tokenizer.
        if self.base_model == "SBERT":
            self.model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")
            self.tokenizer = self.model.tokenizer
        else:
            if self.use_pretrained:
                self.lm_model = EmpathicSummaryModel.load_from_checkpoint("/u/joceshen/socially_connective_dialogue/lightning_logs/version_63/checkpoints/epoch=36-step=5439.ckpt", hp=OmegaConf.load("/u/joceshen/socially_connective_dialogue/config/BART_summary.yaml"))
                self.model = self.lm_model.model
                self.tokenizer = self.lm_model.tokenizer
            else:
                self.lm_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")
                self.model = self.lm_model.model
                self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
        if self.base_model == "SBERT":
            self.learning_rate = 1e-6
        else:
            self.learning_rate = 5e-6
        self.f1_score = F1Score(task="binary")
        self.spearman = SpearmanCorrCoef()

    def forward(self, story):
        story = self.tokenizer(story,padding=True,truncation=True,return_tensors="pt")
        for k in story:
            story[k] = story[k].to(self.device)
        if self.base_model == "SBERT":
            embedding = self.model(story)
            if self.pooling == "MEAN":
                sentence_representation = embedding.sentence_embedding
            else:
                sentence_representation = self.cls_pooling(embedding.token_embeddings)
        else:
            embedding = self.model(**story,output_hidden_states=True) 
            if self.pooling == "MEAN":
                attn = story['attention_mask']
                if self.use_pretrained:
                    sentence_representation = ( embedding.encoder_last_hidden_state * attn.unsqueeze(-1)).sum(-2) / attn.sum(dim=-1).unsqueeze(-1)
                else:
                    sentence_representation = ( embedding.last_hidden_state * attn.unsqueeze(-1)).sum(-2) / attn.sum(dim=-1).unsqueeze(-1)
            else:
                hidden_states = embedding[0]  # last hidden state

                eos_mask = story["input_ids"].eq(self.model.config.eos_token_id).to(hidden_states.device)
                sentence_representation = hidden_states[eos_mask, :].view(hidden_states.size(0), -1, hidden_states.size(-1))[
                    :, -1, :
                ]
        return sentence_representation

    def cls_pooling(self, token_embeddings):
        return token_embeddings[:,0]

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        batch = batch[0]
        story1 = batch[0]
        story2 = batch[1]
        score = batch[2]
        if self.bin:
            score = (score>2.5).float().to(self.device)
        sentence_representation1 = self(story1)
        sentence_representation2 = self(story2)
        loss = 0
        cos_sim = F.cosine_similarity(sentence_representation1, sentence_representation2)
        if "MSE" in self.losses:
            mse_loss = F.mse_loss(cos_sim, score)
            self.log("train_loss (mse)", mse_loss)
            loss += mse_loss
        if "COS" in self.losses:
            pos_labels = (score > 2.5).float()
            neg_labels = -1 * (score <= 2.5).float()
            cos_embedding_loss = F.cosine_embedding_loss(sentence_representation1, sentence_representation2, pos_labels + neg_labels )
            loss += cos_embedding_loss
            self.log("train_loss (cos)", cos_embedding_loss)   
        if "BCE" in self.losses:
            bce_loss = F.binary_cross_entropy_with_logits(cos_sim, score)
            loss += bce_loss 
            self.log("train_loss (bce)", bce_loss)
        if "LM" in self.losses:
            target1 = self.tokenizer(batch[3],padding=True,truncation=True,return_tensors="pt")
            for k in target1:
                target1[k] = target1[k].to(self.device)
            target2 = self.tokenizer(batch[4],padding=True,truncation=True,return_tensors="pt")
            for k in target2:
                target2[k] = target2[k].to(self.device)
            tokens1 = self.tokenizer(story1,padding=True,truncation=True,return_tensors="pt")
            for k in tokens1:
                tokens1[k] = tokens1[k].to(self.device)
            tokens2 = self.tokenizer(story2,padding=True,truncation=True,return_tensors="pt")
            for k in tokens2:
                tokens2[k] = tokens2[k].to(self.device)
            output1 = self.model(**tokens1, labels=target1["input_ids"], output_hidden_states=True)
            output2 = self.model(**tokens2, labels=target2["input_ids"], output_hidden_states=True)
            loss1 = output1.loss
            loss2 = output2.loss
            lm_losses = loss1 + loss2
            loss += lm_losses
            self.log("train_loss (lm)", lm_losses)

        self.log("train_loss", loss)
        return loss

    def eval_step(self,batch,batch_idx,prefix):
        # training_step defines the train loop.
        # it is independent of forward
        # batch = batch[0]
        story1 = batch[0]
        story2 = batch[1]
        score = batch[2]
        if self.bin or "BCE" in self.losses:
            score = (score>2.5).float().to(self.device)
        sentence_representation1 = self(story1)
        sentence_representation2 = self(story2)
        loss = 0
        cos_sim = F.cosine_similarity(sentence_representation1, sentence_representation2)
        self.f1_score = self.f1_score.to(self.device)
        self.spearman = self.spearman.to(self.device)
        f1 = self.f1_score(cos_sim,score)
        spearman = self.spearman(cos_sim.float(), batch[2])
        
        if "MSE" in self.losses:
            mse_loss = F.mse_loss(cos_sim, score)
            self.log(prefix+"_loss (mse)", mse_loss)
            loss += mse_loss
        if "COS" in self.losses:
            pos_labels = (score > 2.5).float()
            neg_labels = -1 * (score <= 2.5).float()
            cos_embedding_loss = F.cosine_embedding_loss(sentence_representation1, sentence_representation2, pos_labels + neg_labels )
            loss += cos_embedding_loss
            self.log(prefix+"_loss (cos)", cos_embedding_loss)   
        if "BCE" in self.losses:
            bce_loss = F.binary_cross_entropy(cos_sim, score)
            loss += bce_loss 
            self.log(prefix+"_loss (bce)", bce_loss)
        if "LM" in self.losses:
            target1 = self.tokenizer(batch[3],padding=True,truncation=True,return_tensors="pt")
            for k in target1:
                target1[k] = target1[k].to(self.device)
            target2 = self.tokenizer(batch[4],padding=True,truncation=True,return_tensors="pt")
            for k in target2:
                target2[k] = target2[k].to(self.device)
            tokens1 = self.tokenizer(story1,padding=True,truncation=True,return_tensors="pt")
            for k in tokens1:
                tokens1[k] = tokens1[k].to(self.device)
            tokens2 = self.tokenizer(story2,padding=True,truncation=True,return_tensors="pt")
            for k in tokens2:
                tokens2[k] = tokens2[k].to(self.device)
            output1 = self.model(**tokens1, labels=target1["input_ids"], output_hidden_states=True)
            output2 = self.model(**tokens2, labels=target2["input_ids"], output_hidden_states=True)
            loss1 = output1.loss
            loss2 = output2.loss
            lm_losses = loss1 + loss2
            loss += lm_losses
            self.log(prefix+"_loss (lm)", lm_losses)
            

        self.log(prefix+"_loss", loss)
        return loss
    
    def on_eval_end(self,prefix):
        f1 = self.f1_score.compute()
        spearman = self.spearman.compute()
        self.log(prefix+"_f1", f1)
        self.log(prefix+"_spearman", spearman)

    def on_validation_epoch_end(self):
        self.on_eval_end(prefix="val")

    def on_test_epoch_end(self):
        self.on_eval_end(prefix="eval")

    def validation_step(self, batch, batch_idx):
        r = self.eval_step(batch,batch_idx,prefix="val")
        return r

    def test_step(self,batch,batch_idx):
        r = self.eval_step(batch,batch_idx,prefix="eval")
        return r

    def configure_optimizers(self):
        # optimizer = optim.Adam(self.parameters(), lr=self.hp.train.adam.lr)
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        if self.base_model == "SBERT":
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=0.5 * self.trainer.estimated_stepping_batches,
                num_training_steps=self.trainer.estimated_stepping_batches,
            )
        else:
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=0.1 * self.trainer.estimated_stepping_batches,
                num_training_steps=self.trainer.estimated_stepping_batches,
            )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]


class EmpathicStoriesDataset(torch.utils.data.Dataset):
    def __init__(self, data, data_stories, limit=-1, use_pretrained = False):
        self.data = data
        self.data_stories = data_stories
        self.limit = limit
        self.use_pretrained = use_pretrained

    def __getitem__(self, idx):
        i = self.data.iloc[idx]
        s1 = i["story_A"].replace("\n", "")
        s2 = i["story_B"].replace("\n", "")
        score = i["similarity_empathy_human_AGG"]
        pair_id = eval(i["pairs"])
        event1 = self.data_stories.iloc[pair_id[0]]["Main Event"]
        event2 = self.data_stories.iloc[pair_id[1]]["Main Event"]
        emotion1 = self.data_stories.iloc[pair_id[0]]["Emotion Description"]
        emotion2 = self.data_stories.iloc[pair_id[1]]["Emotion Description"]
        moral1 = self.data_stories.iloc[pair_id[0]]["Moral"]
        moral2 = self.data_stories.iloc[pair_id[1]]["Moral"]
        s1_summary = "[EVE]" + event1 + "[EMO]" + emotion1 + "[MOR]" + moral1
        s2_summary = "[EVE]" + event2 + "[EMO]" + emotion2 + "[MOR]" + moral2

        if self.use_pretrained:
            return [s1, s2, score, s1_summary, s2_summary]
        else:
            return [s1, s2, score]

    def __len__(self):
        if self.limit!=-1:
            return self.limit
        return len(self.data)

os.environ["WANDB_API_KEY"] = "2aa34769ea8ae63060a052e485040b683333dfed"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, required=True,
                        help="model")
    parser.add_argument('-p', '--pooling', type=str, required=True,
                        help="pooling")
    parser.add_argument('-e', '--epochs', type=str, required=True,
                        help="epochs")
    parser.add_argument('-b', '--bin', type=str, required=True,
                        help="bin")
    parser.add_argument('-l', '--losses', type=str, required=True,
                        help="losses")
    parser.add_argument('-u', '--use_pretrained', type=str, required=True,
                        help="use_pretrained")
    args = parser.parse_args()
    train_d = pd.read_csv("data/PAIRS (train).csv")
    train_d2 = pd.read_csv("data/STORIES (train).csv")
    train_ds = EmpathicStoriesDataset(train_d, train_d2, use_pretrained=args.use_pretrained)
    train_dl = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_d = pd.read_csv("data/PAIRS (dev).csv")
    val_d2 = pd.read_csv("data/STORIES (dev).csv")
    val_ds = EmpathicStoriesDataset(val_d, val_d2, use_pretrained=args.use_pretrained)
    val_dl = DataLoader(val_ds, batch_size=8, shuffle=False)
    test_d = pd.read_csv("data/PAIRS (test).csv")
    test_d2 = pd.read_csv("data/STORIES (test).csv")
    test_ds = EmpathicStoriesDataset(test_d, test_d2, use_pretrained=args.use_pretrained)
    test_dl = DataLoader(test_ds, batch_size=8, shuffle=False)

    lr_monitor = LearningRateMonitor(logging_interval='step')
    spearman_callback = ModelCheckpoint(save_top_k=1, monitor="val_spearman", mode="max")
    # wandb_logger = WandbLogger("jocelyn_struggle", project="jocelynstuff")

    model = EmpathicSimilarityModel(
        model=args.model, 
        pooling = args.pooling, 
        bin =bool(args.bin),
        losses=args.losses,
        use_pretrained = bool(args.use_pretrained)
    )
    precision = 16
    trainer = pl.Trainer(
        log_every_n_steps=5,
        max_epochs=int(args.epochs),
        accelerator="gpu",
        callbacks=[lr_monitor, spearman_callback],
        precision=precision,
        strategy=DDPStrategy(find_unused_parameters=True)
        # logger=wandb_logger

    )
    trainer.fit(model=model,train_dataloaders=[train_dl], val_dataloaders = [val_dl])
    trainer.test(model=model,dataloaders=[test_dl])
