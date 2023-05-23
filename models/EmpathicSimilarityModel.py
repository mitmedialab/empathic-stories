import sys
sys.path.append("./")

import torch.nn.functional as F

import lightning.pytorch as pl
from sentence_transformers import SentenceTransformer

import pytorch_lightning as pl
from torch.optim import AdamW
from torchmetrics import SpearmanCorrCoef, F1Score
from transformers import get_linear_schedule_with_warmup, AutoTokenizer, AutoModelForSeq2SeqLM, AdamW
from sentence_transformers import SentenceTransformer

class EmpathicSimilarityModel(pl.LightningModule):
    def __init__(self, model="BART", pooling="MEAN", bin=True, learning_rate = 5e-6):
        super(EmpathicSimilarityModel, self).__init__()
        self.base_model = model
        self.pooling = pooling
        self.bin = bin
        self.learning_rate = learning_rate

        if self.base_model == "SBERT":
            self.model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")
            self.tokenizer = self.model.tokenizer
        else:
            self.lm_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")
            self.model = self.lm_model.model
            self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
        
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
                hidden_states = embedding[0] 
                eos_mask = story["input_ids"].eq(self.model.config.eos_token_id).to(hidden_states.device)
                sentence_representation = hidden_states[eos_mask, :].view(hidden_states.size(0), -1, hidden_states.size(-1))[
                    :, -1, :
                ]
        return sentence_representation

    def cls_pooling(self, token_embeddings):
        return token_embeddings[:,0]

    def training_step(self, batch, batch_idx):
        batch = batch[0]
        story1 = batch[0]
        story2 = batch[1]
        score = batch[2]
        if self.bin:
            score = (score>2.5).float().to(self.device)
        sentence_representation1 = self(story1)
        sentence_representation2 = self(story2)
        cos_sim = F.cosine_similarity(sentence_representation1, sentence_representation2)
        loss = F.mse_loss(cos_sim, score)
        self.log("train_loss", loss)
        return loss

    def eval_step(self, batch, batch_idx, prefix):
        story1 = batch[0]
        story2 = batch[1]
        score = batch[2]
        if self.bin:
            score = (score>2.5).float().to(self.device)
        sentence_representation1 = self(story1)
        sentence_representation2 = self(story2)
        cos_sim = F.cosine_similarity(sentence_representation1, sentence_representation2)

        self.f1_score = self.f1_score.to(self.device)
        self.spearman = self.spearman.to(self.device)
        f1 = self.f1_score(cos_sim,score)
        spearman = self.spearman(cos_sim.float(), batch[2])

        loss = F.mse_loss(cos_sim, score)
        self.log(prefix+"_loss", loss)
        return loss
    
    def on_eval_end(self, prefix):
        f1 = self.f1_score.compute()
        spearman = self.spearman.compute()
        self.log(prefix+"_f1", f1)
        self.log(prefix+"_spearman", spearman)

    def on_validation_epoch_end(self):
        self.on_eval_end(prefix = "val")

    def on_test_epoch_end(self):
        self.on_eval_end(prefix = "eval")

    def validation_step(self, batch, batch_idx):
        r = self.eval_step(batch,batch_idx,prefix="val")
        return r

    def test_step(self,batch,batch_idx):
        r = self.eval_step(batch,batch_idx,prefix="eval")
        return r

    def configure_optimizers(self):
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

