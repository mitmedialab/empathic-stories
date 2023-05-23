import sys
sys.path.append("./")

import lightning.pytorch as pl
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup, AdamW

from special_tokens import SPECIAL_TOKENS

class EmpathicSummaryModel(pl.LightningModule):
    def __init__(self, hp):
        super(EmpathicSummaryModel, self).__init__()
        self.hp = hp

        # Load pre-trained model weights and initialize corresponding tokenizer.
        if hp.model == 'FlanT5':
            self.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
            self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        elif hp.model == "BART":
            self.model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")
            self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
        else:
            raise Exception("Choose a model that has been implemented")
        
        # Add special tokens and resize model vocabulary.
        self.tokenizer.add_special_tokens(SPECIAL_TOKENS)
        self.model.resize_token_embeddings(len(self.tokenizer))

    def forward(self, input, target):
        story_tokens = self.get_tokens(input)
        label_tokens = self.get_tokens(target)
        return self.model(**story_tokens, labels=label_tokens["input_ids"], output_hidden_states = True)
    
    def get_tokens(self, x):
        tokens = self.tokenizer(
            x, 
            return_tensors = "pt",
            truncation=True, 
            padding="max_length",
            return_attention_mask = True,
            max_length=self.tokenizer.model_max_length
        )
        for key in tokens:
            tokens[key] = tokens[key].squeeze(0)
        return tokens

    def generate(self, x, max_new_tokens = 128, num_beams = 10):
        output_sequences = self.model.generate(
            **x, 
            return_dict_in_generate=True, 
            output_hidden_states=True, 
            top_k=120, 
            top_p=0.9, 
            max_new_tokens=max_new_tokens, 
            num_beams = num_beams, 
            repetition_penalty = 1.5,
            output_scores = True
        ) 
        return output_sequences
             
    def training_step(self, batch, batch_idx):
        story, label = batch
        outputs = self(story, label) 
        loss = outputs.loss
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        story, label = batch
        outputs = self(story, label) 
        loss = outputs.loss
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr = self.hp.train.adam.lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0.1 * self.trainer.estimated_stepping_batches,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]
