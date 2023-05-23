import sys
sys.path.append("./")

import pandas as pd
import argparse
from model.EmpathicSimilarityModel import EmpathicSimilarityModel
from model.EmpathicSummaryModel import EmpathicSummaryModel
from dataset import EmpathicStoriesDataset
from omegaconf import OmegaConf
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader
from pytorch_lightning.strategies import DDPStrategy

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="yaml file for configuration")
    parser.add_argument('-p', '--checkpoint_path', type=str, default=None,
                        help="path of checkpoint pt file to resume training")
    args = parser.parse_args()

    hp = OmegaConf.load(args.config)
    with open(args.config, 'r') as f:
        hp_str = ''.join(f.readlines())

    train_d = pd.read_csv(hp.data.train.filename_pairs)
    train_d2 = pd.read_csv(hp.data.train.filename_stories)
    train_ds = EmpathicStoriesDataset(train_d, train_d2)
    train_dl = DataLoader(train_ds, hp.train.batch_size, shuffle=hp.train.shuffle)

    val_d = pd.read_csv(hp.data.val.filename_pairs)
    val_d2 = pd.read_csv(hp.data.val.filename_stories)
    val_ds = EmpathicStoriesDataset(val_d, val_d2)
    val_dl = DataLoader(val_ds, batch_size=hp.val.batch_size, shuffle=hp.val.shuffle)

    test_d = pd.read_csv(hp.data.test.filename_pairs)
    test_d2 = pd.read_csv(hp.data.test.filename_stories)
    test_ds = EmpathicStoriesDataset(test_d, test_d2)
    test_dl = DataLoader(test_ds, batch_size=hp.test.batch_size, shuffle=hp.test.shuffle)

    lr_monitor = LearningRateMonitor(logging_interval='step')
    spearman_callback = ModelCheckpoint(save_top_k=1, monitor="val_spearman", mode="max")

    if hp.task == "similarity":
        model = EmpathicSimilarityModel(
            model=hp.model, 
            pooling = hp.pooling, 
            bin = bool(hp.bin),
            learning_rate = hp.train.adam.lr
        )
    elif hp.task == "summary":
        model = EmpathicSummaryModel(hp)

    precision = hp.train.precision
    trainer = pl.Trainer(
        log_every_n_steps = 5,
        max_epochs= hp.train.epochs,
        accelerator = "gpu",
        callbacks = [lr_monitor, spearman_callback],
        precision = precision,
        strategy = DDPStrategy(find_unused_parameters=True)
    )
    trainer.fit(model = model, train_dataloaders = [train_dl], val_dataloaders = [val_dl], ckpt_path = args.checkpoint_path)
    trainer.test(model = model, dataloaders = [test_dl])


    