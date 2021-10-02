from tqdm import tqdm
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.progress import ProgressBar
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from .evaluation import test_ratings, test_recommendations, uncertainty_distributions


class LitProgressBar(ProgressBar):
    def init_validation_tqdm(self):
        bar = tqdm(disable=True)
        return bar


def train(model, data):
    prog_bar = LitProgressBar()
    es = EarlyStopping(monitor='val_loss', min_delta=0, patience=4, verbose=False, mode='min')
    trainer = Trainer(gpus=1, max_epochs=200, logger=False, callbacks=[prog_bar, es], checkpoint_callback=False)
    trainer.fit(model, datamodule=data)
