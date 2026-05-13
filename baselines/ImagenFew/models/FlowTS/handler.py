import torch
from torch.optim import Adam
from omegaconf import OmegaConf

from models.generative_handler import generativeHandler
from models.interpretable_diffusion.ema import LitEma
from utils.io_utils import instantiate_from_config


class Handler(generativeHandler):
    def __init__(self, args, rank=None):
        self.config = OmegaConf.to_object(OmegaConf.load(args.config))
        super().__init__(args, rank)

        base_lr = self.config["solver"].get("base_lr", 1e-4)
        self.optimizer = Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=base_lr,
            betas=(0.9, 0.999),
        )
        self.ema = LitEma(self.model, decay=0.995, use_num_upates=True, warmup=args.ema_warmup)
        self.step = 0

    def build_model(self):
        return instantiate_from_config(self.config["model"])

    def train_iter(self, train_dataloader, logger):
        for _, data in enumerate(train_dataloader, 1):
            x = data[0].to(self.device).float()

            loss = self.model(x)
            loss.backward()

            logger.log("train/loss", loss.item(), self.step)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.step += 1
            self.ema(self.model)

    def sample(self, n_samples, class_label=None, class_metadata=None):
        with torch.no_grad():
            with self.ema_scope():
                return self.model.generate_mts(batch_size=n_samples)

    def ema_scope(self):
        class EMAScope:
            def __init__(self, model, ema):
                self.model = model
                self.ema = ema

            def __enter__(self):
                self.ema.store(self.model.parameters())
                self.ema.copy_to(self.model)

            def __exit__(self, exc_type, exc_value, traceback):
                self.ema.restore(self.model.parameters())

        return EMAScope(self.model, self.ema)

