import os
import time

import numpy as np
import torch
from pathlib import Path
from copy import deepcopy
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_


class Trainer(object):
    def __init__(self, config, args, model, dataloader, eval_dataloader=None, logger=None):
        super().__init__()
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.args = args
        self.config = config
        self.logger = logger

        self.train_loader = dataloader["dataloader"]
        self.train_dataset = dataloader["dataset"]
        self.eval_loader = None if eval_dataloader is None else eval_dataloader["dataloader"]
        self.eval_dataset = None if eval_dataloader is None else eval_dataloader["dataset"]

        self.epochs = config["solver"]["max_epochs"]
        self.save_cycle = 100
        self.step = 0
        self.milestone = 0

        base_lr = config["solver"].get("base_lr", 1.0e-4)
        self.ema_decay = float(config["solver"].get("ema", {}).get("decay", 0.995))
        self.opt = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=base_lr, betas=(0.9, 0.999))
        self.ema_model = deepcopy(self.model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)
        self.results_folder = Path(args.save_dir)
        self.results_folder.mkdir(parents=True, exist_ok=True)

    def save(self, milestone):
        data = {
            "step": self.step,
            "model": self.model.state_dict(),
            "ema_model": self.ema_model.state_dict(),
            "opt": self.opt.state_dict(),
            "ema_decay": self.ema_decay,
        }
        torch.save(data, str(self.results_folder / f"checkpoint-{milestone}.pt"))

    def load(self, milestone):
        data = torch.load(str(self.results_folder / f"checkpoint-{milestone}.pt"), map_location=self.device)
        self.model.load_state_dict(data["model"])
        if "ema_model" in data:
            self.ema_model.load_state_dict(data["ema_model"])
        self.opt.load_state_dict(data["opt"])
        self.step = data["step"]
        self.milestone = milestone

    @torch.no_grad()
    def _update_ema(self):
        online_state = self.model.state_dict()
        ema_state = self.ema_model.state_dict()
        for k in ema_state.keys():
            ema_state[k].mul_(self.ema_decay).add_(online_state[k], alpha=1.0 - self.ema_decay)

    def train(self):
        tic = time.time()
        for epoch in range(self.epochs):
            self.model.train()
            epoch_losses = []
            for batch in self.train_loader:
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                x = x.to(self.device).float()
                loss = self.model(x)
                self.opt.zero_grad()
                loss.backward()
                clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()
                self._update_ema()
                self.step += 1
                epoch_losses.append(float(loss.item()))

            mean_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
            if self.logger is not None:
                self.logger.log_info(f"epoch={epoch} train_loss={mean_loss:.6f}")
                self.logger.add_scalar(tag="train/loss", scalar_value=mean_loss, global_step=epoch)

            if (epoch + 1) % self.save_cycle == 0 or (epoch + 1) == self.epochs:
                self.milestone += 1
                self.save(self.milestone)
                if self.eval_dataset is not None:
                    sample0 = self.eval_dataset[0]
                    if isinstance(sample0, (tuple, list)):
                        sample0 = sample0[0]
                    seq_len = int(sample0.shape[0])
                    n_feat = int(sample0.shape[-1]) if sample0.ndim > 1 else 1
                    self.sample_to_file(
                        num=len(self.eval_dataset),
                        shape=[seq_len, n_feat],
                        output_name=f"ddpm_fake_{self.args.name}_{self.args.long_len}_epoch{epoch+1}_valid.npy",
                    )

        if self.logger is not None:
            self.logger.log_info(f"training complete, elapsed={time.time() - tic:.2f}s")

    @torch.no_grad()
    def sample(self, num, size_every=2001, shape=None):
        self.model.eval()
        if shape is None:
            raise ValueError("shape must be provided")
        samples = np.empty((0, shape[0], shape[1]), dtype=np.float32)
        remain = int(num)
        while remain > 0:
            bs = min(size_every, remain)
            sample = self.model.generate_mts(batch_size=bs).detach().cpu().numpy()
            samples = np.row_stack([samples, sample])
            remain -= bs
        return samples

    def sample_to_file(self, num, shape, output_name):
        samples = self.sample(num=num, shape=shape)
        np.save(os.path.join(self.args.save_dir, output_name), samples)
