from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import torch
from Utils.io_utils import write_args, save_config_to_yaml


class Logger(object):
    def __init__(self, args):
        self.args = args
        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.config_dir = os.path.join(self.save_dir, "configs")
        os.makedirs(self.config_dir, exist_ok=True)
        write_args(args, os.path.join(self.config_dir, "args.txt"))

        log_dir = os.path.join(self.save_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        self.text_writer = open(os.path.join(log_dir, "log.txt"), "a")
        self.tb_writer = torch.utils.tensorboard.SummaryWriter(log_dir=log_dir) if args.tensorboard else None

    def save_config(self, config):
        save_config_to_yaml(config, os.path.join(self.config_dir, "config.yaml"))

    def log_info(self, info):
        print(info)
        time_str = time.strftime("%Y-%m-%d-%H-%M")
        line = f"{time_str}: {info}"
        if not line.endswith("\n"):
            line += "\n"
        self.text_writer.write(line)
        self.text_writer.flush()

    def add_scalar(self, **kwargs):
        if self.tb_writer is not None:
            self.tb_writer.add_scalar(**kwargs)
