import contextlib
from pathlib import Path
# import hydra
import numpy as np
from omegaconf import OmegaConf
import os
import random
import torch
import yaml

def seed_everything(seed: int) -> None:
    """Seed everything

    This function sets the seed for random number generators in various libraries to ensure reproducibility.

    Args:
        seed (int): The seed value to be used for seeding the random number generators.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        # torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    tf.compat.v1.set_random_seed(1234)
    tf.random.set_seed(1234)

def makedir(path: str) -> None:
    """Create a directory if it doesn't exist.

    Args:
        path (str): The path to the directory.
    """
    if not os.path.exists(path):
        os.makedirs(path)


# def create_exp_dir(cfg: OmegaConf) -> None:
#     """Create the directory for the experiment/run
#
#     Args:
#         cfg (OmegaConf): config file for experiment
#     """
#     cfg.output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
#     cfg.exp_name = os.path.basename(cfg.output_dir)
#     makedir(cfg.output_dir)
#     with open(os.path.join(cfg.output_dir, 'cfg.yaml'), 'w') as f:
#         yaml.dump(OmegaConf.to_yaml(cfg), f)


def save_checkpoint(
    state: dict,
    save_dir: str,
    chkp_name="checkpoint.pth.tar"
) -> None:
    """Save a model checkpoint.

    Args:
        state (dict): The state of a neural network.
        save_dir (str): The directory where the checkpoint will be saved.
        chkp_name (str, optional): The name of the checkpoint file. Defaults to "checkpoint.pth.tar".
    """
    filename = os.path.join(save_dir, chkp_name)
    torch.save(state, filename)


def load_checkpoint(
    model: torch.nn.Module,
    pretrained_path: str,
    optimizer: torch.optim.Optimizer = None,
    best_model: bool = True
):
    """Load a model checkpoint.

    Args:
        model (torch.nn.Module): The model with random weights.
        pretrained_path (str): The path to the pretrained experiment.
        optimizer (torch.optim.Optimizer, optional): The optimizer. Defaults to None.
        best_model (bool, optional): Whether to load the best model checkpoint. Defaults to True.

    Returns:
        model (torch.nn.Module): The model with pretrained weights loaded.
        optimizer (torch.optim.Optimizer): The optimizer with loaded state if provided.
        epoch (int): The epoch number from the checkpoint.
    """
    chck_name = "best_checkpoint.pth.tar" if best_model else "checkpoint.pth.tar"
    checkpoint = torch.load(os.path.join(pretrained_path, chck_name))

    # little hack to cope with torch.compile and multi-GPU training
    sd = "state_dict_ema" if "state_dict_ema" in checkpoint else "state_dict"
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint[sd].items()}
    state_dict = {k.replace("module.", ""): v for k, v in checkpoint[sd].items()}

    model.load_state_dict(state_dict)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        return model, optimizer, checkpoint["epoch"]
    else:
        return model, checkpoint["epoch"]

def supress_stdout(func):
    """
    A decorator that suppresses the standard output (stdout) of a function.

    Args:
        func: The function to be decorated.

    Returns:
        The decorated function.
    """
    def wrapper(*a, **ka):
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                return func(*a, **ka)
    return wrapper

class ConfigLoader:
    @staticmethod
    def load_vq_config(config):
        project_path =os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        conf_path = OmegaConf.from_cli().get('conf_path', config)
        conf = OmegaConf.load(os.path.join(project_path, conf_path))
        return conf
    
    @staticmethod
    def load_var_config(config):
        project_path =os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        conf_path = OmegaConf.from_cli().get('conf_path', config)
        conf = OmegaConf.load(os.path.join(project_path, conf_path))
        return conf

def load_model_path(root=None, version=None, v_num=None, best=True):
    """ When best = True, return the best model's path in a directory 
        by selecting the best model with largest epoch. If not, return
        the last model saved. You must provide at least one of the 
        first three args.
    Args: 
        root: The root directory of checkpoints. It can also be a
            model ckpt file. Then the function will return it.
        version: The name of the version you are going to load.
        v_num: The version's number that you are going to load.
        best: Whether return the best model.
    """
    def sort_by_step(path):
        name = path.stem
        step=int(name.split('-')[-1].split('=')[1])
        return step


    def sort_by_epoch(path):
        name = path.stem
        epoch=int(name.split('-')[1].split('=')[1])
        return epoch
    
    def generate_root():
        if root is not None:
            return root
        elif version is not None:
            return str(Path('lightning_logs', version, 'checkpoints'))
        else:
            return str(Path('lightning_logs', f'version_{v_num}', 'checkpoints'))

    if root==version==v_num==None:
        return None

    root = generate_root()
    if Path(root).is_file():
        return root
    if best:
        files=[i for i in list(Path(root).iterdir()) if i.stem.startswith('best')]
        files.sort(key=sort_by_step, reverse=True)
        res = str(files[0])
    else:
        res = str(Path(root) / 'last.ckpt')
    return res

def load_model_path_by_config(args):
    return load_model_path(root=args.load_dir, version=args.load_ver, v_num=args.load_v_num)