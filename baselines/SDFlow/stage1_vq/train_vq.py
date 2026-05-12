import os
import sys
from pathlib import Path

SDFLOW_ROOT = Path(__file__).resolve().parents[1]
if str(SDFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(SDFLOW_ROOT))

import options.option_vq as option_vq
args = option_vq.get_args_parser()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import json

import torch
import torch.optim as optim
try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    class SummaryWriter:
        def __init__(self, *args, **kwargs):
            pass

        def add_scalar(self, *args, **kwargs):
            pass

        def close(self):
            pass

import models.vqvae as vqvae

import utils.utils_model as utils_model
from dataset import dataset_VQ
import utils.eval_sdflow as eval_trans

import warnings
warnings.filterwarnings('ignore')
from torch.autograd import Variable
CUDA_LAUNCH_BLOCKING=1


def resolve_device(device_arg):
    if device_arg == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    if device_arg == 'mps' and torch.backends.mps.is_available():
        return torch.device('mps')
    if device_arg in {'cpu', 'cuda', 'mps'}:
        return torch.device('cpu')
    return torch.device(device_arg)


def update_lr_warm_up(optimizer, nb_iter, warm_up_iter, lr):

    current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    return optimizer, current_lr


def log_model_size(model, logger, name):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    msg = (
        f"{name} parameters: total={total_params:,} "
        f"({total_params / 1e6:.3f}M), trainable={trainable_params:,} "
        f"({trainable_params / 1e6:.3f}M)"
    )
    print(msg)
    logger.info(msg)


##### ---- Exp dirs ---- #####
# args = option_vq.get_args_parser()
torch.manual_seed(args.seed)

args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
os.makedirs(args.out_dir, exist_ok = True)

##### ---- Logger ---- #####
logger = utils_model.get_logger(args.out_dir)
writer = SummaryWriter(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))


logger.info(f'Training on {args.dataname}')
device = resolve_device(args.device)
logger.info(f'Using device: {device}')




##### ---- Dataloader ---- #####
train_loader = dataset_VQ.DATALoader(args.dataname,
                                        args.batch_size,
                                        window_size=args.window_size,
                                        unit_length=2**args.down_t,
                                        dataset_type='train',
                                        datasets_dir=args.datasets_dir,
                                        rel_path=args.rel_path,
                                        rel_path_train=args.rel_path_train,
                                        rel_path_valid=args.rel_path_valid,
                                        rel_path_test=args.rel_path_test,
                                        stride=args.stride,
                                        window_stride=args.window_stride,
                                        ts_stride=args.ts_stride,
                                        column=args.column)

if args.input_emb_width is None:
    args.input_emb_width = dataset_VQ.infer_num_channels(train_loader)
logger.info(f'Input channels: {args.input_emb_width}')

train_loader_iter = dataset_VQ.cycle(train_loader)


MSELoss = torch.nn.MSELoss()
##### ---- Network ---- #####
net = vqvae.VQVAE(args, ## use args to define different parameters in different quantizers
                       args.nb_code,
                       args.code_dim,
                       args.down_t,
                       args.stride_t,
                       args.width,
                       args.depth,
                       args.dilation_growth_rate,
                       args.vq_act,
                       args.vq_norm)


if args.resume_pth : 
    logger.info('loading checkpoint from {}'.format(args.resume_pth))
    ckpt = torch.load(args.resume_pth, map_location='cpu')
    net.load_state_dict(ckpt['net'], strict=True)
net.train()
net.to(device)
log_model_size(net, logger, 'VQ-VAE')

##### ---- Optimizer & Scheduler ---- #####
optimizer = optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma)
  


##### ------ warm-up ------- #####
avg_recons, avg_perplexity, avg_commit = 0., 0., 0.

for nb_iter in range(1, args.warm_up_iter):

    optimizer, current_lr = update_lr_warm_up(optimizer, nb_iter, args.warm_up_iter, args.lr)

    input = next(train_loader_iter)
    input = input.to(device).float()
    output, loss_commit, perplexity = net(input)
    loss_recons = MSELoss(output, input)
    loss = args.recon*loss_recons + args.commit * loss_commit

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    avg_recons += loss_recons.item()
    avg_perplexity += perplexity.item()
    avg_commit += loss_commit.item()

    if nb_iter % args.print_iter ==  0 :
        avg_recons /= args.print_iter
        avg_perplexity /= args.print_iter
        avg_commit /= args.print_iter

        logger.info(f"Warmup. Iter {nb_iter} :  lr {current_lr:.5f} \t Commit. {avg_commit:.5f} \t PPL. {avg_perplexity:.2f} \t Recons.  {avg_recons:.8f}")

        avg_recons, avg_perplexity, avg_commit = 0., 0., 0.

####---- Training ---- #####
avg_recons, avg_perplexity, avg_commit = 0., 0., 0.
if args.skip_eval:
    best_iter, best_ds, best_mse = 0, 99999, 99999
else:
    best_iter, best_ds, best_mse, writer, logger = eval_trans.evaluation_vqvae(args,args.out_dir, train_loader, net, logger, writer, 0, best_iter=0, best_ds=99999, best_mse = 99999, device=device)

for nb_iter in range(1, args.total_iter + 1):
    
    input = next(train_loader_iter)
    input = input.to(device).float() # bs, nb_joints, joints_dim, seq_len
    
    output, loss_commit, perplexity = net(input)

    loss_recons = MSELoss(output, input)
    loss = args.recon*loss_recons + args.commit * loss_commit


    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    
    avg_recons += loss_recons.item()
    avg_perplexity += perplexity.item()
    avg_commit += loss_commit.item()
    
    if nb_iter % args.print_iter == 0:
        avg_recons /= args.print_iter
        avg_perplexity /= args.print_iter
        avg_commit /= args.print_iter
        
        writer.add_scalar('./Train/Recons', avg_recons, nb_iter)
        writer.add_scalar('./Train/PPL', avg_perplexity, nb_iter)
        writer.add_scalar('./Train/Commit', avg_commit, nb_iter)
        
        logger.info(f"Train. Iter {nb_iter} : \t Commit. {avg_commit:.8f} \t PPL. {avg_perplexity:.2f} \t Recons.  {avg_recons:.8f}")
        
        avg_recons, avg_perplexity, avg_commit = 0., 0., 0.,

    if not args.skip_eval and nb_iter % args.eval_iter==0:
        best_iter, best_ds, best_mse, writer, logger = eval_trans.evaluation_vqvae(args,args.out_dir, train_loader, net, logger, writer, nb_iter, best_iter, best_ds, best_mse, device=device)
if args.skip_eval:
    torch.save({'net' : net.state_dict(), 'args': vars(args)}, os.path.join(args.out_dir, 'net_best_ds.pth'))
    torch.save({'net' : net.state_dict(), 'args': vars(args)}, os.path.join(args.out_dir, 'net_best_mse.pth'))
msg_final = f"Train. Iter {best_iter} : , MAE. {best_ds:.4f}, MSE. {best_mse:.6f}"
logger.info(msg_final)
