import os, sys
import torch
import numpy as np
import torch.multiprocessing
import logging
from metrics import evaluate_model_uncond
from utils.loggers import NeptuneLogger, PrintLogger, CompositeLogger
from utils.utils import create_model_name_and_dir, print_model_params, log_config_and_tags
from data_provider.data_provider import data_provider
from utils.utils_args import parse_args_uncond

from distributed.distributed import is_main_process, Disributed
import torch.distributed as dist

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.multiprocessing.set_sharing_strategy('file_system')
from data_provider.combined_datasets import dataset_list
from importlib import import_module


def _extract_real_tensor(dataset):
    if hasattr(dataset, "primary_tensor"):
        return dataset.primary_tensor
    if hasattr(dataset, "tensors"):
        return dataset.tensors[0]
    if isinstance(dataset, (tuple, list)):
        return dataset[0]
    return dataset



def _save_eval_samples(args, dataset_name, epoch, real_set, generated_set):
    samples_dir = os.path.join(os.path.dirname(args.log_dir), "eval_samples")
    os.makedirs(samples_dir, exist_ok=True)
    sample_path = os.path.join(
        samples_dir,
        f"{dataset_name}_epoch_{epoch:04d}.pt",
    )
    payload = {
        "dataset": dataset_name,
        "epoch": int(epoch),
        "real_ts": real_set.detach().cpu().float(),
        "sampled_ts": generated_set.detach().cpu().float(),
    }
    torch.save(payload, sample_path)
    print("Saved samples to {}".format(sample_path))


def main(args):
    # Set up basic attributes
    args.finetune = not args.pretrain
    args.train_on_datasets = [dataset for dataset in dataset_list if dataset in args.train_on_datasets]
    
    # Model name and directory
    name = create_model_name_and_dir(args)

    # set-up neptune logger. switch to your desired logger
    with CompositeLogger([NeptuneLogger(), PrintLogger()]) if args.neptune and is_main_process() \
            else CompositeLogger([PrintLogger()]) as logger:
        
        if args.finetune:
            args.tags.append('finetune')
        else:
            args.tags.append('pretrain')
        
        # log config and tags
        log_config_and_tags(args, logger, name, len(args.train_on_datasets) > 1)

        # Setup Data
        dataset_loader, samplers, trainsets, metadatas = data_provider(args)
        args.n_classes = dataset_loader.num_datasets
        if len(args.datasets) > 1:
            logging.info(f'all datasets are ready - Total number of sequences: {sum([len(trainset) for keya, trainset in trainsets.items()])}')
        else:
            logging.info(args.datasets[0]['name'] + ' dataset is ready.')

        # Setup handler
        handler = import_module(args.handler).Handler(args=args, rank=args.device)

        # print model parameters
        print_model_params(logger, handler.model)

        # --- train model ---
        best_score = float('inf')  # marginal score for long-range metrics, dice score for short-range metrics
        for epoch in range(1, args.epochs):
            handler.model.train()
            handler.epoch = epoch
            logger.log_name_params('train/epoch', epoch)

            if args.ddp:
                dist.barrier()
                for key, sampler in samplers.items():
                    sampler.set_epoch(epoch)
                dist.barrier()

            # --- train loop ---
            handler.train_iter(dataset_loader, logger)

            # --- evaluation loop ---
            # if epoch % args.logging_iter == 0:
            if epoch % 1 == 0:
                if not args.no_test_model:
                    scores_mean = {'disc_mean': [], 'disc_std': [], 'pred_mean': [], 'pred_std': [], 'context_fid': []}
                    for dataset in args.train_on_datasets:
                        args.dataset = dataset
                        testset, class_label = dataset_loader.gen_dataloader(dataset)
                        handler.model.eval()
                        with torch.no_grad():
                            generated_set = handler.sample(len(testset), class_label, metadatas[dataset])

                        _save_eval_samples(
                            args,
                            dataset,
                            epoch,
                            _extract_real_tensor(testset),
                            generated_set,
                        )

                        generated_set = generated_set.cpu().detach().numpy()
                        real_set = testset.cpu().detach().numpy()
                        scores = evaluate_model_uncond(real_set, generated_set, dataset, args.device, args.eval_metrics, base_path=args.ts2vec_dir)
                        scores_mean['disc_mean'].append(scores[f'disc_mean'])
                        scores_mean['disc_std'].append(scores[f'disc_std'])
                        scores_mean['pred_mean'].append(scores[f'pred_mean'])
                        scores_mean['pred_std'].append(scores[f'pred_std'])
                        scores_mean['context_fid'].append(scores[f'context_fid'])
                        for key, value in scores.items():
                            logger.log(f'test/{dataset}_{key}', value, epoch)
                    if is_main_process():
                        logger.log(f'test/disc_mean', np.mean(scores_mean['disc_mean']), epoch)
                        logger.log(f'test/disc_std', np.mean(scores_mean['disc_std']), epoch)
                        logger.log(f'test/pred_mean', np.mean(scores_mean['pred_mean']), epoch)
                        logger.log(f'test/pred_std', np.mean(scores_mean['pred_std']), epoch)
                        logger.log(f'test/context_fid', np.mean(scores_mean['context_fid']), epoch)

                        # --- save checkpoint ---
                        disc_mean = np.mean(scores_mean['disc_mean'])
                        if disc_mean < best_score:
                            best_score = disc_mean
                            handler.save_model(args.log_dir)
                else:
                    handler.save_model(args.log_dir)

            if args.ddp:
                dist.barrier()

        logging.info(f"{'Finetune' if args.finetune else 'Pretrain'} is complete")

if __name__ == '__main__':
    args = parse_args_uncond()
    torch.random.manual_seed(args.seed)
    np.random.default_rng(args.seed)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    if args.ddp:
        Disributed(main).run(args)
    else:
        args.gpu_num = 1
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        main(args)
