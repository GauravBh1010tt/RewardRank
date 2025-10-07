
import argparse
from pathlib import Path
import os
import pdb
import glob
import json
import wandb
import logging
import torch
import pandas as pd
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from engine import local_trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

from datasets import load_dataset, load_from_disk, concatenate_datasets
from src.data import collate_fn as cf, collate_fn_llm
from src.utils import get_rank
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

torch.autograd.set_detect_anomaly(True)

def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    parser.add_argument('--lr', default=2e-5, type=float)
    parser.add_argument('--cls_reg_lr', default=0.5, type=float)
    parser.add_argument('--batch_size', default=256, type=int)
    #parser.add_argument('--n_classes', default=80, type=int)
    parser.add_argument('--weight_decay', default=1e-2, type=float)
    parser.add_argument('--epochs', default=51, type=int)
    parser.add_argument('--eval_epochs', default=1, type=int)
    parser.add_argument('--use_model_preds', default=1, type=int)
    parser.add_argument('--print_freq', default=500, type=int)
    parser.add_argument('--max_positions_PE', default=50, type=int)
    parser.add_argument('--max_items_QG', default=21, type=int)
    parser.add_argument('--repo_name', default="philipphager/baidu-ultr_uva-mlm-ctr", choices=['philipphager/baidu-ultr_baidu-mlm-ctr',
                                                                                     'philipphager/baidu-ultr_uva-mlm-ctr'])

    parser.add_argument('--sampling_type', default='rand_perturb', choices=['rand_perturb', 
                                                                      'swap_rand', 'swap_first_click_bot', 
                                                                      'swap_first_click_top', 'swap_first_click_rand'])
    parser.add_argument('--rank_loss', default='pirank', choices=['pirank', 'ips_point',
                                                                      'list_mle', 'list_net', 
                                                                      'ips_list', 'lambdarank'])
    
    parser.add_argument('--gain_fn', default='lin', choices=['lin','exp'])


    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--save_epochs', default=2, type=int)

    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--n_gpus', default=4, type=int,
                        help="Number of GPUs available")
    
    parser.add_argument('--problem_type', default='classification', 
                        choices=['classification','regression'])
    parser.add_argument('--save_fname', default=None, type=str)

    # dataset parameters
    parser.add_argument('--output_path', default='/home/ec2-user/workspace/cf_ranking/outputs/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--data_path', default='/home/ec2-user/workspace/data/custom_click/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--output_folder', default='demo',
                        help='path where to save, empty for no saving')
    parser.add_argument('--load_path', default='',
                        help='path where to load model')
    parser.add_argument('--load_path_reward', default='',
                        help='path where to load model')
    parser.add_argument('--load_path_ranker', default='',
                        help='path where to load model')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default=0, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--use_org_feats', action='store_true')
    parser.add_argument('--reward_sanity', action='store_true')
    parser.add_argument('--eval_llm', action='store_true')
    parser.add_argument('--ste', action='store_true')
    parser.add_argument('--concat_feats', action='store_true')
    parser.add_argument('--pretrain_ranker', action='store_true')
    parser.add_argument('--train_ranker', action='store_true')
    parser.add_argument('--train_ranker_naive', action='store_true')
    parser.add_argument('--use_dcg', action='store_true')
    parser.add_argument('--save_cls', action='store_true')
    parser.add_argument('--save_soft_labels', action='store_true')
    parser.add_argument('--cls_reg', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--use_doc_feat', action='store_true')
    parser.add_argument('--per_item_feats', action='store_true')
    parser.add_argument('--urcc_loss', action='store_true')
    parser.add_argument('--pgrank_loss', action='store_true')
    parser.add_argument('--grpo_loss', action='store_true')
    parser.add_argument('--lau_eval', action='store_true')
    parser.add_argument('--po_eval', action='store_true')
    parser.add_argument('--pgrank_disc', action='store_true')
    parser.add_argument('--pgrank_nobaseline', action='store_true')
    parser.add_argument('--ips_production', action='store_true')
    parser.add_argument('--ips_ideal', action='store_true')
    parser.add_argument('--lin_pos', action='store_true')
    parser.add_argument('--reward_correction', action='store_true')
    parser.add_argument('--reward_plus_proxy', action='store_true')
    parser.add_argument('--ips_sampling', action='store_true')
    parser.add_argument('--reward_loss_cls', action='store_true')
    parser.add_argument('--reward_loss_reg', default=1.0, type=float)
    parser.add_argument('--reward_loss_reg_peritem', default=1.0, type=float)
    parser.add_argument('--residual_coef', default=0.5, type=float)
    parser.add_argument('--grpo_beta', default=0.04, type=float)
    parser.add_argument('--grpo_eps', default=0.2, type=float)
    parser.add_argument('--soft_perm_loss_reg', default=1.0, type=float)
    parser.add_argument('--soft_sort_temp', default=1.0, type=float)

    parser.add_argument('--MC_samples', default=4, type=int, help='number of samples to be used in Monte Carlo sampling. This is used by pgrank_loss')
    
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--grpo_rollouts', default=8, type=int)
    parser.add_argument('--log_file', default=None, type=str)
    parser.add_argument('--wandb_project_name', default='ranking', type=str)

    return parser

def main(args):

    seed = args.seed
    seed_everything(seed, workers=True)

    if args.eval:
        args.log_file = open(args.output_dir+'/out_eval.log','a')
    else:
        args.log_file = open(args.output_dir+'/out.log','a')

    print('Logging: args ', args, file=args.log_file)

    current_rank = get_rank()

    if current_rank>0 or args.debug or args.eval or not args.use_wandb:
        #print ('\n shutting wandb for multiple GPUs. Will only run for rank:0 process. \n')
        os.environ["WANDB_MODE"] = "offline"

    if args.use_wandb:
        wandb.init(
            project=args.wandb_project_name,
            entity=None,
            sync_tensorboard=False,
            config=args,
            name=args.output_folder,
            save_code=True,
        )
    
    checkpoint_callback = ModelCheckpoint(dirpath=args.output_dir, filename='{epoch}')
    logger = CSVLogger(save_dir=args.output_dir, name="lightning_logs")
    
    if args.lau_eval:
        train_files = glob.glob(os.path.join(args.data_path, '*'))
        datasets = [load_from_disk(d) for d in train_files]
        dataset = concatenate_datasets(datasets)

        if args.train_ranker_naive:
            dataset = dataset.filter(lambda example: example['item_position'] != -1 and example['query_id'] != -1)
        else:
            dataset = dataset.filter(lambda example: example['query_id'] != -1)

        with open(args.data_path.replace('processed', 'split_indices.json'), "r") as f:
            data_ids = json.load(f)

        train_dataset = dataset.filter(lambda example: example['query_id'] in data_ids['train'])
        test_dataset = dataset.filter(lambda example: example['query_id'] in data_ids['test'])
        collate_fn = collate_fn_llm
        

        if args.save_soft_labels:

            pur = [i for i in test_dataset['purchase_prob']]
            sns.histplot(pd.DataFrame(pur), bins=15, kde=True, stat='probability')
            plt.legend([], [], frameon=False)
            plt.xlabel(r'$\hat{y} = P(pur)$')
            plt.ylabel('Relative Frequency')

            # Save the plot before showing it
            plt.savefig(f"{args.output_dir}/histogram_purchase_probs.png", dpi=300, bbox_inches='tight')

            pos = [i+1 for i in test_dataset['item_position']]
            item_positions_series = pd.Series(pos)

            # Calculate the relative frequency of each unique item
            frequency = item_positions_series.value_counts(normalize=True)  # normalize=True gives relative frequencies

            # Convert the result into a DataFrame
            df = frequency.reset_index()
            df.columns = ['Item Position', 'Relative Frequency']

            # Plot the bar plot
            sns.barplot(x='Item Position', y='Relative Frequency', data=df)

            # Add title and labels
            plt.xlabel('Item Position')
            plt.ylabel('Relative Frequency')
            plt.savefig(f"{args.output_dir}/histogram_pos.png", dpi=300, bbox_inches='tight')

    elif args.use_org_feats:
        collate_fn = cf
        print ('using org data ..')
        train_dataset = load_dataset(args.repo_name,name="clicks",
                                split="train", # ["train", "test"]
                                cache_dir="/",
                                )
        
        test_dataset = load_dataset(
                    args.repo_name,
                    name="annotations",
                    split="test",
                    cache_dir="/",
                )
    else:
        collate_fn = cf
        train_files = glob.glob(os.path.join(os.path.join(args.data_path, 'train'), '**/*.arrow'), recursive=True)
        train_dataset = load_dataset(path=os.path.join(args.data_path, 'train'),
                                    data_files=train_files, split='train')
        
        if args.eval_rels:
            test_files = glob.glob(os.path.join(os.path.join(args.data_path, 'ann'), '**/*.arrow'), recursive=True)
        else:
            test_files = glob.glob(os.path.join(os.path.join(args.data_path, 'test'), '**/*.arrow'), recursive=True)
        
        
        test_dataset = load_dataset(path=os.path.join(args.data_path, 'test'),
                                    data_files=test_files, split='train')
    
    pyl_trainer = pl.Trainer(devices=list(range(args.n_gpus)), accelerator="gpu", max_epochs=args.epochs, 
                    gradient_clip_val=0.1, 
                    #accumulate_grad_batches=max(1, int(1024/(args.batch_size*args.n_gpus))),
                    check_val_every_n_epoch=args.eval_epochs, callbacks=[checkpoint_callback],
                    log_every_n_steps=args.print_freq, logger=logger, num_sanity_val_steps=0,
                    strategy=DDPStrategy(find_unused_parameters=True),
                    limit_test_batches=args.limit_test_batches, limit_train_batches=args.limit_train_batches,
                    limit_val_batches=args.limit_val_batches,
                    )
    
    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=args.batch_size,
                                  num_workers=args.num_workers, pin_memory=True, drop_last=True, shuffle=False)
        
    test_dataloader = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=args.batch_size,
                                 num_workers=args.num_workers, drop_last=True)

    trainer = local_trainer(train_loader=train_dataloader,val_loader=test_dataloader,
                            test_dataset=test_dataset,args=args)
    
    if args.eval:
        print('\n\n Evaluating ... ', args.save_fname, '\n')
        print('\n\n Evaluating ... ', args.save_fname, file=args.log_file)
        
        if args.train_ranker_naive or args.train_ranker:
            trainer.resume(load_path=args.load_path, model='arranger')
        if args.load_path_reward:
            trainer.resume(load_path=args.load_path_reward, model='reward')
            
        pyl_trainer.validate(trainer,test_dataloader)
    else:
        pyl_trainer.fit(trainer, train_dataloader, test_dataloader)

    args.log_file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Rank BERT', parents=[get_args_parser()])
    args = parser.parse_args()

    if args.debug:
        args.output_folder = 'demo'
        args.num_workers = 0
        args.n_gpus = 1
        args.frint_freq = 10
        args.limit_train_batches=100
        args.limit_val_batches=4
        args.limit_test_batches=4
    else:
        args.limit_train_batches=None
        args.limit_val_batches=None
        args.limit_test_batches=None

    args.output_dir = os.path.join(args.output_path,args.output_folder)

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    logging.basicConfig(filename=os.path.join(args.output_dir,'error.log'), level=logging.ERROR)
    
    try:
        main(args)
    except Exception as e:
        print('\n Aborting.. Error saved in ',args.output_dir+'/error.log \n')
        logging.exception("An error occurred:")