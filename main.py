
import argparse
from pathlib import Path
import os
import pdb
import glob
import wandb
import logging
import torch
from torch.utils.data import DataLoader
#from models import build_model

import pytorch_lightning as pl
from engine import local_trainer, Evaluator
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning import seed_everything
# from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

from datasets import load_dataset
from src.data import collate_fn
from src.utils import get_rank

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
    parser.add_argument('--perturbation_sampling', action='store_true')
    parser.add_argument('--sampling_type', default='rand_perturb', choices=['rand_perturb', 
                                                                      'swap_rand', 'swap_first_click_bot', 
                                                                      'swap_first_click_top', 'swap_first_click_rand'])

    parser.add_argument('--ultr_models', default=None, 
                        choices=['ips','twotower'])
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--save_epochs', default=2, type=int)
    parser.add_argument('--delta_retain', default=0.5, type=float)
    parser.add_argument('--soft_labels', action='store_true')
    
    parser.add_argument('--soft_base', default=0.9, type=float)
    parser.add_argument('--soft_gain', default=0.02, type=float)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--n_gpus', default=4, type=int,
                        help="Number of GPUs available")
    
    parser.add_argument('--problem_type', default='classification', type=str)
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
    parser.add_argument('--n_viz', default=5, type=int)
    parser.add_argument('--resume', default=0, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--eval_ultr', action='store_true')
    parser.add_argument('--ste', action='store_true')
    parser.add_argument('--concat_feats', action='store_true')
    parser.add_argument('--pretrain_ranker', action='store_true')
    parser.add_argument('--merge_imgs', action='store_true')
    parser.add_argument('--train_ranker', action='store_true')
    parser.add_argument('--train_ranker_lambda', action='store_true')
    parser.add_argument('--eval_rels', action='store_true')
    parser.add_argument('--force_tnse', action='store_true')
    parser.add_argument('--use_dcg', action='store_true')
    parser.add_argument('--save_cls', action='store_true')
    parser.add_argument('--cls_reg', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--use_rax', action='store_true')
    parser.add_argument('--use_doc_feat', action='store_true')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--log_file', default=None, type=str)
    parser.add_argument('--wandb_project_name', default='ranking', type=str)

    return parser

def main(args):

    # fix the seed for reproducibility
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
    
    #print('set up processor ...')

    checkpoint_callback = ModelCheckpoint(dirpath=args.output_dir, filename='{epoch}')
    #logger = TensorBoardLogger(save_dir=args.output_dir, version=1, name="lightning_logs")
    logger = CSVLogger(save_dir=args.output_dir, name="lightning_logs")

    if False:
        train_dataset = load_dataset(args.repo_name,name="clicks",
                                split="train", # ["train", "test"]
                                cache_dir="~/.cache/huggingface",
                                )
        
        if args.eval_rels:
            test_dataset = load_dataset(
                        args.repo_name,
                        name="annotations",
                        split="test",
                        cache_dir="~/.cache/huggingface",
                    )
        else:
            test_dataset = load_dataset(args.repo_name,name="clicks",
                                split="test", # ["train", "test"]
                                cache_dir="~/.cache/huggingface",
                                )

    if True:
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
                    gradient_clip_val=0.1, accumulate_grad_batches=1, \
                    check_val_every_n_epoch=args.eval_epochs, callbacks=[checkpoint_callback],
                    log_every_n_steps=args.print_freq, logger=logger, num_sanity_val_steps=0,
                    strategy=DDPStrategy(find_unused_parameters=True),
                    limit_test_batches=args.limit_test_batches, limit_train_batches=args.limit_train_batches,
                    limit_val_batches=args.limit_val_batches,
                    )
    
    #sampler = torch.utils.data.DistributedSampler(train_dataset, shuffle=False)

    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=args.batch_size,
                                  num_workers=args.num_workers, pin_memory=True, drop_last=True, shuffle=False)
        
    test_dataloader = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=args.batch_size,
                                 num_workers=args.num_workers, drop_last=True)

    trainer = local_trainer(train_loader=train_dataloader,val_loader=test_dataloader,
                            test_dataset=test_dataset,args=args)
        
    if args.eval:
        print('\n\n Evaluating ... ', args.save_fname, '\n')
        print('\n\n Evaluating ... ', args.save_fname, file=args.log_file)
        if args.train_ranker_lambda:
            trainer.resume(load_path=args.load_path, model='arranger')
        else:
            trainer.resume(load_path=args.load_path, model='reward')
        pyl_trainer.validate(trainer,test_dataloader)
    else:
        pyl_trainer.fit(trainer, train_dataloader, test_dataloader)

    #############################################################################################################
    args.log_file.close()

    if not args.eval:
        trainer.evaluator.plot_train_val(log_file_path=args.output_dir+'/out.log',output_image_path=args.output_dir+'/eval_train_val.jpg')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Rank BERT', parents=[get_args_parser()])
    args = parser.parse_args()

    if args.debug:
        args.output_folder = 'demo'
        args.num_workers = 0
        args.n_gpus = 1
        args.batch_size = 3
        args.frint_freq = 10
        args.limit_train_batches=4
        args.limit_val_batches=4
        args.limit_test_batches=4
        #args.n_viz = 500
        
    else:
        args.limit_train_batches=None
        args.limit_val_batches=None
        args.limit_test_batches=None

    args.output_dir = args.output_path+args.output_folder

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    logging.basicConfig(filename=args.output_dir+'/error.log', level=logging.ERROR)
    
    try:
        main(args)
    except Exception as e:
        print('\n Aborting.. Error saved in ',args.output_dir+'/error.log \n')
        logging.exception("An error occurred:")