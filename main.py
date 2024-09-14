
import argparse
from pathlib import Path
import os
import pdb
import wandb
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
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--batch_size', default=512, type=int)
    #parser.add_argument('--n_classes', default=80, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=51, type=int)
    parser.add_argument('--eval_epochs', default=1, type=int)
    parser.add_argument('--use_model_preds', default=1, type=int)
    parser.add_argument('--print_freq', default=500, type=int)
    parser.add_argument('--max_positions_PE', default=50, type=int)
    parser.add_argument('--repo_name', default="philipphager/baidu-ultr_baidu-mlm-ctr", choices=['philipphager/baidu-ultr_baidu-mlm-ctr',
                                                                                     'philipphager/baidu-ultr_uva-mlm-ctr'])
   
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--save_epochs', default=2, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--n_gpus', default=4, type=int,
                        help="Number of GPUs available")
    
    parser.add_argument('--problem_type', default='classification', type=str)

    # dataset parameters
    parser.add_argument('--output_path', default='/home/ggbhatt/workspace/cf_ranking/outputs/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--output_folder', default='demo',
                        help='path where to save, empty for no saving')
    parser.add_argument('--load_path', default='',
                        help='path where to load model')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default=0, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--save_cls', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--use_doc_feat', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
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

    if current_rank>0 or args.debug or args.eval:
        print ('\n shutting wandb for multiple GPUs. Will only run for rank:0 process. \n')
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

    train_dataset = load_dataset(args.repo_name,name="clicks",
                            split="train", # ["train", "test"]
                            cache_dir="~/.cache/huggingface",
                            )
    
    test_dataset = load_dataset(args.repo_name,name="clicks",
                            split="test", # ["train", "test"]
                            cache_dir="~/.cache/huggingface",
                            )
    
    pyl_trainer = pl.Trainer(devices=list(range(args.n_gpus)), accelerator="gpu", max_epochs=args.epochs, 
                    gradient_clip_val=0.1, accumulate_grad_batches=1, \
                    check_val_every_n_epoch=args.eval_epochs, callbacks=[checkpoint_callback],
                    log_every_n_steps=args.print_freq, logger=logger, num_sanity_val_steps=0,
                    strategy=DDPStrategy(find_unused_parameters=True),
                    limit_test_batches=args.limit_test_batches, limit_train_batches=args.limit_train_batches,
                    limit_val_batches=args.limit_val_batches,
                    )

    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=args.batch_size,
                                  num_workers=args.num_workers, pin_memory=True)
        
    test_dataloader = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=args.batch_size,
                                 num_workers=args.num_workers)

    trainer = local_trainer(train_loader=train_dataloader,val_loader=test_dataloader,
                            test_dataset=test_dataset,args=args)
        
    if args.eval:
        print('\n\n Evaluating ... \n\n')
        trainer.resume(load_path=args.load_path)
        pyl_trainer.validate(trainer,test_dataloader)
    else:
        pyl_trainer.fit(trainer, train_dataloader, test_dataloader)

    #############################################################################################################
    args.log_file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Rank BERT', parents=[get_args_parser()])
    args = parser.parse_args()

    if args.debug:
        args.output_folder = 'demo'
        args.num_workers = 0
        args.n_gpus = 1
        args.batch_size = 5
        args.limit_train_batches=15
        args.limit_val_batches=15
        args.limit_test_batches=15
    else:
        args.limit_train_batches=None
        args.limit_val_batches=None
        args.limit_test_batches=None

    args.output_dir = args.output_path+args.output_folder

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)