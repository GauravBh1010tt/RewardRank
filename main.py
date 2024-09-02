
import argparse
from pathlib import Path
import os
import pdb
from torch.utils.data import DataLoader
#from models import build_model

import pytorch_lightning as pl
from engine import local_trainer, Evaluator
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning import seed_everything
# from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from datasets import load_dataset
from src.data import collate_fn

def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--batch_size', default=6, type=int)
    parser.add_argument('--n_classes', default=80, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=51, type=int)
    parser.add_argument('--eval_epochs', default=2, type=int)
    parser.add_argument('--print_freq', default=500, type=int)
    parser.add_argument('--repo_name', default="SenseTime/deformable-detr", type=str)
   
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--save_epochs', default=10, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--n_gpus', default=1, type=int,
                        help="Number of GPUs available")

    # dataset parameters
    parser.add_argument('--output_dir', default='/home/ggbhatt/workspace/cf_ranking/outputs/demo',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default=0, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--eval_every', default=1, type=int)
    parser.add_argument('--num_workers', default=2, type=int)

    return parser


def main(args):

    # fix the seed for reproducibility
    seed = args.seed
    seed_everything(seed, workers=True)
    
    #print('set up processor ...')

    checkpoint_callback = ModelCheckpoint(dirpath=args.output_dir, filename='{epoch}')
    #logger = TensorBoardLogger(save_dir=args.output_dir, version=1, name="lightning_logs")
    logger = CSVLogger(save_dir=args.output_dir, name="lightning_logs")

    train_dataset = load_dataset("philipphager/baidu-ultr_baidu-mlm-ctr",name="clicks",
                            split="train", # ["train", "test"]
                            cache_dir="~/.cache/huggingface",
                            )
    
    test_dataset = load_dataset("philipphager/baidu-ultr_baidu-mlm-ctr",name="clicks",
                            split="test", # ["train", "test"]
                            cache_dir="~/.cache/huggingface",
                            )
    
    pyl_trainer = pl.Trainer(devices=list(range(args.n_gpus)), accelerator="gpu", max_epochs=args.epochs, 
                    gradient_clip_val=0.1, accumulate_grad_batches=max(1,int(32/(args.n_gpus*args.batch_size))), \
                    check_val_every_n_epoch=args.eval_epochs, callbacks=[checkpoint_callback],
                    log_every_n_steps=args.print_freq, logger=logger, num_sanity_val_steps=0)

    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=args.batch_size,
                                  num_workers=args.num_workers, pin_memory=True)
        
    test_dataloader = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=args.batch_size,
                                 num_workers=args.num_workers)

    trainer = local_trainer(train_loader=train_dataloader,val_loader=test_dataloader,
                            test_dataset=test_dataset,args=args)
        
    pyl_trainer.fit(trainer, train_dataloader, test_dataloader)

    #############################################################################################################
    # args.log_file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    out_dir = args.output_dir

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)