import os
import argparse
from pathlib import Path

def get_args():
    parser = argparse.ArgumentParser()
    # different from PointMamba (ours)
    parser.add_argument('--w_CE', type=float, default=1.0, help='weight of the CE loss')
    parser.add_argument('--w_GLR', type=float, default=1.0, help='weight of the GLR loss')
    parser.add_argument('--w_importance', type=float, default=1.0, help='weight of the importance loss')
    parser.add_argument('--type_pooling', type=str, default='ave', help='the pooling operation before the classifier, ave or max or important (ours)')
    parser.add_argument('--type_weighting', type=str, default='direct', help='the pooling operation before the classifier, direct (score * f) or drop_neg (pos_score * f) or sfm (sfm(score) * f)')
    parser.add_argument('--detach_mapping', action='store_true', default=False, help = 'detach the gradient before the mapping operation')
    parser.add_argument('--detach_score_prediction', action='store_true', default=False, help = 'detach the gradient before the score prediction operation')
    parser.add_argument('--mode_sort', type=str, default='max', help='max: max to min, min: min to max, both: max to min and min to max, random: random, no: initial order, triple_xyz: following PointMamba')
    parser.add_argument('--mode_group', type=str, default='SA', help='SA: PointNet++, Attention: attention in SA, EdgeConv: no downsampling, ComplexAttention: cat global with patch feature, two layer')
    parser.add_argument('--mode_patch_feature', type=str, default='cat_sort', help='cat_sort: cat both feature, else: direct pooling')
    parser.add_argument('--mode_encoder', type=str, default='Mamba', help='Mamba or Transformer')

    parser.add_argument('--use_importance_order', action='store_true', default=False, help = 'use importance order')
    parser.add_argument('--use_xyz_order', action='store_true', default=False, help = 'use xyz order')
    parser.add_argument('--use_map_order', action='store_true', default=False, help = 'use map order')

    parser.add_argument('--use_simple_score_predictor', action='store_true', default=False, help = 'incase the score predictor is too heavy')

    # attention settings
    parser.add_argument('--attention_use_cls_token', action='store_true', default=False, help = 'use attention cls token')
    parser.add_argument('--attention_depth', type=int, default=4, help='attention depth')
    parser.add_argument('--attention_drop_path_rate', type=float, default=0.1, help='attention drop_path_rate')
    parser.add_argument('--attention_num_heads', type=int, default=6, help='attention num_heads')

    parser.add_argument('--Transformer_encoder_num_heads', type=int, default=6, help='attention num_heads in Transformer encoder')

    # the second SA layer (for PointMambaFormer++)
    parser.add_argument('--SA_attention_use_cls_token', action='store_true', default=False, help = 'use attention cls token of the Transformer SA layer')

    parser.add_argument('--use_logits_sfm', action='store_true', default=False, help = 'use sfm when calculating the mean of different logits of different orders')
    parser.add_argument('--use_vote', action='store_true', default=False, help = 'use vote to test')

    parser.add_argument(
        '--config', 
        type = str, 
        help = 'yaml config file')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch'],
        default='none',
        help='job launcher')     
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=8)
    # seed 
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')      
    # bn
    parser.add_argument(
        '--sync_bn', 
        action='store_true', 
        default=False, 
        help='whether to use sync bn')
    # some args
    parser.add_argument('--exp_name', type = str, default='default', help = 'experiment name')
    parser.add_argument('--loss', type=str, default='cd1', help='loss name')
    parser.add_argument('--start_ckpts', type = str, default=None, help = 'reload used ckpt path')
    parser.add_argument('--ckpts', type = str, default=None, help = 'test used ckpt path')
    parser.add_argument('--val_freq', type = int, default=1, help = 'test freq')
    parser.add_argument(
        '--vote',
        action='store_true',
        default=False,
        help = 'vote acc')
    parser.add_argument(
        '--resume', 
        action='store_true', 
        default=False, 
        help = 'autoresume training (interrupted by accident)')
    parser.add_argument(
        '--test', 
        action='store_true', 
        default=False, 
        help = 'test mode for certain ckpt')
    parser.add_argument(
        '--finetune_model', 
        action='store_true', 
        default=False, 
        help = 'finetune modelnet with pretrained weight')
    parser.add_argument(
        '--scratch_model', 
        action='store_true', 
        default=False, 
        help = 'training modelnet from scratch')
    parser.add_argument(
        '--mode', 
        choices=['easy', 'median', 'hard', None],
        default=None,
        help = 'difficulty mode for shapenet')        
    parser.add_argument(
        '--way', type=int, default=-1)
    parser.add_argument(
        '--shot', type=int, default=-1)
    parser.add_argument(
        '--fold', type=int, default=-1)
    
    args = parser.parse_args()

    if args.test and args.resume:
        raise ValueError(
            '--test and --resume cannot be both activate')

    if args.resume and args.start_ckpts is not None:
        raise ValueError(
            '--resume and --start_ckpts cannot be both activate')

    if args.test and args.ckpts is None:
        raise ValueError(
            'ckpts shouldnt be None while test mode')

    if args.finetune_model and args.ckpts is None:
        print(
            'training from scratch')

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.test:
        args.exp_name = 'test_' + args.exp_name
    if args.mode is not None:
        args.exp_name = args.exp_name + '_' +args.mode
    args.experiment_path = os.path.join('./experiments', Path(args.config).stem, Path(args.config).parent.stem, args.exp_name)
    args.tfboard_path = os.path.join('./experiments', Path(args.config).stem, Path(args.config).parent.stem,'TFBoard' ,args.exp_name)
    args.log_name = Path(args.config).stem
    create_experiment_dir(args)
    return args

def create_experiment_dir(args):
    if not os.path.exists(args.experiment_path):
        os.makedirs(args.experiment_path)
        print('Create experiment path successfully at %s' % args.experiment_path)
    if not os.path.exists(args.tfboard_path):
        os.makedirs(args.tfboard_path)
        print('Create TFBoard path successfully at %s' % args.tfboard_path)

