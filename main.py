from tools import pretrain_run_net as pretrain
from tools import finetune_run_net as finetune
from tools import run_net_test_vote as test_net
from utils import parser, dist_utils, misc
from utils.logger import *
from utils.config import *
import time
import os
import torch
import pdb
from tensorboardX import SummaryWriter

def main():
    # args
    args = parser.get_args()

    # CUDA
    args.use_gpu = torch.cuda.is_available()
    if args.use_gpu:
        torch.backends.cudnn.benchmark = True
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        args.distributed = False
    else:
        args.distributed = True
        dist_utils.init_dist(args.launcher)
        # re-set gpu_ids with distributed training mode
        _, world_size = dist_utils.get_dist_info()
        args.world_size = world_size
    # logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(args.experiment_path, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, name=args.log_name)
    # define the tensorboard writer
    if not args.test:
        if args.local_rank == 0:
            train_writer = SummaryWriter(os.path.join(args.tfboard_path, 'train'))
            val_writer = SummaryWriter(os.path.join(args.tfboard_path, 'test'))
        else:
            train_writer = None
            val_writer = None
    # config
    config = get_config(args, logger = logger)

    # some settings
    config.model.type_pooling = args.type_pooling
    config.model.type_weighting = args.type_weighting
    config.model.detach_mapping = args.detach_mapping
    config.model.detach_score_prediction = args.detach_score_prediction
    config.model.mode_sort = args.mode_sort
    config.model.mode_group = args.mode_group
    config.model.mode_encoder = args.mode_encoder
    config.model.Transformer_encoder_num_heads = args.Transformer_encoder_num_heads

    config.model.use_importance_order = args.use_importance_order
    config.model.use_xyz_order = args.use_xyz_order
    config.model.use_map_order = args.use_map_order

    config.model.use_simple_score_predictor = args.use_simple_score_predictor

    config.model.attention_use_cls_token = args.attention_use_cls_token
    config.model.attention_depth = args.attention_depth
    config.model.attention_drop_path_rate = args.attention_drop_path_rate
    config.model.attention_num_heads = args.attention_num_heads
    config.model.mode_patch_feature = args.mode_patch_feature

    config.model.mode_patch_feature = args.mode_patch_feature

    config.model.use_logits_sfm = args.use_logits_sfm

    # batch size
    if args.distributed:
        assert config.total_bs % world_size == 0
        config.dataset.train.others.bs = config.total_bs // world_size
        if config.dataset.get('extra_train'):
            config.dataset.extra_train.others.bs = config.total_bs // world_size * 2
        config.dataset.val.others.bs = config.total_bs // world_size * 2
        if config.dataset.get('test'):
            config.dataset.test.others.bs = config.total_bs // world_size 
    else:
        config.dataset.train.others.bs = config.total_bs
        if config.dataset.get('extra_train'):
            config.dataset.extra_train.others.bs = config.total_bs * 2
        config.dataset.val.others.bs = config.total_bs * 2
        if config.dataset.get('test'):
            config.dataset.test.others.bs = config.total_bs 
    # log 
    log_args_to_file(args, 'args', logger = logger)
    log_config_to_file(config, 'config', logger = logger)
    # exit()
    logger.info(f'Distributed training: {args.distributed}')
    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        misc.set_random_seed(args.seed, deterministic=args.deterministic) # seed + rank, for augmentation
    if args.distributed:
        assert args.local_rank == torch.distributed.get_rank()
        
    # run
    if args.test:
        test_net(args, config)
    else:
        if args.finetune_model or args.scratch_model:
            finetune(args, config, train_writer, val_writer)
        else:
            pretrain(args, config, train_writer, val_writer)


if __name__ == '__main__':
    main()
