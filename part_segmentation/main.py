import argparse
import os
import random
import torch.nn as nn
import torch.nn.functional as F
import torch
import datetime
import logging
import sys
import importlib
import shutil
import provider
import numpy as np
import torch.optim as optim
from timm.scheduler import CosineLRScheduler
from pathlib import Path
from tqdm import tqdm
from dataset import PartNormalDataset
import pdb

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
               'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37],
               'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49],
               'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointramba', help='model name')
    parser.add_argument('--batch_size', type=int, default=16, help='batch Size during training')
    parser.add_argument('--epoch', default=300, type=int, help='epoch to run')
    parser.add_argument('--warmup_epoch', default=10, type=int, help='warmup epoch')
    parser.add_argument('--learning_rate', default=0.0002, type=float, help='initial learning rate')
    parser.add_argument('--gpu', type=str, default='0', help='specify GPU devices')
    # parser.add_argument('--optimizer', type=str, default='AdamW', help='Adam or SGD')
    parser.add_argument('--log_dir', type=str, default='./exp', help='log path')
    # parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--npoint', type=int, default=2048, help='point Number')
    parser.add_argument('--normal', action='store_true', default=False, help='use normals')
    parser.add_argument('--config', type=str, default=None, help='config file')
    # parser.add_argument('--step_size', type=int, default=20, help='decay step for lr decay')
    # parser.add_argument('--lr_decay', type=float, default=0.5, help='decay rate for lr decay')
    parser.add_argument('--ckpts', type=str, default=None, help='ckpts')
    parser.add_argument('--root', type=str, default='/mnt/dongxu-fs2/data-ssd/wangzicheng/data2/ShapeNetPart/shapenetcore_partanno_segmentation_benchmark_v0_normal/',
                        help='data root')
    # mine
    parser.add_argument('--type_pooling', type=str, default='important', help='the pooling operation before the classifier, ave or max or important (ours)')
    parser.add_argument('--type_weighting', type=str, default='drop_neg', help='the pooling operation before the classifier, direct (score * f) or drop_neg (pos_score * f) or sfm (sfm(score) * f)')
    parser.add_argument('--detach_mapping', action='store_true', default=False, help = 'detach the gradient before the mapping operation')
    parser.add_argument('--detach_score_prediction', action='store_true', default=False, help = 'detach the gradient before the score prediction operation')
    parser.add_argument('--mode_sort', type=str, default='both', help='max: max to min, min: min to max, both: max to min and min to max, random: random, no: initial order, triple_xyz: following PointMamba')
    parser.add_argument('--mode_group', type=str, default='Attention', help='SA: PointNet++, Attention: attention in SA')
    parser.add_argument('--cat_group_token', action='store_true', default=False, help = 'cat group token in feature propagation')

    parser.add_argument('--attention_use_cls_token', action='store_true', default=False, help='use attention cls token')
    parser.add_argument('--attention_depth', type=int, default=4, help='attention depth')
    parser.add_argument('--attention_drop_path_rate', type=float, default=0.1, help='attention drop_path_rate')
    parser.add_argument('--attention_num_heads', type=int, default=6, help='attention num_heads')
    parser.add_argument('--use_simple_score_predictor', action='store_true', default=False, help = 'incase the score predictor is too heavy')
    parser.add_argument('--mode_map_feature', type=str, default='all', help='max, ave, all')

    parser.add_argument('--num_group', type=int, default=128, help='number of group')
    parser.add_argument('--group_size', type=int, default=32, help='number of neighbor')

    parser.add_argument('--use_cls', action='store_true', default=False, help='use cls or not')
    parser.add_argument('--w_seg', type=float, default=1.0, help='weight of the seg CE loss')
    parser.add_argument('--w_cls', type=float, default=1.0, help='weight of the cls CE loss')
    parser.add_argument('--w_GLR', type=float, default=1.0, help='weight of the GLR loss')
    parser.add_argument('--w_importance', type=float, default=1.0, help='weight of the importance loss')

    parser.add_argument('--seed', type=int, default=3302, help='random seed')

    return parser.parse_args()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('part_seg')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    root = args.root

    TRAIN_DATASET = PartNormalDataset(root=root, npoints=args.npoint, split='trainval', normal_channel=args.normal)
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True,
                                                  num_workers=10, drop_last=True)
    TEST_DATASET = PartNormalDataset(root=root, npoints=args.npoint, split='test', normal_channel=args.normal)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=10)
    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d" % len(TEST_DATASET))

    num_classes = 16
    num_part = 50

    '''MODEL LOADING'''
    shutil.copy('models/%s.py' % args.model, str(exp_dir))
    # shutil.copy('models/pointnet2_utils.py', str(exp_dir))
    if args.config is not None:
        from utils.config import cfg_from_yaml_file
        from utils.logger import print_log
        if args.config[:13] == "segmentation/":
            args.config = args.config[13:]
        config = cfg_from_yaml_file(args.config)

        config.detach_mapping = args.detach_mapping
        config.detach_score_prediction = args.detach_score_prediction
        config.mode_sort = args.mode_sort
        config.mode_group = args.mode_group
        config.type_pooling = args.type_pooling
        config.type_weighting = args.type_weighting

        config.attention_use_cls_token = args.attention_use_cls_token
        config.attention_depth = args.attention_depth
        config.attention_drop_path_rate = args.attention_drop_path_rate
        config.attention_num_heads = args.attention_num_heads
        config.use_simple_score_predictor = args.use_simple_score_predictor
        config.mode_map_feature = args.mode_map_feature
        config.cat_group_token = args.cat_group_token

        config.num_group = args.num_group
        config.group_size = args.group_size

        log_string(config)
        if hasattr(config, 'epoch'):
            args.epoch = config.epoch
        if hasattr(config, 'batch_size'):
            args.epoch = config.batch_size
        if hasattr(config, 'learning_rate'):
            args.learning_rate = config.learning_rate
        if hasattr(config, 'ckpt') and args.ckpts is None:
            args.ckpts = config.ckpts
        if hasattr(config, 'model'):
            MODEL = importlib.import_module(config.model) if hasattr(config, 'model') else importlib.import_module(
                args.model)
            classifier = MODEL.get_model(num_part, config).cuda()
        else:
            MODEL = importlib.import_module(args.model)
            classifier = MODEL.get_model(num_part).cuda()
    else:
        MODEL = importlib.import_module(args.model)
        shutil.copy('models/%s.py' % args.model, str(exp_dir))
        classifier = MODEL.get_model(num_part).cuda()
    criterion = MODEL.get_loss().cuda()
    criterion_GLR = MODEL.GLR_loss().cuda()
    criterion_score = nn.SmoothL1Loss(reduction='none').cuda()
    criterion_cls = nn.CrossEntropyLoss().cuda()
    classifier.apply(inplace_relu)
    print('# generator parameters:', sum(param.numel() for param in classifier.parameters()))
    start_epoch = 0

    if args.ckpts is not None:
        if args.ckpts[:13] == "segmentation/":
            args.ckpts = args.ckpts[13:]
        classifier.load_model_from_ckpt(args.ckpts)
        log_string('Load model from %s' % args.ckpts)
    else:
        log_string('No existing model, starting training from scratch...')

    ## we use adamw and cosine scheduler
    def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
        decay = []
        no_decay = []
        num_trainable_params = 0
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith(".bias") or 'token' in name or name in skip_list:
                # print(name)
                no_decay.append(param)
                num_trainable_params += param.numel()
            else:
                decay.append(param)
                num_trainable_params += param.numel()

        total_params = sum([v.numel() for v in model.parameters()])
        non_trainable_params = total_params - num_trainable_params
        log_string('########################################################################')
        log_string('>> {:25s}\t{:.2f}\tM  {:.2f}\tK'.format(
            '# TrainableParams:', num_trainable_params / (1.0 * 10 ** 6), num_trainable_params / (1.0 * 10 ** 3)))
        log_string('>> {:25s}\t{:.2f}\tM'.format('# NonTrainableParams:', non_trainable_params / (1.0 * 10 ** 6)))
        log_string('>> {:25s}\t{:.2f}\tM'.format('# TotalParams:', total_params / (1.0 * 10 ** 6)))
        log_string('>> {:25s}\t{:.2f}\t%'.format('# TuningRatio:', num_trainable_params / total_params * 100.))
        log_string('########################################################################')

        return [
            {'params': no_decay, 'weight_decay': 0.},
            {'params': decay, 'weight_decay': weight_decay}]

    param_groups = add_weight_decay(classifier, weight_decay=0.05)
    optimizer = optim.AdamW(param_groups, lr=args.learning_rate, weight_decay=0.05)

    scheduler = CosineLRScheduler(optimizer,
                                  t_initial=args.epoch,
                                  t_mul=1,
                                  lr_min=1e-6,
                                  decay_rate=0.1,
                                  warmup_lr_init=1e-6,
                                  warmup_t=args.warmup_epoch,
                                  cycle_limit=1,
                                  t_in_epochs=True)

    best_acc = 0
    global_epoch = 0
    best_class_avg_iou = 0
    best_inctance_avg_iou = 0

    classifier.zero_grad()
    for epoch in range(start_epoch, args.epoch):
        mean_correct = []

        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        '''Adjust learning rate and BN momentum'''

        classifier = classifier.train()
        loss_batch = []
        loss_batch_seg = []
        loss_batch_cls = []
        loss_batch_GLR = []
        loss_batch_importance = []
        num_iter = 0
        '''learning one epoch'''
        for i, (points, label, target) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            num_iter += 1
            points = points.data.numpy()
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
            points = points.transpose(2, 1)

            seg_pred, cls_pred, map_patch_feature, map_global_feature, cal_importance = classifier(points, to_categorical(label, num_classes))
            seg_pred = seg_pred.contiguous().view(-1, num_part)
            target = target.view(-1, 1)[:, 0]
            pred_choice = seg_pred.data.max(1)[1]

            correct = pred_choice.eq(target.data).cpu().sum()
            mean_correct.append(correct.item() / (args.batch_size * args.npoint))

            # loss
            loss_seg = criterion(seg_pred, target)
            loss_cls = criterion_cls(cls_pred, label.squeeze(-1))
            loss_GLR = criterion_GLR(map_global_feature, map_patch_feature)
            cos_sim = F.cosine_similarity(map_global_feature.unsqueeze(1).repeat(1, map_patch_feature.shape[1], 1), map_patch_feature, dim=2)
            loss_importance = criterion_score(cal_importance.squeeze(-1), cos_sim).sum(dim=-1).mean()

            if args.use_cls:
                loss = args.w_seg * loss_seg + args.w_cls * loss_cls + args.w_GLR * loss_GLR + args.w_importance * loss_importance
            else:
                loss = args.w_seg * loss_seg + args.w_GLR * loss_GLR + args.w_importance * loss_importance

            loss.backward()
            optimizer.step()
            loss_batch.append(loss.detach().cpu())
            loss_batch_seg.append(loss_seg.detach().cpu())
            loss_batch_cls.append(loss_cls.detach().cpu())
            loss_batch_GLR.append(loss_GLR.detach().cpu())
            loss_batch_importance.append(loss_importance.detach().cpu())

            if num_iter == 1:
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), 10, norm_type=2)
                num_iter = 0
                optimizer.step()
                classifier.zero_grad()

        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)

        train_instance_acc = np.mean(mean_correct)
        loss1 = np.mean(loss_batch)
        loss2 = np.mean(loss_batch_seg)
        loss3 = np.mean(loss_batch_GLR)
        loss4 = np.mean(loss_batch_importance)
        loss5 = np.mean(loss_batch_cls)
        log_string('Train accuracy is: %.5f' % train_instance_acc)
        log_string('Train loss: %.5f' % loss1)
        log_string('Train seg loss: %.5f' % loss2)
        log_string('Train cls loss: %.5f' % loss5)
        log_string('Train GLR loss: %.5f' % loss3)
        log_string('Train importance loss: %.5f' % loss4)
        
        log_string('lr: %.6f' % optimizer.param_groups[0]['lr'])

        with torch.no_grad():
            test_metrics = {}
            total_correct = 0
            total_seen = 0
            total_seen_class = [0 for _ in range(num_part)]
            total_correct_class = [0 for _ in range(num_part)]
            shape_ious = {cat: [] for cat in seg_classes.keys()}
            seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}

            for cat in seg_classes.keys():
                for label in seg_classes[cat]:
                    seg_label_to_cat[label] = cat

            classifier = classifier.eval()

            for batch_id, (points, label, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader),
                                                          smoothing=0.9):
                cur_batch_size, NUM_POINT, _ = points.size()
                points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
                points = points.transpose(2, 1)
                seg_pred, _, _, _, _ = classifier(points, to_categorical(label, num_classes))
                cur_pred_val = seg_pred.cpu().data.numpy()
                cur_pred_val_logits = cur_pred_val
                cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
                target = target.cpu().data.numpy()

                for i in range(cur_batch_size):
                    cat = seg_label_to_cat[target[i, 0]]
                    logits = cur_pred_val_logits[i, :, :]
                    cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]

                correct = np.sum(cur_pred_val == target)
                total_correct += correct
                total_seen += (cur_batch_size * NUM_POINT)

                for l in range(num_part):
                    total_seen_class[l] += np.sum(target == l)
                    total_correct_class[l] += (np.sum((cur_pred_val == l) & (target == l)))

                for i in range(cur_batch_size):
                    segp = cur_pred_val[i, :]
                    segl = target[i, :]
                    cat = seg_label_to_cat[segl[0]]
                    part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
                    for l in seg_classes[cat]:
                        if (np.sum(segl == l) == 0) and (
                                np.sum(segp == l) == 0):  # part is not present, no prediction as well
                            part_ious[l - seg_classes[cat][0]] = 1.0
                        else:
                            part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                                np.sum((segl == l) | (segp == l)))
                    shape_ious[cat].append(np.mean(part_ious))

            all_shape_ious = []
            for cat in shape_ious.keys():
                for iou in shape_ious[cat]:
                    all_shape_ious.append(iou)
                shape_ious[cat] = np.mean(shape_ious[cat])
            mean_shape_ious = np.mean(list(shape_ious.values()))
            test_metrics['accuracy'] = total_correct / float(total_seen)
            test_metrics['class_avg_accuracy'] = np.mean(
                np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float64))
            for cat in sorted(shape_ious.keys()):
                log_string('eval mIoU of %s %f' % (cat + ' ' * (14 - len(cat)), shape_ious[cat]))
            test_metrics['class_avg_iou'] = mean_shape_ious
            test_metrics['inctance_avg_iou'] = np.mean(all_shape_ious)

        log_string('Epoch %d test Accuracy: %f  Class avg mIOU: %f   Inctance avg mIOU: %f' % (
            epoch + 1, test_metrics['accuracy'], test_metrics['class_avg_iou'], test_metrics['inctance_avg_iou']))
        if (test_metrics['inctance_avg_iou'] >= best_inctance_avg_iou):
            logger.info('Save model...')
            savepath = str(checkpoints_dir) + '/best_model.pth'
            log_string('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'train_acc': train_instance_acc,
                'test_acc': test_metrics['accuracy'],
                'class_avg_iou': test_metrics['class_avg_iou'],
                'inctance_avg_iou': test_metrics['inctance_avg_iou'],
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string('Saving model....')

        if test_metrics['accuracy'] > best_acc:
            best_acc = test_metrics['accuracy']
        if test_metrics['class_avg_iou'] > best_class_avg_iou:
            best_class_avg_iou = test_metrics['class_avg_iou']
        if test_metrics['inctance_avg_iou'] > best_inctance_avg_iou:
            best_inctance_avg_iou = test_metrics['inctance_avg_iou']
        log_string('Best accuracy is: %.5f' % best_acc)
        log_string('Best class avg mIOU is: %.5f' % best_class_avg_iou)
        log_string('Best inctance avg mIOU is: %.5f' % best_inctance_avg_iou)
        global_epoch += 1


if __name__ == '__main__':
    args = parse_args()
    main(args)
