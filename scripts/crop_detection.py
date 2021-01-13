import os.path as osp
import huepy as hue
import json
import numpy as np
import torch
from torch.backends import cudnn

import sys
sys.path.append('./')
from configs import args_faster_rcnn_ortho_featuring

from lib.datasets import get_data_loader
from lib.model.faster_rcnn_ortho_featuring import get_model
from lib.utils.misc import lazy_arg_parse, Nestedspace, \
    resume_from_checkpoint
from lib.utils.evaluator import inference, detection_performance_calc, draw


def main(new_args, get_model_fn):
    args = Nestedspace()
    args.load_from_json(osp.join(new_args.path, 'args.json'))
    args.from_dict(new_args.to_dict())  # override previous args

    device = torch.device(args.device)
    cudnn.benchmark = False

    print(hue.info(hue.bold(hue.lightgreen(
        'Working directory: {}'.format(args.path)))))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Do not use flip
    args.train.use_flipped=False
    
    # train_loader, train_info = get_data_loader(args, train=True)
    gallery_loader, probe_loader = get_data_loader(args, train=False)
    
    model = get_model_fn(args, training=False,
                         pretrained_backbone=False)
    model.eval().to(device)
    
    args.resume = osp.join(args.path, new_args.test.checkpoint_name)

    if osp.exists(args.resume):
       
        args, model, _, _ = resume_from_checkpoint(args, model)

        # draw(model, train_loader, device)
        draw(model, gallery_loader, device)

if __name__ == '__main__':
    arg_parser = args_faster_rcnn_ortho_featuring()
    new_args = lazy_arg_parse(arg_parser)
    main(new_args, get_model)
