import os 
import sys
sys.path.append('./')

from datetime import datetime
import numpy as np
import huepy as hue

import torch
from configs import args_faster_rcnn

from lib.model.faster_rcnn import get_model
from lib.datasets import get_data_loader
from lib.utils.misc import Nestedspace, get_optimizer, get_lr_scheduler
from lib.utils.trainer import get_trainer


def main(args, get_model_fn):

    device = torch.device(args.device)
    torch.cuda.set_device(0)

    ## Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    ## Determine checkpoint path and files
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    args.path = os.path.join(args.path, current_time)
    try:
        os.makedirs(args.path)
    except:
        pass 
    print(hue.info(hue.bold(hue.lightgreen(
        'Working directory: {}'.format(args.path)))))
    
    args.export_to_json(os.path.join(args.path, 'args.json'))
    train_loader, train_info = get_data_loader(args, train=True)

    ## Load model
    model = get_model_fn(args, training=True,
                         pretrained_backbone=True)
    model.to(device)
    model.roi_heads.reid_regressor.set_scene_vector(train_info)

    ## Set optimizer and scheduler
    optimizer = get_optimizer(args, model)
    lr_scheduler = get_lr_scheduler(args, optimizer)
    
    ## Define and run trainer
    trainer = get_trainer(args, model, train_loader, optimizer,
                          lr_scheduler, device)
    trainer.run(train_loader, max_epochs=args.train.epochs)

if __name__ == '__main__':
    arg_parser = args_faster_rcnn()
    args = arg_parser.parse_args(namespace=Nestedspace())
    main(args, get_model)
    