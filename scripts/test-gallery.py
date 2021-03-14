import os.path as osp
import huepy as hue
import json
import numpy as np
import torch
from torch.backends import cudnn

import sys
sys.path.append('./')
from configs import args_faster_rcnn

from lib.datasets import get_data_loader
from lib.model.faster_rcnn import get_model
from lib.utils.misc import lazy_arg_parse, Nestedspace, \
    resume_from_checkpoint
from lib.utils.evaluator import inference, detection_performance_calc


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

    gallery_loader, probe_loader = get_data_loader(args, train=False)

    model = get_model_fn(args, training=False,
                         pretrained_backbone=False)
    model.to(device)
    
    args.resume = osp.join(args.path, new_args.test.checkpoint_name)
    
    if osp.exists(args.resume):
        if not osp.exists(args.resume.replace('.pth', '-gallery.json')):
            args, model, _, _ = resume_from_checkpoint(args, model)

            name_to_boxes, all_feats, probe_feats = \
                inference(model, gallery_loader, probe_loader, device)

            print(hue.run('Evaluating detections:'))
            precision, recall = detection_performance_calc(gallery_loader.dataset,
                                                        name_to_boxes.values(),
                                                        det_thresh=0.01)

            print(hue.run('Evaluating search: '))
            # gallery_size = 100 
            if args.dataset == 'CUHK-SYSU' :
                gallery_sizes = [ 200, 500, 1000, 2000, 4000, -1]
            else:
                gallery_sizes = [-1]
            for gallery_size in gallery_sizes:
                ret = gallery_loader.dataset.search_performance_calc(
                    gallery_loader.dataset, probe_loader.dataset,
                    name_to_boxes.values(), all_feats, probe_feats,
                    det_thresh=0.9, gallery_size=gallery_size)

                performance = {}
                performance['mAP'] = ret['mAP']
                performance['top_k'] = ret['accs'].tolist()
                performance['precision'] = precision
                performance['recall'] = recall
                print(performance)
                if gallery_size > 0:
                    with open(args.resume.replace('.pth', '-gallery-%d.json'%gallery_size), 'w') as f:
                        json.dump(performance, f)
                else:
                    with open(args.resume.replace('.pth', '-gallery.json'), 'w') as f:
                        json.dump(performance, f)

            # import IPython
            # IPython.embed()
            return ret['mAP']

if __name__ == '__main__':
    arg_parser = args_faster_rcnn()
    new_args = lazy_arg_parse(arg_parser)
    main(new_args, get_model)
