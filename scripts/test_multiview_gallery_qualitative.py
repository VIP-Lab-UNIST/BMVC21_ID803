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

import cv2
from tqdm import tqdm

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
        print(args.resume.replace('.pth', '_multiview_gallery_qualitative.json'))
        if not osp.exists(args.resume.replace('.pth', '_multiview_gallery_qualitative.json')):
            args, model, _, _ = resume_from_checkpoint(args, model)

            # name_to_boxes, all_feats, probe_feats = \
            #     inference(model, gallery_loader, probe_loader, device)

            # torch.save([name_to_boxes, all_feats, probe_feats], 'inference_results.pth')
            # # assert False
            # # [name_to_boxes, all_feats, probe_feats] = torch.load('inference_results.pth')
            # # print('inference_results.pth is loaded')

            # # print(hue.run('Evaluating detections:'))
            # # precision, recall = detection_performance_calc(gallery_loader.dataset,
            # #                                             name_to_boxes.values(),
            # #                                             det_thresh=0.01)

            # print(hue.run('Evaluating search: '))
            # if args.dataset == 'CUHK-SYSU':
            #     gallery_size = 100 
            #     torch.save([name_to_boxes, all_feats, probe_feats], 'inference_cuhk.pth')
            #     ret = gallery_loader.dataset.search_performance_calc(
            #         gallery_loader.dataset, probe_loader.dataset,
            #         name_to_boxes.values(), all_feats, probe_feats,
            #         det_thresh=0.9, gallery_size=gallery_size)
            #     torch.save(ret, 'inference_ret_cuhk.pth')
            #     dataset_path = '/root/workspace/datasets/PersonSearch/CUHK-SYSU/Image/SSM'
            # else :
            #     gallery_size = -1 
            #     torch.save([name_to_boxes, all_feats, probe_feats], 'inference_prw.pth')
            #     ret = gallery_loader.dataset.search_performance_calc(
            #         gallery_loader.dataset, probe_loader.dataset,
            #         name_to_boxes.values(), all_feats, probe_feats,
            #         det_thresh=0.9, gallery_size=gallery_size,
            #         ignore_cam_id=False,
            #         remove_unlabel=False)
            #     torch.save(ret, 'inference_ret_prw.pth')
            #     dataset_path = '/root/workspace/datasets/PersonSearch/PRW-v16.04.20/frames'

            # assert False

            ret = torch.load('inference_ret_prw.pth')
            dataset_path = '/root/workspace/datasets/PersonSearch/PRW-v16.04.20/frames'

            performance = {}
            performance['mAP'] = ret['mAP']
            performance['top_k'] = ret['accs'].tolist()
            
            ## Save qualitative results
            try:
                os.mkdir(args.resume[:-4])
            except:
                pass
            
            for k in tqdm(range(len(ret['results']))):
                results_dir = osp.join(args.resume[:-4], '%04d_%s' % (k+1, ret['results'][k]['probe_img'][:-4]))
                # print(results_dir)
                try:
                    os.mkdir(results_dir)
                except:
                    pass
                img = cv2.imread('%s/%s' % (dataset_path, ret['results'][k]['probe_img']))
                ## Cropped results
                rect = ret['results'][k]['probe_roi']
                results_filename = '00_probe.jpg'
                crop_img = img[int(rect[1]):int(rect[3]), int(rect[0]):int(rect[2])]
                cv2.imwrite(osp.join(results_dir, results_filename), crop_img)
                for p, matching in enumerate(ret['results'][k]['gallery']):
                    img = cv2.imread('%s/%s' % (dataset_path, matching['img']))
                    rect = matching['roi']
                    if matching['correct']>0:    
                        results_filename = '%02d_%s_true.jpg' %(p+1, matching['img'][:-4])
                    else:
                        results_filename = '%02d_%s_false.jpg' %(p+1, matching['img'][:-4])
                    crop_img = img[int(rect[1]):int(rect[3]), int(rect[0]):int(rect[2])]
                    cv2.imwrite(osp.join(results_dir, results_filename), crop_img)

            
            # with open(args.resume.replace('.pth', '_multiview_gallery_qualitative.json'), 'w') as f:
            #     json.dump(performance, f)

            # import IPython
            # IPython.embed()
            return ret['mAP']

if __name__ == '__main__':
    arg_parser = args_faster_rcnn()
    new_args = lazy_arg_parse(arg_parser)
    main(new_args, get_model)
