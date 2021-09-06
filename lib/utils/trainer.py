import time
import os.path as osp
import huepy as hue

import torch
from torch.nn.utils import clip_grad_norm_

from .logger import MetricLogger
from .misc import ship_data_to_cuda, warmup_lr_scheduler, resume_from_checkpoint

class Trainer():

    def __init__(self, args, model, train_loader, optimizer, lr_scheduler, device):

        self.args = args
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device

    def run(self):

        steps = 0
        for epoch in range(self.args.train.start_epoch, self.args.train.epochs):

            ## Initial epochs
            if epoch == 1 and self.args.train.lr_warm_up:
                warmup_factor = 1. / 1000
                warmup_iters = len(self.train_loader) - 1
                sub_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)
            metric_logger = MetricLogger()
            print(hue.info(hue.bold(hue.green("Start training from %s epoch"%str(epoch)))))

            for iteration, data in enumerate(self.train_loader):
                ## Initial iterations
                steps = epoch*len(self.train_loader) + iteration
                if steps % self.args.train.disp_interval == 0:
                    start = time.time()

                # Load data
                images, targets = ship_data_to_cuda(data, self.device)

                # Pass data to model
                loss_dict = self.model(epoch, images, targets)

                # Total loss
                losses = self.args.train.w_RPN_loss_cls * loss_dict['loss_objectness'] \
                    + self.args.train.w_RPN_loss_box * loss_dict['loss_rpn_box_reg'] \
                    + self.args.train.w_RCNN_loss_bbox * loss_dict['loss_box_reg'] \
                    + self.args.train.w_RCNN_loss_cls * loss_dict['loss_detection'] \
                    + self.args.train.w_OIM_loss_oim * loss_dict['loss_reid']

                self.optimizer.zero_grad()
                losses.backward()
                if self.args.train.clip_gradient > 0:
                    clip_grad_norm_(self.model.parameters(), self.args.train.clip_gradient)
                self.optimizer.step()

                ## Post iteraions
                if epoch == 1 and self.args.train.lr_warm_up:
                    sub_scheduler.step()

                if steps % self.args.train.disp_interval == 0:
                    # Print 
                    loss_value = losses.item()
                    state = dict(loss_value=loss_value,
                                lr=self.optimizer.param_groups[0]['lr'])
                    state.update(loss_dict)

                    # Update logger
                    batch_time = time.time() - start
                    metric_logger.update(batch_time=batch_time)
                    metric_logger.update(**state)
                        
                    # Print log on console
                    metric_logger.print_log(epoch, iteration, len(self.train_loader))
                else:
                    state = None

            ## Post epochs
            self.lr_scheduler.step()
            if epoch % 2 == 0:
                save_name = osp.join(self.args.path, 'checkpoint_epoch%d.pth'%epoch)
            else:
                save_name = osp.join(self.args.path, 'checkpoint_epoch_last.pth')

            torch.save({
                'epoch': epoch,
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict()
            }, save_name)
            print(hue.good('save model: {}'.format(save_name)))

        return None
