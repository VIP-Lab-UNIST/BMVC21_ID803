import time
import os.path as osp
import huepy as hue

import torch
from torch.nn.utils import clip_grad_norm_
from ignite.engine.engine import Engine, Events

from .logger import MetricLogger
from .misc import ship_data_to_cuda, warmup_lr_scheduler, resume_from_checkpoint

def get_trainer(args, model, train_loader, optimizer, lr_scheduler, device):

    def _update_model(engine, data):

        # Load data
        images, targets = ship_data_to_cuda(data, device)

        # Pass data to model
        loss_dict = model(engine.state.epoch, images, targets)

        # Total loss
        losses = args.train.w_RPN_loss_cls * loss_dict['loss_objectness'] \
            + args.train.w_RPN_loss_box * loss_dict['loss_rpn_box_reg'] \
            + args.train.w_RCNN_loss_bbox * loss_dict['loss_box_reg'] \
            + args.train.w_RCNN_loss_cls * loss_dict['loss_detection'] \
            + args.train.w_OIM_loss_oim * loss_dict['loss_reid']

        # Print 
        if engine.state.iteration % args.train.disp_interval == 0:
            loss_value = losses.item()
            state = dict(loss_value=loss_value,
                         lr=optimizer.param_groups[0]['lr'])
            state.update(loss_dict)
        else:
            state = None

        optimizer.zero_grad()
        losses.backward()
        if args.train.clip_gradient > 0:
            clip_grad_norm_(model.parameters(), args.train.clip_gradient)
        optimizer.step()

        return state

    trainer = Engine(_update_model)

    @trainer.on(Events.STARTED)
    def _init_run(engine):
        engine.state.epoch = args.train.start_epoch
        engine.state.iteration = args.train.start_epoch * len(train_loader)

    @trainer.on(Events.EPOCH_STARTED)
    def _init_epoch(engine):
        if engine.state.epoch == 1 and args.train.lr_warm_up:
            warmup_factor = 1. / 1000
            warmup_iters = len(train_loader) - 1
            engine.state.sub_scheduler = warmup_lr_scheduler(
                optimizer, warmup_iters, warmup_factor)
        engine.state.metric_logger = MetricLogger()
        print(hue.info(hue.bold(hue.green("Start training from %s epoch"%str(engine.state.epoch)))))

    @trainer.on(Events.ITERATION_STARTED)
    def _init_iter(engine):
        if engine.state.iteration % args.train.disp_interval == 0:
            engine.state.start = time.time()

    @trainer.on(Events.ITERATION_COMPLETED)
    def _post_iter(engine):
        if engine.state.epoch == 1 and args.train.lr_warm_up:  # epoch start from 1
            engine.state.sub_scheduler.step()

        if engine.state.iteration % args.train.disp_interval == 0:
            
            # Update logger
            batch_time = time.time() - engine.state.start
            engine.state.metric_logger.update(batch_time=batch_time)
            engine.state.metric_logger.update(**engine.state.output)
                
            # Print log on console
            step = (engine.state.iteration - 1) % len(train_loader) + 1
            engine.state.metric_logger.print_log(engine.state.epoch, step,
                                                    len(train_loader))

    @trainer.on(Events.EPOCH_COMPLETED)
    def _post_epoch(engine):
        lr_scheduler.step()
        
        if engine.state.epoch % 2 == 0:
            save_name = osp.join(args.path, 'checkpoint_epoch%d.pth'%engine.state.epoch)
        else:
            save_name = osp.join(args.path, 'checkpoint_epoch_last.pth')

        torch.save({
            'epoch': engine.state.epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict()
        }, save_name)

        print(hue.good('save model: {}'.format(save_name)))
        
    return trainer
