""" Training augmented model """
import os
import torch
import torch.nn as nn
import random
import logging
import copy
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf as omg
from sr_models.test_arch import ManualCNN, ESPCN, SRESPCN, SRResNet

from sr_models.augment_cnn import AugmentCNN
import utils
from sr_base.datasets import CropDataset

from genotypes import from_str


def train_setup(cfg):

    # INIT FOLDERS & cfg
    cfg.env.save = utils.get_run_path(
        cfg.env.log_dir, "TUNE_" + cfg.env.run_name
    )
    logger = utils.get_logger(cfg.env.save + "/log.txt")

    # FIX SEED
    np.random.seed(cfg.env.seed)
    if cfg.env.gpu != "cpu":
        torch.cuda.set_device(cfg.env.gpu)
    np.random.seed(cfg.env.seed)
    torch.manual_seed(cfg.env.seed)
    torch.cuda.manual_seed_all(cfg.env.seed)
    torch.backends.cudnn.benchmark = True

    writer = SummaryWriter(log_dir=os.path.join(cfg.env.save, "board_train"))

    writer.add_hparams(
        hparam_dict={str(k): str(cfg[k]) for k in cfg},
        metric_dict={"tune/train/loss": 0},
    )

    with open(os.path.join(cfg.env.save, "config.txt"), "w") as f:
        for k, v in cfg.items():
            f.write(f"{str(k)}:{str(v)}\n")

    return cfg, writer, logger


def run_train(cfg):
    cfg, writer, logger = train_setup(cfg)
    logger.info("Logger is set - training start")

    # set default gpu device id
    device = cfg.env.gpu
    if cfg.env.gpu != "cpu":
        torch.cuda.set_device(device)

    # TODO fix here and passing params from search config too
    # cfg_dataset.subset = None
    train_data = CropDataset(cfg.dataset, train=True)
    val_data = CropDataset(cfg.dataset, train=False)

    if cfg.dataset.debug_mode:
        indices = list(range(300))
        random.shuffle(indices)
        sampler_train = torch.utils.data.sampler.SubsetRandomSampler(
            indices[:150]
        )
    else:
        sampler_train = torch.utils.data.sampler.SubsetRandomSampler(
            list(range(len(train_data)))
        )

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=cfg.dataset.batch_size,
        sampler=sampler_train,
        # shuffle=True,
        num_workers=cfg.env.workers,
        pin_memory=False,
    )

    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=1,
        # sampler=sampler_val,
        shuffle=True,
        num_workers=cfg.env.workers,
        pin_memory=False,
    )

    criterion = nn.L1Loss().to(device)

    with open(cfg.train.genotype_path, "r") as f:
        genotype = from_str(f.read())

    writer.add_text(tag="tune/arch/", text_string=str(genotype))
    print(genotype)

    # model = ManualCNN(cfg.channels, cfg.repeat_factor)
    # model = SRResNet(4)
    # model = ESPCN(4)
    model = AugmentCNN(
        cfg.arch.channels,
        cfg.arch.c_fixed,
        cfg.arch.scale,
        genotype,
        blocks=cfg.arch.body_cells,
    )

    model.to(device)

    # model size
    mb_params = utils.param_size(model)
    logger.info("Model size = {:.3f} MB".format(mb_params))
    writer.add_text(
        tag="ModelParams",
        text_string=str("Model size = {:.3f} MB".format(mb_params)),
    )

    # weights optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )
    scheduler = {
        "cosine": torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, cfg.train.epochs
        ),
        "linear": torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=3, gamma=0.7
        ),
    }

    lr_scheduler = scheduler[cfg.train.lr_scheduler]

    best_score = 0.0
    # training loop
    for epoch in range(cfg.train.epochs):
        # training
        train(
            train_loader,
            model,
            optimizer,
            criterion,
            epoch,
            writer,
            logger,
            device,
            cfg,
        )
        lr_scheduler.step()
        # validation
        cur_step = (epoch + 1) * len(train_loader)
        score_val = validate(
            val_loader,
            model,
            criterion,
            epoch,
            cur_step,
            writer,
            logger,
            device,
            cfg,
        )

        # save
        if best_score < score_val:
            best_score = score_val
            is_best = True
        else:
            is_best = False
        utils.save_checkpoint(model, cfg.env.save, is_best)
        print("")
        writer.add_scalars("psnr/tune", {"val": score_val}, epoch)

    logger.info("Final best PSNR = {:.4%}".format(best_score))

    # FINISH TRAINING
    logging.shutdown()
    del model


def train(
    train_loader,
    model,
    optimizer,
    criterion,
    epoch,
    writer,
    logger,
    device,
    cfg,
):

    loss_meter = utils.AverageMeter()
    cur_step = epoch * len(train_loader)
    cur_lr = optimizer.param_groups[0]["lr"]
    logger.info("Epoch {} LR {}".format(epoch, cur_lr))
    writer.add_scalar("tune/train/lr", cur_lr, cur_step)
    model.train()

    for step, (X, y, _, _) in enumerate(train_loader):
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        N = X.size(0)
        optimizer.zero_grad()
        preds = model(X)
        loss = criterion(preds, y)
        loss_meter.update(loss.item(), N)
        loss.backward()

        optimizer.step()

        # loss_inter.update(intermediate_l[0].item(), N)

        if step % cfg.env.print_freq == 0 or step == len(train_loader) - 1:
            # if step % 3 == 0:
            #     logger.info(f"w skips: {[w.item() for w in model.skip_w]}")
            logger.info(
                "Train: [{:3d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.4f} ".format(
                    epoch + 1,
                    cfg.train.epochs,
                    step,
                    len(train_loader) - 1,
                    losses=loss_meter,
                )
            )

        writer.add_scalar("tune/train/loss", loss_meter.avg, cur_step)

        cur_step += 1

    return loss_meter.avg


def validate(
    valid_loader, model, criterion, epoch, cur_step, writer, logger, device, cfg
):
    val_psnr_meter = utils.AverageMeter()
    loss_meter = utils.AverageMeter()

    model.eval()

    with torch.no_grad():
        for step, (X, y, path_l, path_h) in enumerate(valid_loader):
            X, y = X.to(device, non_blocking=True), y.to(
                device, non_blocking=True
            )
            N = 1  # N = X.size(0)

            preds = model(X).clamp(0.0, 1.0)
            loss = criterion(preds.detach(), y)

            psnr = utils.compute_psnr(preds, y)
            loss_meter.update(loss.item(), N)
            val_psnr_meter.update(psnr, N)

        if step % cfg.env.print_freq == 0 or step == len(valid_loader) - 1:
            logger.info(
                "VAL: [{:3d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                "PSNR ({score.avg:.3f})".format(
                    epoch + 1,
                    cfg.train.epochs,
                    step,
                    len(valid_loader) - 1,
                    losses=loss_meter,
                    score=val_psnr_meter,
                )
            )

    writer.add_scalar("tune/val/loss", loss_meter.avg, cur_step)
    writer.add_scalar("tune/val/psnr", val_psnr_meter.avg, cur_step)

    logger.info(
        "Valid: [{:3d}/{}] Final PSNR{:.3f}".format(
            epoch + 1, cfg.train.epochs, val_psnr_meter.avg
        )
    )

    indx = random.randint(0, len(preds) - 1)
    utils.save_images(
        cfg.env.save, path_l[indx], path_h[indx], preds[indx], epoch, writer
    )

    return val_psnr_meter.avg


if __name__ == "__main__":
    CFG_PATH = "./configs/sr_config.yaml"
    cfg = omg.load(CFG_PATH)
    run_train(cfg)
