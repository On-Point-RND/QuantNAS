""" Training augmented model """
import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf as omg

from models.augment_cnn import AugmentCNN
import utils
from architect import Architect, ArchConstrains
from sr_base.datasets import PatchDataset
from genotypes import from_str

CFG_PATH = "./configs/config.yaml"


def train_setup(cfg):

    # INIT FOLDERS & cfg
    cfg_dataset = cfg.dataset
    cfg = cfg.train
    cfg.save = utils.get_run_path(cfg.log_dir, "TUNE_" + cfg.run_name)

    logger = utils.get_logger(cfg.save + "/log.txt")

    # FIX SEED
    np.random.seed(cfg.seed)
    torch.cuda.set_device(cfg.gpu)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.benchmark = True

    writer = SummaryWriter(log_dir=os.path.join(cfg.save, "board_train"))

    writer.add_hparams(
        hparam_dict={str(k): str(cfg[k]) for k in cfg},
        metric_dict={"tune/train/loss": 0},
    )

    with open(os.path.join(cfg.save, "config.txt"), "w") as f:
        for k, v in cfg.items():
            f.write(f"{str(k)}:{str(v)}\n")

    return cfg, writer, logger, cfg_dataset


def run_train(cfg):
    cfg, writer, logger, cfg_dataset = train_setup(cfg)

    logger.info("Logger is set - training start")

    # set default gpu device id
    device = cfg.gpu
    torch.cuda.set_device(device)

    train_data = PatchDataset(cfg_dataset, train=True)
    val_data = PatchDataset(cfg_dataset, train=False)

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.workers,
        pin_memory=False,
    )

    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.workers,
        pin_memory=False,
    )

    criterion = criterion = nn.L1Loss().to(device)

    with open(cfg.genotype_path, "r") as f:
        genotype = from_str(f.read())

    writer.add_text(tag="tune/arch/", text_string=str(genotype))
    print(genotype)

    model = AugmentCNN(
        cfg.channels,
        cfg.repeat_factor,
        genotype,
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
    optimizer = torch.optim.SGD(
        model.parameters(),
        cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
    )

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, cfg.epochs
    )

    best_score = 0.0
    # training loop
    for epoch in range(cfg.epochs):
        lr_scheduler.step()
        drop_prob = cfg.drop_path_prob * epoch / cfg.epochs
        model.drop_path_prob(drop_prob)

        # training
        score_train = train(
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
            score_val = score_val
            is_best = True
        else:
            is_best = False
        utils.save_checkpoint(model, cfg.save, is_best)

        print("")
        writer.add_scalars(
            "psnr/tune", {"val": score_val, "train": score_train}, epoch
        )

    logger.info("Final best PSNR = {:.4%}".format(best_score))


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
    psnr_meter = utils.AverageMeter()
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
        preds, aux_logits = model(X)
        loss = criterion(preds, y)
        loss.backward()
        # gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        psnr = utils.calc_psnr(preds, y)
        loss_meter.update(loss.item(), N)
        psnr_meter.update(psnr.item(), N)

        if step % cfg.print_freq == 0 or step == len(train_loader) - 1:
            logger.info(
                "Train: [{:3d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                "PSNR ({score.avg:.3f})".format(
                    epoch + 1,
                    cfg.epochs,
                    step,
                    len(train_loader) - 1,
                    losses=loss_meter,
                    score=psnr_meter,
                )
            )

        writer.add_scalar("tune/train/loss", loss.item(), cur_step)

        cur_step += 1

    logger.info(
        "Train: [{:3d}/{}] Final PSNR{:.3f}".format(
            epoch + 1, cfg.epochs, psnr_meter.avg
        )
    )
    return psnr_meter.avg


def validate(
    valid_loader, model, criterion, epoch, cur_step, writer, logger, device, cfg
):
    psnr_meter = utils.AverageMeter()
    loss_meter = utils.AverageMeter()

    model.eval()

    with torch.no_grad():
        for step, (
            X,
            y,
            x_path,
            y_path,
        ) in enumerate(valid_loader):
            X, y = X.to(device, non_blocking=True), y.to(
                device, non_blocking=True
            )
            N = X.size(0)

            preds, _ = model(X)
            loss = criterion(preds, y)

            psnr = utils.calc_psnr(preds, y)
            loss_meter.update(loss.item(), N)
            psnr_meter.update(psnr.item(), N)

        if step % cfg.print_freq == 0 or step == len(valid_loader) - 1:
            logger.info(
                "Train: [{:3d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                "PSNR ({score.avg:.3f})".format(
                    epoch + 1,
                    cfg.epochs,
                    step,
                    len(valid_loader) - 1,
                    losses=loss_meter,
                    score=psnr_meter,
                )
            )

    writer.add_scalar("tune/val/loss", loss_meter.avg, cur_step)

    logger.info(
        "Valid: [{:3d}/{}] Final PSNR{:.3f}".format(
            epoch + 1, cfg.epochs, psnr_meter.avg
        )
    )

    utils.save_images(cfg.save, x_path[0], y_path[0], preds[0], epoch, writer)

    return psnr_meter.avg


if __name__ == "__main__":
    cfg = omg.load(CFG_PATH)
    run_train(cfg)
