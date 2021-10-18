import os
import torch
import random
import logging
from omegaconf import OmegaConf as omg
from genotypes import from_str
from sr_models.augment_cnn import AugmentCNN
import utils
from sr_base.datasets import PatchDataset
from genotypes import from_str


def get_model(
    weights_path, device, genotype, channels=3, repeat_factor=16, blocks=4
):
    model = AugmentCNN(channels, repeat_factor, genotype, blocks)

    model_ = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(model_.state_dict())
    model.to(device)
    return model


def run_val(model, cfg_val, save_dir, device):
    # set default gpu device id
    torch.cuda.set_device(device)
    val_data = PatchDataset(cfg_val, train=None)

    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=1,
        # sampler=sampler_val,
        shuffle=False,
        num_workers=2,
        pin_memory=False,
    )

    score_val = validate(val_loader, model, device, save_dir)
    random_image = torch.randn(1, 3, 32, 32).cuda(device)
    _ = model(random_image)
    flops_32 = model.fetch_flops()

    random_image = torch.randn(1, 3, 256, 256).cuda(device)
    _ = model(random_image)
    flops_256 = model.fetch_flops()

    mb_params = utils.param_size(model)
    return score_val, flops_32, flops_256, mb_params


def validate(valid_loader, model, device, save_dir):
    psnr_meter = utils.AverageMeter()

    model.eval()

    with torch.no_grad():
        for (
            X,
            y,
            x_path,
            y_path,
        ) in valid_loader:
            X, y = X.to(device, non_blocking=True), y.to(device)
            N = X.size(0)

            preds = model(X).clamp(0.0, 1.0)

            psnr = utils.calc_psnr(preds, y)
            psnr_meter.update(psnr.item(), N)

    indx = random.randint(0, len(x_path) - 1)
    utils.save_images(
        save_dir,
        x_path[indx],
        y_path[indx],
        preds[indx],
        cur_iter=0,
        logger=None,
    )

    return psnr_meter.avg


def dataset_loop(valid_cfg, model, logger, save_dir, device):
    for dataset in valid_cfg:
        os.makedirs(os.path.join(save_dir, str(dataset)), exist_ok=True)
        score_val, flops_32, flops_256, mb_params = run_val(
            model,
            valid_cfg[dataset],
            os.path.join(save_dir, str(dataset)),
            device,
        )
        logger.info("\n{}:".format(str(dataset)))
        logger.info("Model size = {:.3f} MB".format(mb_params))
        logger.info("Flops = {:.2e} operations 32x32".format(flops_32))
        logger.info("Flops = {:.2e} operations 256x256".format(flops_256))
        logger.info("PSNR = {:.3}%".format(score_val))


if __name__ == "__main__":
    CFG_PATH = "./sr_models/valsets4x.yaml"
    valid_cfg = omg.load(CFG_PATH)
    run_name = "TEST_2"
    genotype_path = "./genotype_example_sr.gen"
    weights_path = "/home/dev/data/logs/TUNE_TEST-2021-10-11-18/best.pth.tar"
    log_dir = "/home/dev/data/logs/VAL_LOGS"
    save_dir = os.path.join(log_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)
    channels = 3
    repeat_factor = 16
    device = 0

    with open(genotype_path, "r") as f:
        genotype = from_str(f.read())

    logger = utils.get_logger(save_dir + "/validation_log.txt")
    logger.info(genotype)
    model = get_model(
        weights_path, device, genotype, channels=3, repeat_factor=16, blocks=3
    )
    dataset_loop(valid_cfg, model, logger, save_dir, device)
