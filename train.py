# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
from pathlib import Path
import time
from cp_dataset import CPDataset, CPDataLoader
from networks import GicLoss, GMM, UnetGenerator, VGGLoss, load_checkpoint, save_checkpoint
from tensorboardX import SummaryWriter
from visualization import board_add_image, board_add_images
from PIL import Image
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="GMM")
    parser.add_argument("--gpu_ids", default="")
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('-b', '--batch-size', type=int, default=2)
    parser.add_argument("--dataroot", default="data")
    parser.add_argument("--datamode", default="train")
    parser.add_argument("--stage", default="GMM")
    parser.add_argument("--data_list", default="train_pairs.txt")
    parser.add_argument("--val_data_list", default="val_pairs.txt")
    parser.add_argument("--fine_width", type=int, default=192)
    parser.add_argument("--fine_height", type=int, default=256)
    parser.add_argument("--radius", type=int, default=5)
    parser.add_argument("--grid_size", type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument("--display_count", type=int, default=20)
    parser.add_argument("--save_count", type=int, default=1000)
    parser.add_argument("--keep_step", type=int, default=50000)
    parser.add_argument("--decay_step", type=int, default=20000)
    parser.add_argument("--shuffle", action='store_true')
    opt = parser.parse_args()
    return opt

def validate_gmm(model, val_loader, criterionL1, gicloss):
    model.eval()
    total_loss = 0.0
    total_gic = 0.0
    num_batches = len(val_loader.data_loader)
    val_iter = val_loader.data_iter

    with torch.no_grad():
        for _ in range(num_batches):
            try:
                inputs = val_loader.next_batch()
                im = inputs['image'].cuda()
                im_pose = inputs['pose_image'].cuda()
                im_h = inputs['head'].cuda()
                shape = inputs['shape'].cuda()
                agnostic = inputs['agnostic'].cuda()
                c = inputs['cloth'].cuda()
                cm = inputs['cloth_mask'].cuda()
                im_c = inputs['parse_cloth'].cuda()
                im_g = inputs['grid_image'].cuda()

                grid, theta = model(agnostic, cm)
                warped_cloth = F.grid_sample(c, grid, padding_mode='border')
                warped_mask = F.grid_sample(cm, grid, padding_mode='zeros')
                warped_grid = F.grid_sample(im_g, grid, padding_mode='zeros')

                Lwarp = criterionL1(warped_cloth, im_c)
                Lgic = gicloss(grid) / (grid.shape[0] * grid.shape[1] * grid.shape[2])
                loss = Lwarp + 40 * Lgic

                total_loss += loss.item()
                total_gic += (40 * Lgic).item()
            except Exception as e:
                logger.error(f"Validation error at batch {_}: {str(e)}")
                continue

    model.train()
    avg_loss = total_loss / max(1, num_batches)
    avg_gic = total_gic / max(1, num_batches)
    return avg_loss, avg_gic

def train_gmm(opt, train_loader, val_loader, model, board):
    model.cuda()
    model.train()
    criterionL1 = nn.L1Loss()
    gicloss = GicLoss(opt)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0 -
                                                 max(0, step - opt.keep_step) / float(opt.decay_step + 1))
    best_val_loss = float('inf')
    patience = 5000
    trigger_times = 0
    checkpoint_dir = Path(opt.checkpoint_dir) / opt.name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    warp_dir = checkpoint_dir / 'warp-cloth'
    warp_mask_dir = checkpoint_dir / 'warp-mask'
    warp_dir.mkdir(parents=True, exist_ok=True)
    warp_mask_dir.mkdir(parents=True, exist_ok=True)

    for step in range(opt.keep_step + opt.decay_step):
        iter_start_time = time.time()
        try:
            inputs = train_loader.next_batch()
            im = inputs['image'].cuda()
            im_pose = inputs['pose_image'].cuda()
            im_h = inputs['head'].cuda()
            shape = inputs['shape'].cuda()
            agnostic = inputs['agnostic'].cuda()
            c = inputs['cloth'].cuda()
            cm = inputs['cloth_mask'].cuda()
            im_c = inputs['parse_cloth'].cuda()
            im_g = inputs['grid_image'].cuda()

            grid, theta = model(agnostic, cm)
            warped_cloth = F.grid_sample(c, grid, padding_mode='border')
            warped_mask = F.grid_sample(cm, grid, padding_mode='zeros')
            warped_grid = F.grid_sample(im_g, grid, padding_mode='zeros')

            if (step + 1) % opt.save_count == 0:
                for i in range(min(1, c.size(0))):
                    c_name = inputs['c_name'][i].split('/')[-1].replace('.jpg', f'_step{step+1:06d}.png')
                    save_cloth = (warped_cloth[i].detach().cpu().clamp(-1, 1) * 0.5 + 0.5).numpy().transpose(1, 2, 0) * 255
                    save_mask = (warped_mask[i].detach().cpu().numpy()[0] * 255).astype(np.uint8)
                    Image.fromarray(save_cloth.astype(np.uint8)).save(warp_dir / c_name)
                    Image.fromarray(save_mask).save(warp_mask_dir / c_name)

            visuals = [[im_h, shape, im_pose],
                      [c, warped_cloth, im_c],
                      [warped_grid, (warped_cloth + im) * 0.5, im]]

            Lwarp = criterionL1(warped_cloth, im_c)
            Lgic = gicloss(grid) / (grid.shape[0] * grid.shape[1] * grid.shape[2])
            loss = Lwarp + 40 * Lgic

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (step + 1) % opt.display_count == 0:
                board_add_images(board, 'combine', visuals, step + 1)
                board.add_scalar('loss', loss.item(), step + 1)
                board.add_scalar('40*Lgic', (40 * Lgic).item(), step + 1)
                board.add_scalar('Lwarp', Lwarp.item(), step + 1)
                t = time.time() - iter_start_time
                logger.info(f'Step: {step + 1}, Time: {t:.3f}s, Loss: {loss.item():.4f}, '
                           f'40*Lgic: {(40 * Lgic).item():.8f}, Lwarp: {Lwarp.item():.6f}')

            if (step + 1) % 500 == 0:
                val_loss, val_gic = validate_gmm(model, val_loader, criterionL1, gicloss)
                board.add_scalar('val_loss', val_loss, step + 1)
                board.add_scalar('val_40*Lgic', val_gic, step + 1)
                logger.info(f'Validation - Step: {step + 1}, Val Loss: {val_loss:.4f}, Val 40*Lgic: {val_gic:.4f}')
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    trigger_times = 0
                    save_checkpoint(model, checkpoint_dir / f'step_{step+1:06d}.pth')
                    logger.info(f'Saved best model at step {step + 1} with val_loss {best_val_loss:.4f}')
                else:
                    trigger_times += 500
                    if trigger_times >= patience:
                        logger.info(f'Early stopping at step {step + 1} with best val_loss {best_val_loss:.4f}')
                        break

            if (step + 1) % opt.save_count == 0:
                save_checkpoint(model, checkpoint_dir / f'step_{step+1:06d}.pth')

        except Exception as e:
            logger.error(f"Training error at step {step + 1}: {str(e)}")
            continue

def train_tom(opt, train_loader, val_loader, model, board):
    model.cuda()
    model.train()
    gmm_model = GMM(opt)
    gmm_checkpoint_path = opt.checkpoint if opt.checkpoint else '/kaggle/working/model_weights/step_010000.pth'
    if not os.path.exists(gmm_checkpoint_path):
        raise FileNotFoundError(f"GMM checkpoint not found at: {gmm_checkpoint_path}")
    load_checkpoint(gmm_model, gmm_checkpoint_path)
    gmm_model.eval()

    criterionL1 = nn.L1Loss()
    criterionVGG = VGGLoss()
    criterionMask = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0 -
                                                 max(0, step - opt.keep_step) / float(opt.decay_step + 1))

    for step in range(opt.keep_step + opt.decay_step):
        iter_start_time = time.time()
        try:
            inputs = train_loader.next_batch()
            im = inputs['image'].cuda()
            im_pose = inputs['pose_image']
            im_h = inputs['head']
            shape = inputs['shape']
            agnostic = inputs['agnostic'].cuda()
            original_c = inputs['cloth'].cuda()
            original_cm = inputs['cloth_mask'].cuda()
            pcm = inputs['parse_cloth_mask'].cuda()

            with torch.no_grad():
                grid, _ = gmm_model(agnostic, original_cm)
            c = F.grid_sample(original_c, grid, padding_mode='border')
            cm = F.grid_sample(original_cm, grid, padding_mode='zeros')

            outputs = model(torch.cat([agnostic, c, cm], 1))
            p_rendered, m_composite = torch.split(outputs, 3, 1)
            p_rendered = F.tanh(p_rendered)
            m_composite = F.sigmoid(m_composite)
            p_tryon = c * m_composite + p_rendered * (1 - m_composite)

            visuals = [[im_h, shape, im_pose],
                      [original_c, c, m_composite * 2 - 1],
                      [p_rendered, p_tryon, im]]

            loss_l1 = criterionL1(p_tryon, im)
            loss_vgg = criterionVGG(p_tryon, im)
            loss_mask = criterionMask(m_composite, pcm)
            loss = loss_l1 + loss_vgg + loss_mask
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (step + 1) % opt.display_count == 0:
                board_add_images(board, 'combine', visuals, step + 1)
                board.add_scalar('metric', loss.item(), step + 1)
                board.add_scalar('L1', loss_l1.item(), step + 1)
                board.add_scalar('VGG', loss_vgg.item(), step + 1)
                board.add_scalar('MaskL1', loss_mask.item(), step + 1)
                t = time.time() - iter_start_time
                logger.info(f'Step: {step + 1}, Time: {t:.3f}s, Loss: {loss.item():.4f}, '
                           f'L1: {loss_l1.item():.4f}, VGG: {loss_vgg.item():.4f}, Mask: {loss_mask.item():.4f}')

            if (step + 1) % opt.save_count == 0:
                save_checkpoint(model, Path(opt.checkpoint_dir) / opt.name / f'step_{step+1:06d}.pth')

        except Exception as e:
            logger.error(f"Training error at step {step + 1}: {str(e)}")
            continue

def main():
    opt = get_opt()
    logger.info(str(opt))
    logger.info(f"Start to train stage: {opt.stage}, named: {opt.name}")

    # Validate dataset
    data_path = Path(opt.dataroot)
    required_dirs = ['cloth', 'cloth-mask', 'image', 'image-parse-new', 'image-mask', 'openpose_json']
    for d in required_dirs:
        if not (data_path / d).exists():
            raise FileNotFoundError(f"Required directory {data_path / d} does not exist")

    # Create dataset and dataloader
    train_dataset = CPDataset(opt)
    train_loader = CPDataLoader(opt, train_dataset)
    val_dataset = CPDataset(opt, datamode='val', data_list=opt.val_data_list)
    val_loader = CPDataLoader(opt, val_dataset)

    # Visualization
    tensorboard_dir = Path(opt.tensorboard_dir) / opt.name
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    board = SummaryWriter(logdir=str(tensorboard_dir))

    # Create model & train
    if opt.stage == 'GMM':
        model = GMM(opt)
        if opt.checkpoint and os.path.exists(opt.checkpoint):
            load_checkpoint(model, opt.checkpoint)
        train_gmm(opt, train_loader, val_loader, model, board)
        save_checkpoint(model, Path(opt.checkpoint_dir) / opt.name / 'gmm_final.pth')
    elif opt.stage == 'TOM':
        model = UnetGenerator(35, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d)
        if opt.checkpoint and os.path.exists(opt.checkpoint):
            load_checkpoint(model, opt.checkpoint)
        train_tom(opt, train_loader, val_loader, model, board)
        save_checkpoint(model, Path(opt.checkpoint_dir) / opt.name / 'tom_final.pth')
    else:
        raise NotImplementedError(f'Model {opt.stage} is not implemented')

    logger.info(f'Finished training {opt.stage}, named: {opt.name}')

if __name__ == "__main__":
    main()
