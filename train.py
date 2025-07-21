# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import os
import time
from cp_dataset import CPDataset, CPDataLoader
from networks import GicLoss, GMM, UnetGenerator, VGGLoss, load_checkpoint, save_checkpoint

from tensorboardX import SummaryWriter
from visualization import board_add_image, board_add_images


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="GMM")
    parser.add_argument("--gpu_ids", default="")
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('-b', '--batch-size', type=int, default=2)  # Reduced from 4
    parser.add_argument("--dataroot", default="data")
    parser.add_argument("--datamode", default="train")
    parser.add_argument("--stage", default="GMM")
    parser.add_argument("--data_list", default="train_pairs.txt")
    parser.add_argument("--val_data_list", default="val_pairs.txt")  # New validation file
    parser.add_argument("--fine_width", type=int, default=192)
    parser.add_argument("--fine_height", type=int, default=256)
    parser.add_argument("--radius", type=int, default=5)
    parser.add_argument("--grid_size", type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.00002,  # Reduced from 0.0001
                        help='initial learning rate for adam')
    parser.add_argument('--tensorboard_dir', type=str,
                        default='tensorboard', help='save tensorboard infos')
    parser.add_argument('--checkpoint_dir', type=str,
                        default='checkpoints', help='save checkpoint infos')
    parser.add_argument('--checkpoint', type=str, default='',
                        help='model checkpoint for initialization')
    parser.add_argument("--display_count", type=int, default=20)
    parser.add_argument("--save_count", type=int, default=1000)  # More frequent saves
    parser.add_argument("--keep_step", type=int, default=50000)  # Increased for exploration
    parser.add_argument("--decay_step", type=int, default=20000)  # Earlier decay
    parser.add_argument("--shuffle", action='store_true',
                        help='shuffle input data')

    opt = parser.parse_args()
    return opt


def validate_gmm(model, val_loader, criterionL1, gicloss):
    model.eval()
    total_loss = 0.0
    total_gic = 0.0
    with torch.no_grad():
        for inputs in val_loader:
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

    model.train()
    return total_loss / len(val_loader), total_gic / len(val_loader)


def train_gmm(opt, train_loader, val_loader, model, board):
    model.cuda()
    model.train()

    criterionL1 = nn.L1Loss()
    gicloss = GicLoss(opt)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0 -
                                                  max(0, step - opt.keep_step) / float(opt.decay_step + 1))

    best_val_loss = float('inf')
    patience = 5000  # Steps to wait for improvement
    trigger_times = 0

    for step in range(opt.keep_step + opt.decay_step):
        iter_start_time = time.time()
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

        visuals = [[im_h, shape, im_pose],
                   [c, warped_cloth, im_c],
                   [warped_grid, (warped_cloth+im)*0.5, im]]

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
            print('step: %8d, time: %.3f, loss: %4f, (40*Lgic): %.8f, Lwarp: %.6f' %
                  (step + 1, t, loss.item(), (40 * Lgic).item(), Lwarp.item()), flush=True)

        if (step + 1) % 500 == 0:  # Validate every 500 steps
            val_loss, val_gic = validate_gmm(model, val_loader, criterionL1, gicloss)
            board.add_scalar('val_loss', val_loss, step + 1)
            board.add_scalar('val_40*Lgic', val_gic, step + 1)
            print(f'Validation - step: {step + 1}, val_loss: {val_loss:.4f}, val_40*Lgic: {val_gic:.4f}')
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                trigger_times = 0
                save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, f'step_{step+1:06d}.pth'))
                print(f'Saved best model at step {step + 1} with val_loss {val_loss:.4f}')
            else:
                trigger_times += 500
                if trigger_times >= patience:
                    print(f'Early stopping at step {step + 1} with best val_loss {best_val_loss:.4f}')
                    break

        if (step + 1) % opt.save_count == 0:
            save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, f'step_{step+1:06d}.pth'))


def train_tom(opt, train_loader, val_loader, model, board):
    model.cuda()
    model.train()

    gmm_model = GMM(opt)
    gmm_checkpoint_path = os.path.join(opt.checkpoint_dir, "GMM-train-1", "step_034000.pth")
    if not os.path.exists(gmm_checkpoint_path):
        raise FileNotFoundError(f"GMM checkpoint not found at: {gmm_checkpoint_path}. Please ensure GMM training was completed and checkpoint exists.")
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
                   [original_c, c, m_composite*2-1],
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
            print('step: %8d, time: %.3f, loss: %.4f, l1: %.4f, vgg: %.4f, mask: %.4f'
                  % (step + 1, t, loss.item(), loss_l1.item(),
                     loss_vgg.item(), loss_mask.item()), flush=True)

        if (step + 1) % opt.save_count == 0:
            save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, f'step_{step+1:06d}.pth'))


def main():
    opt = get_opt()
    print(opt)
    print("Start to train stage: %s, named: %s!" % (opt.stage, opt.name))

    # create dataset and dataloader
    train_dataset = CPDataset(opt)
    train_loader = CPDataLoader(opt, train_dataset)

    # create validation dataloader
    val_dataset = CPDataset(opt, datamode='val', data_list=opt.val_data_list)
    val_loader = CPDataLoader(opt, val_dataset)

    # visualization
    if not os.path.exists(opt.tensorboard_dir):
        os.makedirs(opt.tensorboard_dir)
    board = SummaryWriter(logdir=os.path.join(opt.tensorboard_dir, opt.name))

    # create model & train & save the final checkpoint
    if opt.stage == 'GMM':
        model = GMM(opt)
        if not opt.checkpoint == '' and os.path.exists(opt.checkpoint):
            load_checkpoint(model, opt.checkpoint)
        train_gmm(opt, train_loader, val_loader, model, board)
        save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'gmm_final.pth'))
    elif opt.stage == 'TOM':
        model = UnetGenerator(35, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d)
        if not opt.checkpoint == '' and os.path.exists(opt.checkpoint):
            load_checkpoint(model, opt.checkpoint)
        train_tom(opt, train_loader, val_loader, model, board)
        save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'tom_final.pth'))
    else:
        raise NotImplementedError('Model [%s] is not implemented' % opt.stage)

    print('Finished training %s, named: %s!' % (opt.stage, opt.name))


if __name__ == "__main__":
    main()
