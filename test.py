# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from pathlib import Path
import argparse
import time
from cp_dataset import CPDataset, CPDataLoader
from networks import GMM, UnetGenerator, load_checkpoint
from tensorboardX import SummaryWriter
from visualization import board_add_image, board_add_images, save_images
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="TOM-train-1")
    parser.add_argument("--gpu_ids", default="0")
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('-b', '--batch-size', type=int, default=4)
    parser.add_argument("--dataroot", default="data")
    parser.add_argument("--datamode", default="test")
    parser.add_argument("--stage", default="TOM")
    parser.add_argument("--data_list", default="test_pairs.txt")
    parser.add_argument("--fine_width", type=int, default=192)
    parser.add_argument("--fine_height", type=int, default=256)
    parser.add_argument("--radius", type=int, default=5)
    parser.add_argument("--grid_size", type=int, default=5)
    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard')
    parser.add_argument('--result_dir', type=str, default='result')
    parser.add_argument('--checkpoint', type=str, default='/kaggle/working/model_weights/step_001000.pth')
    parser.add_argument('--gmm_checkpoint', type=str, default='/kaggle/working/model_weights/step_010000.pth')
    parser.add_argument("--display_count", type=int, default=1)
    parser.add_argument("--shuffle", action='store_true')
    opt = parser.parse_args()
    return opt

def test_gmm(opt, test_loader, model, board):
    model.cuda()
    model.eval()
    base_name = os.path.basename(opt.checkpoint)
    name = opt.name
    save_dir = Path(opt.result_dir) / name / opt.datamode
    save_dir.mkdir(parents=True, exist_ok=True)
    warp_cloth_dir = save_dir / 'warp-cloth'
    warp_mask_dir = save_dir / 'warp-mask'
    result_dir1 = save_dir / 'result_dir'
    overlayed_TPS_dir = save_dir / 'overlayed_TPS'
    warped_grid_dir = save_dir / 'warped_grid'
    for d in [warp_cloth_dir, warp_mask_dir, result_dir1, overlayed_TPS_dir, warped_grid_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    for step, inputs in enumerate(test_loader.data_loader):
        try:
            iter_start_time = time.time()
            c_names = inputs['c_name']
            im_names = inputs['im_name']
            im = inputs['image'].cuda()
            im_pose = inputs['pose_image'].cuda()
            im_h = inputs['head'].cuda()
            shape = inputs['shape'].cuda()
            agnostic = inputs['agnostic'].cuda()
            c = inputs['cloth'].cuda()
            cm = inputs['cloth_mask'].cuda()
            im_c = inputs['parse_cloth'].cuda()
            im_g = inputs['grid_image'].cuda()
            shape_ori = inputs['shape_ori'].cuda()

            grid, theta = model(agnostic, cm)
            warped_cloth = F.grid_sample(c, grid, padding_mode='border')
            warped_mask = F.grid_sample(cm, grid, padding_mode='zeros')
            warped_grid = F.grid_sample(im_g, grid, padding_mode='zeros')
            overlay = 0.7 * warped_cloth + 0.3 * im

            visuals = [[im_h, shape, im_pose],
                      [c, warped_cloth, im_c],
                      [warped_grid, (warped_cloth + im) * 0.5, im]]

            save_images(warped_cloth, im_names, warp_cloth_dir)
            save_images(warped_mask * 2 - 1, im_names, warp_mask_dir)
            save_images(shape_ori * 0.2 + warped_cloth * 0.8, im_names, result_dir1)
            save_images(warped_grid, im_names, warped_grid_dir)
            save_images(overlay, im_names, overlayed_TPS_dir)

            if (step + 1) % opt.display_count == 0:
                board_add_image(board, 'combine', visuals, step + 1)
                t = time.time() - iter_start_time
                logger.info(f'Step: {step + 1}, Time: {t:.3f}s')
        except Exception as e:
            logger.error(f"Error processing batch {step + 1}: {str(e)}")
            continue

def test_tom(opt, test_loader, model, gmm_model, board):
    model.cuda()
    model.eval()
    gmm_model.cuda()
    gmm_model.eval()
    save_dir = Path(opt.result_dir) / opt.name / opt.datamode
    save_dir.mkdir(parents=True, exist_ok=True)
    try_on_dir = save_dir / 'try-on'
    p_rendered_dir = save_dir / 'p_rendered'
    m_composite_dir = save_dir / 'm_composite'
    im_pose_dir = save_dir / 'im_pose'
    shape_dir = save_dir / 'shape'
    im_h_dir = save_dir / 'im_h'
    for d in [try_on_dir, p_rendered_dir, m_composite_dir, im_pose_dir, shape_dir, im_h_dir]:
        d.mkdir(parents=True, exist_ok=True)

    logger.info(f'Dataset size: {len(test_loader.dataset)}')
    for step, inputs in enumerate(test_loader.data_loader):
        try:
            iter_start_time = time.time()
            im_names = inputs['im_name']
            im = inputs['image'].cuda()
            im_pose = inputs['pose_image']
            im_h = inputs['head']
            shape = inputs['shape']
            agnostic = inputs['agnostic'].cuda()
            original_c = inputs['cloth'].cuda()
            original_cm = inputs['cloth_mask'].cuda()
            pcm = inputs['parse_cloth_mask'].cuda()

            adjusted_c_names = [c_name.replace('train/', '') for c_name in inputs['c_name']]

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
                      [original_c, 2 * original_cm - 1, m_composite],
                      [p_rendered, p_tryon, im]]

            save_images(p_tryon, im_names, try_on_dir)
            save_images(im_h, im_names, im_h_dir)
            save_images(shape, im_names, shape_dir)
            save_images(im_pose, im_names, im_pose_dir)
            save_images(m_composite, im_names, m_composite_dir)
            save_images(p_rendered, im_names, p_rendered_dir)

            if (step + 1) % opt.display_count == 0:
                board_add_images(board, 'combine', visuals, step + 1)
                t = time.time() - iter_start_time
                logger.info(f'Step: {step + 1}, Time: {t:.3f}s')
        except Exception as e:
            logger.error(f"Error processing batch {step + 1}: {str(e)}")
            continue

def main():
    opt = get_opt()
    logger.info(str(opt))
    logger.info(f"Start to test stage: {opt.stage}, named: {opt.name}")

    # Validate dataset
    required_dirs = ['cloth', 'cloth-mask', 'image', 'image-parse-new', 'image-mask', 'openpose_json']
    data_path = Path(opt.dataroot)
    for d in required_dirs:
        if not (data_path / d).exists():
            raise FileNotFoundError(f"Required directory {data_path / d} does not exist")

    # Create dataset and dataloader
    test_dataset = CPDataset(opt)
    test_loader = CPDataLoader(opt, test_dataset)

    # Visualization
    tensorboard_dir = Path(opt.tensorboard_dir) / opt.name
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    board = SummaryWriter(logdir=str(tensorboard_dir))

    # Create model & test
    if opt.stage == 'GMM':
        model = GMM(opt)
        load_checkpoint(model, opt.checkpoint)
        with torch.no_grad():
            test_gmm(opt, test_loader, model, board)
    elif opt.stage == 'TOM':
        model = UnetGenerator(35, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d)
        gmm_model = GMM(opt)
        load_checkpoint(model, opt.checkpoint)
        load_checkpoint(gmm_model, opt.gmm_checkpoint)
        with torch.no_grad():
            test_tom(opt, test_loader, model, gmm_model, board)
    else:
        raise NotImplementedError(f'Model {opt.stage} is not implemented')

    logger.info(f'Finished test {opt.stage}, named: {opt.name}')

if __name__ == "__main__":
    main()
