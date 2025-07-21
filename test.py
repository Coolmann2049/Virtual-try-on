import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import argparse
from networks import GMM, UnetGenerator
from cp_dataset import CPDataset, CPDataLoader
from tensorboardX import SummaryWriter
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="GMM")
    parser.add_argument("--gpu_ids", default="")
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--dataroot', default="data")
    parser.add_argument('--datamode', default="train")
    parser.add_argument('--stage', default="GMM")
    parser.add_argument('--data_list', default="train_pairs.txt")
    parser.add_argument('--fine_width', type=int, default=192)
    parser.add_argument('--fine_height', type=int, default=256)
    parser.add_argument('--radius', type=int, default=5)
    parser.add_argument('--grid_size', type=int, default=5)
    parser.add_argument('--tensorboard_dir', default='tensorboard')
    parser.add_argument('--result_dir', default='result')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--gmm_checkpoint', default='')
    parser.add_argument('--display_count', type=int, default=1)
    parser.add_argument('--shuffle', action='store_true')
    opt = parser.parse_args()
    return opt

def save_image(tensor, path, nrow=1):
    from torchvision.utils import save_image
    try:
        save_image(tensor, path, nrow=nrow, normalize=True)
    except Exception as e:
        logger.error(f"Error saving image {path}: {e}")

def test_gmm(opt, test_loader, model, board):
    model.eval()
    model.cuda()
    base_name = f"{opt.name}_GMM"
    save_dir = Path(opt.result_dir) / base_name / opt.datamode / 'warp-cloth'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for step, inputs in enumerate(test_loader.data_loader):
        try:
            im = inputs['image'].cuda()
            c = inputs['cloth'].cuda()
            im_name = inputs['im_name']
            c_name = inputs['c_name']
            
            with torch.no_grad():
                grid, theta = model(c, im)
                warped_cloth = F.grid_sample(c, grid, padding_mode='border')
            
            for i in range(im.size(0)):
                save_path = save_dir / f"{im_name[i]}_{c_name[i]}"
                save_image(warped_cloth[i], save_path)
                board.add_image(f'warp-cloth/{im_name[i]}', warped_cloth[i], step * opt.batch_size + i)
                
        except Exception as e:
            logger.error(f"Error processing batch {step}: {e}")
            continue

def test_tom(opt, test_loader, model, gmm_model, board):
    model.eval()
    gmm_model.eval()
    model.cuda()
    gmm_model.cuda()
    base_name = f"{opt.name}"
    save_dir = Path(opt.result_dir) / base_name / opt.datamode / 'try-on'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for step, inputs in enumerate(test_loader.data_loader):
        try:
            im = inputs['image'].cuda()
            c = inputs['cloth'].cuda()
            cm = inputs['cloth_mask'].cuda()
            agnostic = inputs['agnostic'].cuda()
            grid_image = inputs['grid_image'].cuda() if opt.stage == 'GMM' else None
            im_name = inputs['im_name']
            c_name = inputs['c_name']
            
            with torch.no_grad():
                if opt.stage == 'GMM':
                    grid, theta = gmm_model(c, im)
                    warped_cloth = F.grid_sample(c, grid, padding_mode='border')
                    output = warped_cloth
                else:
                    grid, theta = gmm_model(c, agnostic)
                    warped_cloth = F.grid_sample(c, grid, padding_mode='border')
                    warped_mask = F.grid_sample(cm, grid, padding_mode='zeros')
                    output = model(agnostic, warped_cloth, warped_mask)
            
            for i in range(im.size(0)):
                save_path = save_dir / f"{im_name[i]}_{c_name[i]}"
                save_image(output[i], save_path)
                board.add_image(f'try-on/{im_name[i]}', output[i], step * opt.batch_size + i)
                
        except Exception as e:
            logger.error(f"Error processing batch {step}: {e}")
            continue

def main():
    opt = get_opt()
    
    # Validate required subdirectories based on datamode
    data_path = Path(opt.dataroot)
    required_dirs = [
        f'{opt.datamode}/cloth',
        f'{opt.datamode}/cloth-mask',
        f'{opt.datamode}/image',
        f'{opt.datamode}/image-parse-new',
        'image-mask',
        f'{opt.datamode}/openpose_json'
    ]
    for d in required_dirs:
        if not (data_path / d).exists():
            raise FileNotFoundError(f"Required directory {data_path / d} does not exist")
    
    # Initialize dataset and dataloader
    test_dataset = CPDataset(opt, datamode=opt.datamode)
    test_loader = CPDataLoader(opt, test_dataset)
    logger.info(f"Dataset size: {len(test_dataset):05d}")
    
    # Initialize models
    if opt.stage == 'GMM':
        model = GMM(opt)
        model.load_state_dict(torch.load(opt.gmm_checkpoint, map_location='cpu'), strict=False)
        logger.info(f"Loaded GMM checkpoint {opt.gmm_checkpoint}")
        board = SummaryWriter(log_dir=Path(opt.tensorboard_dir) / opt.name)
        test_gmm(opt, test_loader, model, board)
    else:
        model = UnetGenerator(25, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d)
        model.load_state_dict(torch.load(opt.checkpoint, map_location='cpu'), strict=False)
        gmm_model = GMM(opt)
        gmm_model.load_state_dict(torch.load(opt.gmm_checkpoint, map_location='cpu'), strict=False)
        logger.info(f"Loaded TOM checkpoint {opt.checkpoint} and GMM checkpoint {opt.gmm_checkpoint}")
        board = SummaryWriter(log_dir=Path(opt.tensorboard_dir) / opt.name)
        test_tom(opt, test_loader, model, gmm_model, board)
    
    board.close()

if __name__ == "__main__":
    main()
