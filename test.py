import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from PIL import ImageDraw
import os.path as osp
from pathlib import Path
import numpy as np
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CPDataset(data.Dataset):
    """Dataset for CP-VTON+."""
    def __init__(self, opt, datamode='train', data_list=None):
        super(CPDataset, self).__init__()
        self.opt = opt
        self.root = Path(opt.dataroot)
        self.datamode = datamode
        self.stage = opt.stage
        self.data_list = data_list if data_list else opt.data_list
        self.fine_height = opt.fine_height
        self.fine_width = opt.fine_width
        self.radius = opt.radius
        self.data_path = self.root

        # Validate required subdirectories based on datamode
        base_path = self.data_path / self.datamode
        required_dirs = [
            'cloth',
            'cloth-mask',
            'image',
            'image-parse-new',
            'image-mask',
            'openpose_json'
        ]
        for d in required_dirs:
            if not (base_path / d).exists():
                raise FileNotFoundError(f"Directory {base_path / d} does not exist")

        # Define transforms
        if self.datamode == 'train':
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

        # Load data list
        im_names, c_names = [], []
        data_list_path = self.data_path / self.data_list
        if not data_list_path.exists():
            raise FileNotFoundError(f"Data list {data_list_path} does not exist")
        with open(data_list_path, 'r') as f:
            for line in f.readlines():
                im_name, c_name = line.strip().split()
                c_name = c_name.replace('train/', '').replace('test/', '')  # Remove train/ or test/ prefix
                im_names.append(im_name)
                c_names.append(c_name)
        self.im_names = im_names
        self.c_names = c_names
        logger.info(f"Loaded {len(im_names)} pairs from {data_list_path}")

    def name(self):
        return "CPDataset"

    def __getitem__(self, index):
        c_name = self.c_names[index]
        im_name = self.im_names[index]

        # Adjust base path based on datamode
        base_path = self.data_path / self.datamode

        # Try loading cloth from test/cloth/ or train/cloth/
        cloth_path = base_path / 'cloth' / c_name
        cloth_alt_path = self.data_path / ('train' if self.datamode == 'test' else 'test') / 'cloth' / c_name
        try:
            c = Image.open(cloth_path if cloth_path.exists() else cloth_alt_path)
        except FileNotFoundError:
            logger.error(f"Cloth image not found: {cloth_path} or {cloth_alt_path}")
            raise FileNotFoundError(f"Cloth image {c_name} not found")
        
        # Try loading cloth mask from test/cloth-mask/ or train/cloth-mask/
        cm_name = c_name.replace('.jpg', '_mask.jpg')
        cm_path = base_path / 'cloth-mask' / cm_name
        cm_alt_path = self.data_path / ('train' if self.datamode == 'test' else 'test') / 'cloth-mask' / cm_name
        try:
            cm = Image.open(cm_path if cm_path.exists() else cm_alt_path).convert('L')
        except FileNotFoundError:
            logger.error(f"Cloth mask not found: {cm_path} or {cm_alt_path}")
            raise
        
        c = c.resize((self.fine_width, self.fine_height), Image.BILINEAR)
        cm = cm.resize((self.fine_width, self.fine_height), Image.BILINEAR)
        c = self.transform(c)
        cm_array = np.array(cm)
        cm_array = (cm_array >= 128).astype(np.float32)
        cm = torch.from_numpy(cm_array).unsqueeze(0)

        # Load person image
        try:
            im = Image.open(base_path / 'image' / im_name)
        except FileNotFoundError:
            logger.error(f"Person image not found: {base_path / 'image' / im_name}")
            raise
        im = im.resize((self.fine_width, self.fine_height), Image.BILINEAR)
        im = self.transform(im)

        # Parse images
        parse_name = im_name.replace('.jpg', '.png')
        try:
            im_parse = Image.open(base_path / 'image-parse-new' / parse_name).convert('L')
            im_mask = Image.open(base_path / 'image-mask' / parse_name).convert('L')
        except FileNotFoundError as e:
            logger.error(f"Parse or mask image not found: {e}")
            raise
        im_parse = im_parse.resize((self.fine_width, self.fine_height), Image.BILINEAR)
        im_mask = im_mask.resize((self.fine_width, self.fine_height), Image.BILINEAR)
        parse_array = np.array(im_parse)
        mask_array = np.array(im_mask)

        parse_shape = (mask_array > 0).astype(np.float32)
        if self.stage == 'GMM':
            parse_head = (parse_array == 1).astype(np.float32) + \
                        (parse_array == 4).astype(np.float32) + \
                        (parse_array == 13).astype(np.float32)
        else:
            parse_head = (parse_array == 1).astype(np.float32) + \
                        (parse_array == 2).astype(np.float32) + \
                        (parse_array == 4).astype(np.float32) + \
                        (parse_array == 9).astype(np.float32) + \
                        (parse_array == 12).astype(np.float32) + \
                        (parse_array == 13).astype(np.float32) + \
                        (parse_array == 16).astype(np.float32) + \
                        (parse_array == 17).astype(np.float32)
        parse_cloth = (parse_array == 5).astype(np.float32) + \
                     (parse_array == 6).astype(np.float32) + \
                     (parse_array == 7).astype(np.float32)

        parse_shape_ori = Image.fromarray((parse_shape * 255).astype(np.uint8))
        parse_shape = parse_shape_ori.resize((self.fine_width // 16, self.fine_height // 16), Image.BILINEAR)
        parse_shape = parse_shape.resize((self.fine_width, self.fine_height), Image.BILINEAR)
        parse_shape_ori = parse_shape_ori.resize((self.fine_width, self.fine_height), Image.BILINEAR)

        shape_ori = self.transform(parse_shape_ori.convert('RGB'))
        shape = self.transform(parse_shape.convert('RGB'))
        phead = torch.from_numpy(parse_head)
        pcm = torch.from_numpy(parse_cloth)

        im_c = im * pcm + (1 - pcm)
        im_h = im * phead - (1 - phead)

        # Load pose points
        pose_name = im_name.replace('.jpg', '_keypoints.json')
        try:
            with open(base_path / 'openpose_json' / pose_name, 'r') as f:
                pose_label = json.load(f)
                pose_data = np.array(pose_label['people'][0]['pose_keypoints_2d']).reshape((-1, 3))
        except FileNotFoundError:
            logger.error(f"Pose file not found: {base_path / 'openpose_json' / pose_name}")
            raise

        point_num = pose_data.shape[0]
        pose_map = torch.zeros(point_num, self.fine_height, self.fine_width)
        r = self.radius
        im_pose = Image.new('L', (self.fine_width, self.fine_height))
        pose_draw = ImageDraw.Draw(im_pose)
        for i in range(point_num):
            one_map = Image.new('L', (self.fine_width, self.fine_height))
            draw = ImageDraw.Draw(one_map)
            pointx, pointy = pose_data[i, 0], pose_data[i, 1]
            if pointx > 1 and pointy > 1:
                draw.rectangle((pointx - r, pointy - r, pointx + r, pointy + r), 'white', 'white')
                pose_draw.rectangle((pointx - r, pointy - r, pointx + r, pointy + r), 'white', 'white')
            one_map = self.transform(one_map.convert('RGB'))
            pose_map[i] = one_map[0]

        im_pose = im_pose.resize((self.fine_width, self.fine_height), Image.BILINEAR)
        im_pose = self.transform(im_pose.convert('RGB'))

        agnostic = torch.cat([shape, im_h, pose_map], 0)

        if self.stage == 'GMM':
            im_g = Image.open('grid.png')
            im_g = im_g.resize((self.fine_width, self.fine_height), Image.BILINEAR)
            im_g = self.transform(im_g)
        else:
            im_g = ''

        pcm.unsqueeze_(0)

        result = {
            'c_name': c_name,
            'im_name': im_name,
            'cloth': c,
            'cloth_mask': cm,
            'image': im,
            'agnostic': agnostic,
            'parse_cloth': im_c,
            'shape': shape,
            'head': im_h,
            'pose_image': im_pose,
            'grid_image': im_g,
            'parse_cloth_mask': pcm,
            'shape_ori': shape_ori,
        }
        return result

    def __len__(self):
        return len(self.im_names)

class CPDataLoader(object):
    def __init__(self, opt, dataset):
        super(CPDataLoader, self).__init__()
        if opt.shuffle:
            train_sampler = torch.utils.data.sampler.RandomSampler(dataset)
        else:
            train_sampler = None
        self.data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
            num_workers=opt.workers, pin_memory=True, sampler=train_sampler)
        self.dataset = dataset
        self.data_iter = self.data_loader.__iter__()

    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()
        return batch

if __name__ == "__main__":
    print("Check the dataset for geometric matching module!")
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", default="data")
    parser.add_argument("--datamode", default="train")
    parser.add_argument("--stage", default="GMM")
    parser.add_argument("--data_list", default="train_pairs.txt")
    parser.add_argument("--val_data_list", default="val_pairs.txt")
    parser.add_argument("--fine_width", type=int, default=192)
    parser.add_argument("--fine_height", type=int, default=256)
    parser.add_argument("--radius", type=int, default=3)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')
    parser.add_argument('-b', '--batch-size', type=int, default=2)
    parser.add_argument('-j', '--workers', type=int, default=1)
    opt = parser.parse_args()
    dataset = CPDataset(opt)
    data_loader = CPDataLoader(opt, dataset)
    print('Size of the dataset: %05d, dataloader: %04d' % (len(dataset), len(data_loader)))
    first_item = dataset.__getitem__(0)
    first_batch = data_loader.next_batch()
    from IPython import embed
    embed()
