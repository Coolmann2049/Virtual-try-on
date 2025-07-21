import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image
from PIL import ImageDraw

import os.path as osp
import numpy as np
import json


class CPDataset(data.Dataset):
    """Dataset for CP-VTON+.
    """

    def __init__(self, opt, datamode='train', data_list=None):
        super(CPDataset, self).__init__()
        # Base settings
        self.opt = opt
        self.root = opt.dataroot
        self.datamode = datamode  # Explicitly set via argument
        self.stage = opt.stage  # 'GMM' or 'TOM'
        self.data_list = data_list if data_list else opt.data_list  # Use provided data_list or default
        self.fine_height = opt.fine_height
        self.fine_width = opt.fine_width
        self.radius = opt.radius
        self.data_path = osp.join(opt.dataroot, self.datamode if osp.exists(osp.join(opt.dataroot, datamode)) else 'train')

        # Define transforms with augmentation, disabled for validation
        if self.datamode == 'train':
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:  # 'val' mode
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

        # Load data list based on mode
        im_names = []
        c_names = []
        with open(osp.join(opt.dataroot, self.data_list), 'r') as f:
            for line in f.readlines():
                im_name, c_name = line.strip().split()
                # Remove 'train/' prefix from c_name if present
                c_name = c_name.replace('train/', '')
                im_names.append(im_name)
                c_names.append(c_name)

        self.im_names = im_names
        self.c_names = c_names

    def name(self):
        return "CPDataset"

    def __getitem__(self, index):
        c_name = self.c_names[index]
        im_name = self.im_names[index]

        # Load and resize cloth and cloth mask
        c = Image.open(osp.join(self.data_path, 'cloth', c_name))
        cm = Image.open(osp.join(self.data_path, 'cloth-mask', c_name)).convert('L')
        c = c.resize((self.fine_width, self.fine_height), Image.BILINEAR)
        cm = cm.resize((self.fine_width, self.fine_height), Image.BILINEAR)

        c = self.transform(c)  # [-1,1] with augmentation for train, none for val
        cm_array = np.array(cm)
        cm_array = (cm_array >= 128).astype(np.float32)
        cm = torch.from_numpy(cm_array)  # [0,1]
        cm.unsqueeze_(0)

        # Load and resize person image
        im = Image.open(osp.join(self.data_path, 'image', im_name))
        im = im.resize((self.fine_width, self.fine_height), Image.BILINEAR)
        im = self.transform(im)  # [-1,1] with augmentation for train, none for val

        # Load and resize parsing image
        parse_name = im_name.replace('.jpg', '.png')
        im_parse = Image.open(osp.join(self.data_path, 'image-parse-new', parse_name)).convert('L')
        im_parse = im_parse.resize((self.fine_width, self.fine_height), Image.BILINEAR)
        parse_array = np.array(im_parse)

        im_mask = Image.open(osp.join(self.data_path, 'image-mask', parse_name)).convert('L')
        im_mask = im_mask.resize((self.fine_width, self.fine_height), Image.BILINEAR)
        mask_array = np.array(im_mask)

        # Parse shape from body mask (CP-VTON+)
        parse_shape = (mask_array > 0).astype(np.float32)

        if self.stage == 'GMM':
            parse_head = (parse_array == 1).astype(np.float32) + \
                         (parse_array == 4).astype(np.float32) + \
                         (parse_array == 13).astype(np.float32)  # CP-VTON+ GMM input
        else:  # For TOM stage
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
                      (parse_array == 7).astype(np.float32)  # upper-clothes labels

        # Resize and transform shape
        parse_shape_ori = Image.fromarray((parse_shape * 255).astype(np.uint8))
        parse_shape = parse_shape_ori.resize((self.fine_width//16, self.fine_height//16), Image.BILINEAR)
        parse_shape = parse_shape.resize((self.fine_width, self.fine_height), Image.BILINEAR)
        parse_shape_ori = parse_shape_ori.resize((self.fine_width, self.fine_height), Image.BILINEAR)

        shape_ori = self.transform(parse_shape_ori.convert('RGB'))  # [-1,1]
        shape = self.transform(parse_shape.convert('RGB'))  # [-1,1]
        phead = torch.from_numpy(parse_head)  # [0,1]
        pcm = torch.from_numpy(parse_cloth)  # [0,1]

        # Upper cloth and head masks
        im_c = im * pcm + (1 - pcm)  # [-1,1], fill 1 for other parts
        im_h = im * phead - (1 - phead)  # [-1,1], fill -1 for other parts

        # Load and resize pose points
        pose_name = im_name.replace('.jpg', '_keypoints.json')
        with open(osp.join(self.data_path, 'openpose_json', pose_name), 'r') as f:
            pose_label = json.load(f)
            pose_data = np.array(pose_label['people'][0]['pose_keypoints_2d'])
            pose_data = pose_data.reshape((-1, 3))

        point_num = pose_data.shape[0]
        pose_map = torch.zeros(point_num, self.fine_height, self.fine_width)
        r = self.radius
        im_pose = Image.new('L', (self.fine_width, self.fine_height))
        pose_draw = ImageDraw.Draw(im_pose)
        for i in range(point_num):
            one_map = Image.new('L', (self.fine_width, self.fine_height))
            draw = ImageDraw.Draw(one_map)
            pointx = pose_data[i, 0]
            pointy = pose_data[i, 1]
            if pointx > 1 and pointy > 1:
                draw.rectangle((pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')
                pose_draw.rectangle((pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')
            one_map = self.transform(one_map.convert('RGB'))
            pose_map[i] = one_map[0]

        # Resize and transform pose image
        im_pose = im_pose.resize((self.fine_width, self.fine_height), Image.BILINEAR)
        im_pose = self.transform(im_pose.convert('RGB'))  # [-1,1] with augmentation for train, none for val

        # Cloth-agnostic representation
        agnostic = torch.cat([shape, im_h, pose_map], 0)

        # Grid image for GMM visualization
        if self.stage == 'GMM':
            im_g = Image.open('grid.png')
            im_g = im_g.resize((self.fine_width, self.fine_height), Image.BILINEAR)
            im_g = self.transform(im_g)
        else:
            im_g = ''  # Not used as input for TOM

        pcm.unsqueeze_(0)  # CP-VTON+

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
