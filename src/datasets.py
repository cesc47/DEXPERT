import os
import torch
import json
import presets
import numpy as np

from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision.transforms.functional import InterpolationMode


class DATES(Dataset):
    """Date estimation in the wild dataset. Used to train the specialist or general models."""

    def __init__(self, args, split='train'):
        # initialize the dataset variables
        self.data_path = args.data_path
        self.balanced = args.balanced

        # if the dataset is balanced, load the balanced dataset
        if self.balanced:
            self.root_dir = os.path.join(args.data_path, 'Date_Estimation_in_the_Wild_Balanced')
        else:
            self.root_dir = os.path.join(args.data_path, 'Date_Estimation_in_the_Wild')

        self.split = split
        self.specialist = args.specialist
        self.len_dew = 0

        # load the annotations depending on the split and if the dataset is balanced or not
        if self.split == 'train' and self.balanced:
            self.filenames, self.labels, self.bboxes, self.scores = self.load_annotations(
                os.path.join(self.root_dir, f'gt_{split}.csv'))
        else:
            self.filenames, self.labels, self.bboxes, self.scores = self.load_annotations(os.path.join(self.data_path, 'Date_Estimation_in_the_Wild', f'gt_{split}_ok.csv'))

        # check if the task is regression or classification and set the number of classes in function of that
        self.regression = args.regression
        self.classes = 1 if self.regression else 14

        # set the transformations depending on the split
        if self.split == 'train':
            self.transforms = presets.ClassificationPresetTrain(
                crop_size=args.train_crop_size,
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                interpolation=InterpolationMode(args.interpolation),
                auto_augment_policy=getattr(args, "auto_augment", None),
                random_erase_prob=getattr(args, "random_erase", 0.0),
                ra_magnitude=getattr(args, "ra_magnitude", None),
                augmix_severity=getattr(args, "augmix_severity", None),
            )
        else:
            self.transforms = presets.ClassificationPresetEval(
                crop_size=args.val_crop_size,
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                resize_size=args.val_resize_size,
                interpolation=InterpolationMode(args.interpolation)
            )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # load the image depending on the split and if the dataset is balanced or not
        if self.split == 'train' and self.balanced:
            img_name = os.path.join(self.root_dir, str(self.labels[idx]), self.filenames[idx][0],
                                    self.filenames[idx][1:3], f'{self.filenames[idx]}.jpg')
        else:
            img_name = os.path.join(self.data_path, 'Date_Estimation_in_the_Wild', 'images', self.filenames[idx][0],
                                    self.filenames[idx][1:3], f'{self.filenames[idx]}.jpg')

        image = Image.open(img_name)
        image = image.convert('RGB')

        image = self.transforms(image)

        # return a label between (1930,1999) if regression, else return a label between 0 and 13 (half decades)
        if self.regression:
            labels = np.expand_dims(int(self.labels[idx]), axis=0)
        else:
            labels = np.array(int(self.labels[idx] / 5) - int(1930/5))

        return np.array(image), labels

    def load_annotations(self, path):
        """
        Load the annotations from the csv file.
        """
        with open(path, 'r') as f:
            lines = f.readlines()
        lines = [line[:-1] for line in lines]

        filenames = [line.split(',')[1] for line in lines]
        labels = [int(line.split(',')[0]) for line in lines]
        bboxes = None
        scores = None

        # use the detections from DETR if we are training an specialist
        if self.specialist is not None:
            # map coco labels to idxs
            if self.specialist == 'person':
                specialist = 1
            elif self.specialist == 'train':
                specialist = 7
            elif self.specialist == 'car':
                specialist = 3
            elif self.specialist == 'boat':
                specialist = 9
            elif self.specialist == 'bus':
                specialist = 6
            elif self.specialist == 'airplane':
                specialist = 5
            else:
                raise ValueError('Specialist not found')

            # for each filename, load the json file and check if the specialist is in the image
            filenames_specialist = []
            bbox_specialist = []
            scores_specialist = []
            print('getting filenames of specialist...')

            if self.split == 'train' and self.balanced:
                path = 'results_object_detection_detr_Balanced'
            else:
                path = 'results_object_detection_detr'

            for i, filename in tqdm(enumerate(filenames)):
                # read json file and check if specialist is in the image
                if self.split == 'train' and self.balanced:
                    path_detection = os.path.join(path, str(labels[i]), filename[0], filename[1:3], f'{filename}.json')
                else:
                    path_detection = os.path.join(path, filename[0], filename[1:3], f'{filename}.json')
                with open(path_detection) as f:
                    try:
                        data = json.load(f)
                    except:
                        print('file corrupted: ' + filename + '.json')
                        continue

                for j, label in enumerate(data['labels']):
                    # add the filename to the list if the data['boxes'] area > than 10000. format is [x1, y1, x2, y2]
                    if label == specialist and (data['boxes'][j][2] - data['boxes'][j][0]) * (data['boxes'][j][3] - data['boxes'][j][1]) > 10000:
                        filenames_specialist.append(filename)
                        bbox_specialist.append(data['boxes'][j])
                        scores_specialist.append(data['scores'][j])

            # for filename_specialist, find in the position of the filename in the original list. Then, add the label
            print('getting labels of specialist...')

            # Create a dictionary to store the filenames as keys and corresponding labels as values
            filename_labels = dict(zip(filenames, labels))

            # Use list comprehension to get the labels for filenames in filenames_specialist
            labels_specialist = [filename_labels[filename] for filename in filenames_specialist]

            filenames = filenames_specialist
            self.len_dew = len(filenames_specialist)

            labels = labels_specialist
            bboxes = bbox_specialist
            scores = scores_specialist

        return filenames, labels, bboxes, scores


class DATES_ALLINONE(Dataset):
    """Date estimation in the wild dataset. Used to train the DEXPERT."""

    def __init__(self, args, split='train'):
        self.root_dir = os.path.join(args.data_path, 'Date_Estimation_in_the_Wild')
        #self.root_dir_balanced = os.path.join(args.data_path, 'Date_Estimation_in_the_Wild_Balanced')
        self.split = split
        #if self.split == 'train':
        #    self.filenames, self.labels = self.load_annotations(os.path.join(self.root_dir_balanced, f'gt_{split}.csv'))
        #else:
        #    self.filenames, self.labels = self.load_annotations(os.path.join(self.root_dir, f'gt_{split}_ok.csv'))
        self.filenames, self.labels = self.load_annotations(os.path.join(self.root_dir, f'gt_{split}_ok.csv'))
        self.classes = 14

        if self.split == 'train':
            self.transforms = presets.ClassificationPresetTrain(
                crop_size=args.train_crop_size,
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                interpolation=InterpolationMode(args.interpolation),
                auto_augment_policy=getattr(args, "auto_augment", None),
                random_erase_prob=getattr(args, "random_erase", 0.0),
                ra_magnitude=getattr(args, "ra_magnitude", None),
                augmix_severity=getattr(args, "augmix_severity", None),
            )
        else:
            self.transforms = presets.ClassificationPresetEval(
                crop_size=args.val_crop_size,
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                resize_size=args.val_resize_size,
                interpolation=InterpolationMode(args.interpolation)
            )

        self.args = args
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, 'images', self.filenames[idx][0], self.filenames[idx][1:3],
                                f'{self.filenames[idx]}.jpg')
        image = Image.open(img_name).convert('RGB')

        # todo: generalize for when there are more than 1 detections per specialist
        images = {
            'general': [image],
            'person': [],
            'car': [],
            'boat': [],
            'bus': [],
            'airplane': [],
            'train': []
        }

        # read .json of object detection
        json_filename = os.path.join(self.root_dir, 'results_object_detection_detr', self.filenames[idx][0],
                                     self.filenames[idx][1:3], f'{self.filenames[idx]}.json')
        with open(json_filename) as f:
            try:
                data = json.load(f)
                for j, index in enumerate(data['labels']):
                    box_area = (data['boxes'][j][2] - data['boxes'][j][0]) * (data['boxes'][j][3] - data['boxes'][j][1])
                    if box_area > 10000:
                        label = self.idx2label_coco(index)
                        if label is not None:
                            images[label].append(image.crop(
                                (data['boxes'][j][0], data['boxes'][j][1], data['boxes'][j][2], data['boxes'][j][3])))

            except:
                print('file corrupted: ' + json_filename)

        # transform images
        for label in images.keys():
            images[label] = [self.transforms(img) for img in images[label]]

        # Create a list of tuples containing the images and their corresponding labels
        samples = [(img, label) for label, images_list in images.items() for img in images_list]

        return samples, np.array(int(self.labels[idx] / 5) - int(1930 / 5))

    def collate_fn(self, batch):
        """
        Custom collate_fn for the dataloader.
        """
        # get the maximum number of images in a batch
        max_len = max([len(sample[0]) for sample in batch])
        # generate tuple to stack: a torch tensor with all zeros and None
        tuple_to_stack = (torch.zeros((3, 224, 224)), 'padded')
        # for each element of the batch, add the tuple_to_stack until the length of the batch is max_len
        for i in range(len(batch)):
            while len(batch[i][0]) < max_len:
                batch[i][0].append(tuple_to_stack)

        # stack the images (batch, len_patches, rgb, h, w) and labels
        images = torch.stack([torch.stack([img[0] for img in sample[0]], dim=0) for sample in batch], dim=0)
        labels = np.array([int(sample[1]) for sample in batch])
        labels = torch.from_numpy(labels)
        # generate a np array with the labels of the patches. of shape (batch, len_patches)
        labels_patches = np.array([[label[1] for label in sample[0]] for sample in batch])

        images = (images, labels_patches)
        return images, labels

    def load_annotations(self, path):
        """
        Load the annotations from the csv file.
        """
        with open(path, 'r') as f:
            lines = f.readlines()
        lines = [line[:-1] for line in lines]

        filenames = [line.split(',')[1] for line in lines]
        labels = [int(line.split(',')[0]) for line in lines]

        return filenames, labels

    @staticmethod
    def idx2label_coco(idx):
        """
        Convert the idxs to the corresponding labels.
        """
        if idx == 1:
            return 'person'
        elif idx == 3:
            return 'car'
        elif idx == 5:
            return 'airplane'
        elif idx == 6:
            return 'bus'
        elif idx == 7:
            return 'train'
        elif idx == 9:
            return 'boat'
        else:
            return None


class DATES_ALLINONE_TEST(Dataset):
    """Date estimation in the wild dataset. Used to test the DEXPERT"""

    def __init__(self, args, split='train'):
        self.root_dir = os.path.join(args.data_path, 'Date_Estimation_in_the_Wild')
        # self.root_dir_balanced = os.path.join(args.data_path, 'Date_Estimation_in_the_Wild_Balanced')
        self.split = split
        # if self.split == 'train':
        #    self.filenames, self.labels = self.load_annotations(os.path.join(self.root_dir_balanced, f'gt_{split}.csv'))
        # else:
        #    self.filenames, self.labels = self.load_annotations(os.path.join(self.root_dir, f'gt_{split}_ok.csv'))
        self.filenames, self.labels = self.load_annotations(os.path.join(self.root_dir, f'gt_{split}_ok.csv'))
        self.classes = 14
        self.args = args

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, 'images', self.filenames[idx][0], self.filenames[idx][1:3],
                                f'{self.filenames[idx]}.jpg')
        image = Image.open(img_name).convert('RGB')

        # todo: generalize for when there are more than 1 detections per specialist
        images = {
            'general': [image],
            'person': [],
            'car': [],
            'boat': [],
            'bus': [],
            'airplane': [],
            'train': []
        }

        # read .json of object detection
        json_filename = os.path.join(self.root_dir, 'results_object_detection_detr', self.filenames[idx][0],
                                     self.filenames[idx][1:3], f'{self.filenames[idx]}.json')
        with open(json_filename) as f:
            try:
                data = json.load(f)
                for j, index in enumerate(data['labels']):
                    box_area = (data['boxes'][j][2] - data['boxes'][j][0]) * (data['boxes'][j][3] - data['boxes'][j][1])
                    if box_area > 10000:
                        label = self.idx2label_coco(index)
                        if label is not None:
                            images[label].append(image.crop(
                                (data['boxes'][j][0], data['boxes'][j][1], data['boxes'][j][2], data['boxes'][j][3])))
            except:
                print('file corrupted: ' + json_filename)

        # transform images
        # for label in images.keys():
        #    images[label] = [self.transforms(img) for img in images[label]]

        # Create a list of tuples containing the images and their corresponding labels
        samples = [(img, label) for label, images_list in images.items() for img in images_list]

        return samples

    def collate_fn(self, batch):
        """
        Custom collate_fn for the dataloader.
        """
        # todo: fix for when a sample[0] is None
        # get the maximum number of images in a batch
        max_len = max([len(sample[0]) for sample in batch])
        # generate tuple to stack: a torch tensor with all zeros and None
        tuple_to_stack = (torch.zeros((3, 224, 224)), 'padded')
        # for each element of the batch, add the tuple_to_stack until the length of the batch is max_len
        for i in range(len(batch)):
            while len(batch[i][0]) < max_len:
                batch[i][0].append(tuple_to_stack)

        # stack the images (batch, len_patches, rgb, h, w) and labels
        images = torch.stack([torch.stack([img[0] for img in sample[0]], dim=0) for sample in batch], dim=0)
        labels = np.array([int(sample[1]) for sample in batch])
        labels = torch.from_numpy(labels)
        # generate a np array with the labels of the patches. of shape (batch, len_patches)
        labels_patches = np.array([[label[1] for label in sample[0]] for sample in batch])

        images = (images, labels_patches)
        return images, labels

    def load_annotations(self, path):
        """
        Load the annotations from the csv file.
        """
        with open(path, 'r') as f:
            lines = f.readlines()
        lines = [line[:-1] for line in lines]

        filenames = [line.split(',')[1] for line in lines]
        labels = [int(line.split(',')[0]) for line in lines]

        return filenames, labels

    @staticmethod
    def idx2label_coco(idx):
        """
        Convert the idxs to the corresponding labels.
        """
        if idx == 1:
            return 'person'
        elif idx == 3:
            return 'car'
        elif idx == 5:
            return 'airplane'
        elif idx == 6:
            return 'bus'
        elif idx == 7:
            return 'train'
        elif idx == 9:
            return 'boat'
        else:
            return None


