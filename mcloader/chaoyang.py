import os
import json
from PIL import Image

from mcloader.base import BaseDataset
from mcloader.io import pathmgr


class Chaoyang(BaseDataset):

    def __init__(self, data_path, split,transform):
        super(Chaoyang, self).__init__(split, transform)
        assert pathmgr.exists(data_path), "Data path '{}' not found".format(data_path)
        splits = ["train", "test"]
        assert split in splits, "Split '{}' not supported for Chaoyang".format(split)
        self.data_path = data_path
        with open(os.path.join(data_path, f'{split}.json'), 'r') as f:
            anns = json.load(f)
        self.data = anns
        self.nb_classes=4

    def __len__(self):
        return len(self.data)

    def _get_data(self, index):
        ann = self.data[index]
        img = Image.open(os.path.join(self.data_path, ann['name']))
        img = img.convert('RGB')
        return img, ann['label']
