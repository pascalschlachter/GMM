import torch
import torchvision
import torchvision.transforms as T
import lightning as L

from networks import SourceModule


def train_transform(resize_size=256, crop_size=224):
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    return T.Compose([
        T.Resize((resize_size, resize_size)),
        T.RandomCrop(crop_size),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize
    ])


def test_transform(resize_size=256, crop_size=224):
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return T.Compose([
        T.Resize((resize_size, resize_size)),
        T.CenterCrop(crop_size),
        T.ToTensor(),
        normalize
    ])


class SFUniDADataModuleBase(L.LightningDataModule):
    def __init__(self, batch_size, data_dir, category_shift, train_domain, test_domain, shared_class_num,
                 source_private_class_num, target_private_class_num):
        super(SFUniDADataModuleBase, self).__init__()
        self.batch_size = batch_size
        self.train_domain = train_domain
        self.test_domain = test_domain
        self.category_shift = category_shift

        self.train_set = None
        self.test_set = None

        self.data_dir = data_dir

        self.shared_class_num = shared_class_num
        self.source_private_class_num = source_private_class_num
        self.target_private_class_num = target_private_class_num
        self.total_class_num = shared_class_num + source_private_class_num + target_private_class_num

        self.shared_classes = [i for i in range(shared_class_num)]
        self.source_private_classes = [i + shared_class_num for i in range(source_private_class_num)]
        self.target_private_classes = [self.total_class_num - 1 - i for i in range(target_private_class_num)]

        self.source_classes = self.shared_classes + self.source_private_classes
        self.target_classes = self.shared_classes + self.target_private_classes

    def setup(self, stage):
        self.train_set = torchvision.datasets.ImageFolder(root=self.data_dir+self.train_domain,
                                                          transform=train_transform())
        self.test_set = torchvision.datasets.ImageFolder(root=self.data_dir+self.test_domain,
                                                         transform=test_transform())

        train_indices = [idx for idx, target in enumerate(self.train_set.targets) if target in self.source_classes]
        self.train_set = torch.utils.data.Subset(self.train_set, train_indices)

        test_indices = [idx for idx, target in enumerate(self.test_set.targets) if target in self.target_classes]
        self.test_set = torch.utils.data.Subset(self.test_set, test_indices)

    def train_dataloader(self):
        if isinstance(self.trainer.lightning_module, SourceModule):
            return torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, drop_last=True,
                                               num_workers=8)
        else:
            return torch.utils.data.DataLoader(self.test_set, batch_size=self.batch_size, shuffle=True, drop_last=True,
                                               num_workers=8)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=8)


class DomainNetDataModule(SFUniDADataModuleBase):
    def __init__(self, batch_size, category_shift='', train_domain='painting', test_domain='real'):
        data_dir = 'data/domainnet/'

        if category_shift == 'PDA':
            self.shared_class_num = 200
            self.source_private_class_num = 145
            self.target_private_class_num = 0
        elif category_shift == 'ODA':
            self.shared_class_num = 200
            self.source_private_class_num = 0
            self.target_private_class_num = 145
        elif category_shift == 'OPDA':
            self.shared_class_num = 150
            self.source_private_class_num = 50
            self.target_private_class_num = 145
        else:
            self.shared_class_num = 345
            self.source_private_class_num = 0
            self.target_private_class_num = 0

        super(DomainNetDataModule, self).__init__(batch_size, data_dir, category_shift, train_domain,
                                                  test_domain, self.shared_class_num, self.source_private_class_num,
                                                  self.target_private_class_num)


class VisDADataModule(SFUniDADataModuleBase):
    def __init__(self, batch_size, category_shift='', train_domain='train', test_domain='validation'):
        data_dir = 'data/visda/'

        train_domain = 'train'
        test_domain = 'validation'

        if category_shift == 'PDA':
            self.shared_class_num = 6
            self.source_private_class_num = 6
            self.target_private_class_num = 0
        elif category_shift == 'ODA':
            self.shared_class_num = 6
            self.source_private_class_num = 0
            self.target_private_class_num = 6
        elif category_shift == 'OPDA':
            self.shared_class_num = 6
            self.source_private_class_num = 3
            self.target_private_class_num = 3
        else:
            self.shared_class_num = 12
            self.source_private_class_num = 0
            self.target_private_class_num = 0

        super(VisDADataModule, self).__init__(batch_size, data_dir, category_shift, train_domain,
                                              test_domain, self.shared_class_num, self.source_private_class_num,
                                              self.target_private_class_num)
