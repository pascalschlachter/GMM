import lightning as L
import torch
import torch.nn as nn
from torchmetrics import Accuracy, StatScores
from scipy.stats import entropy
from torchvision.models import resnet50
import math
import os
from utils import HScore, CrossEntropyLabelSmooth, CustomLRScheduler


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


class ResNetBackbone(nn.Module):
    def __init__(self):
        super(ResNetBackbone, self).__init__()
        model_resnet = resnet50(pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.output_dim = model_resnet.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, feature_dim=256, type='ori'):
        super(FeatureExtractor, self).__init__()
        self.feature_dim = feature_dim
        self.bn = nn.BatchNorm1d(feature_dim, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(input_dim, feature_dim)
        self.bottleneck.apply(init_weights)
        self.type = type

    def forward(self, x):
        x = self.bottleneck(x)
        if self.type == 'bn':
            x = self.bn(x)
        return x


class Classifier(nn.Module):
    def __init__(self, feature_dim, class_num, type='linear'):
        super(Classifier, self).__init__()

        self.type = type
        if type == 'wn':
            self.fc = nn.utils.weight_norm(nn.Linear(feature_dim, class_num), name='weight')
            self.fc.apply(init_weights)
        else:
            self.fc = nn.Linear(feature_dim, class_num)
            self.fc.apply(init_weights)

    def forward(self, x):
        x = self.fc(x)
        return x


class BaseModule(L.LightningModule):
    def __init__(self, datamodule, feature_dim, lr, ckpt_dir):
        super(BaseModule, self).__init__()

        self.known_classes_num = datamodule.shared_class_num + datamodule.source_private_class_num

        self.backbone = ResNetBackbone()
        self.feature_extractor = FeatureExtractor(self.backbone.output_dim, feature_dim, type='bn')
        self.classifier = Classifier(feature_dim, self.known_classes_num, type='wn')

        if ckpt_dir != '':
            checkpoint = torch.load(ckpt_dir, map_location=torch.device('cpu'))
            self.backbone.load_state_dict(checkpoint['backbone_state_dict'])
            self.feature_extractor.load_state_dict(checkpoint['feature_extractor_state_dict'])
            self.classifier.load_state_dict(checkpoint['classifier_state_dict'])

        self.lr = lr

        if datamodule.category_shift == 'OPDA' or datamodule.category_shift == 'ODA':
            self.open_flag = True
        else:
            self.open_flag = False

    def forward(self, x, apply_softmax=True):
        x = self.backbone(x)
        feature_embed = self.feature_extractor(x)
        x = self.classifier(feature_embed)
        if apply_softmax:
            x = nn.Softmax(dim=1)(x)
        return x, feature_embed


class SourceModule(BaseModule):
    def __init__(self, datamodule, rejection_threshold=0.5, feature_dim=256, lr=1e-2, source_train_type='smooth',
                 ckpt_dir=''):
        super(SourceModule, self).__init__(datamodule, feature_dim, lr, ckpt_dir)

        if source_train_type == 'smooth':
            self.train_loss = CrossEntropyLabelSmooth(num_classes=self.known_classes_num, epsilon=0.1, reduction=True)
        elif source_train_type == 'vanilla':
            self.train_loss = CrossEntropyLabelSmooth(num_classes=self.known_classes_num, epsilon=0.0, reduction=True)
        else:
            raise ValueError('Unknown source_train_type:', source_train_type)

        self.rejection_threshold = rejection_threshold

        self.total_train_acc = Accuracy(task='multiclass', num_classes=self.known_classes_num)

        self.total_test_acc = Accuracy(task='multiclass', num_classes=self.known_classes_num + 1)
        self.test_statscores = StatScores(task='binary')
        self.test_hscore = HScore(self.known_classes_num, datamodule.shared_class_num)

    def configure_optimizers(self):
        # define different learning rates for different subnetworks
        params_group = []
        for k, v in self.backbone.named_parameters():
            params_group += [{'params': v, 'lr': self.lr * 0.1}]
        for k, v in self.feature_extractor.named_parameters():
            params_group += [{'params': v, 'lr': self.lr}]
        for k, v in self.classifier.named_parameters():
            params_group += [{'params': v, 'lr': self.lr}]

        iter_max = self.trainer.max_epochs * math.ceil(len(self.trainer.datamodule.train_set) /
                                                       self.trainer.datamodule.batch_size)

        optimizer = torch.optim.SGD(params_group)
        scheduler = CustomLRScheduler(optimizer, iter_max)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            },
        }

    def lr_scheduler_step(self, scheduler, *args):
        scheduler.step(iter_num=self.global_step)

    def training_step(self, train_batch):
        x, y = train_batch
        y_hat, _ = self.forward(x, apply_softmax=True)
        # transform y to one-hot encoding
        onehot_label = torch.zeros_like(y_hat).scatter(1, y.unsqueeze(1), 1)
        loss = self.train_loss(y_hat, onehot_label)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        self.total_train_acc(y_hat, y)
        self.log('total_train_acc', self.total_train_acc, on_epoch=True, prog_bar=True)
        return loss

    def generate_class_prototypes(self):
        aggregated_class_features = torch.zeros(self.known_classes_num, self.feature_extractor.feature_dim)
        class_sample_counter = torch.zeros(self.known_classes_num)

        for x, y in self.trainer.datamodule.train_dataloader():
            with torch.no_grad():
                _, feature_embedding = self.forward(x.to(self.device))
                feature_embedding = feature_embedding.cpu()
                for c in range(self.known_classes_num):
                    idx = torch.where(y == c)
                    aggregated_class_features[c] += feature_embedding[idx].sum(dim=0)
                    class_sample_counter[c] += len(idx[0])

        return aggregated_class_features / torch.unsqueeze(class_sample_counter, -1)

    def on_train_end(self):
        print('Generating source prototypes...')
        prototypes = self.generate_class_prototypes()
        print('Save checkpoint...')
        os.makedirs(os.path.join(self.trainer.log_dir, 'checkpoints'))
        torch.save({
            'backbone_state_dict': self.backbone.state_dict(),
            'feature_extractor_state_dict': self.feature_extractor.state_dict(),
            'classifier_state_dict': self.classifier.state_dict(),
            'class_prototypes': prototypes,
        }, self.trainer.log_dir + '/checkpoints/source_ckpt.pt')

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        y = torch.where(y >= self.known_classes_num, self.known_classes_num, y)
        y_hat, feature_embedding = self.forward(x, apply_softmax=True)

        y_hat_entropy = torch.tensor(entropy(y_hat.cpu(), axis=1) / math.log(self.known_classes_num))
        pred = torch.where(y_hat_entropy <= self.rejection_threshold, torch.argmax(y_hat.cpu(), dim=1),
                           self.known_classes_num).to(self.device)
        self.total_test_acc(pred, y)
        self.log('total_test_acc', self.total_test_acc, on_step=False, on_epoch=True)

        if self.open_flag:
            self.test_hscore.update(pred, y)

            # calculate stat scores of rejection (number of TPs, FPs, TNs and FNs)
            rej_target = torch.where(y == self.known_classes_num, 1, 0)
            rej_pred = torch.where(pred == self.known_classes_num, 1, 0)
            self.test_statscores.update(rej_pred, rej_target)

    def on_test_epoch_end(self):
        if self.open_flag:
            h_score, known_acc, unknown_acc = self.test_hscore.compute()
            self.log('H-Score', h_score)
            self.log('KnownAcc', known_acc)
            self.log('UnknownAcc', unknown_acc)

