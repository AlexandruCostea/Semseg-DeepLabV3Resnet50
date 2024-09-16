import os
import time
import json
from dotenv import load_dotenv

import shutil

import torch
import torch.utils.data as data
import torch.backends.cudnn as cudnn

from data_loader.semseg import Semseg
from utils.focal_loss import FocalLoss
from utils.lr_scheduler import LRScheduler
from utils.metric import SegmentationMetric
from torchinfo import summary

from models.custom_model import SemsegCustom

class Trainer(object):
    def __init__(self):
        cudnn.benchmark = True

        images_path = os.getenv('IMAGES_PATH')
        labels_path = os.getenv('LABELS_PATH')

        image_paths = [os.path.join(images_path, file) for file in sorted(os.listdir(images_path))]
        label_paths = [os.path.join(labels_path, file) for file in sorted(os.listdir(labels_path))]

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


        # dataset and dataloader
        train_split = float(os.getenv('TRAIN_SPLIT'))

        dataset = Semseg(image_paths, label_paths, small_sample=False)

        train_size = int(train_split * len(dataset))
        val_size = len(dataset) - train_size

        train_dataset, val_dataset = data.random_split(dataset, [train_size, val_size])

        self.train_loader = data.DataLoader(dataset=train_dataset,
                                            batch_size=int(os.getenv('BATCH_SIZE')),
                                            shuffle=True,
                                            drop_last=True)
        self.val_loader = data.DataLoader(dataset=val_dataset,
                                          batch_size=1,
                                          shuffle=False)
        

        # create network
        checkpoint = os.getenv('CHECKPOINT')
        if checkpoint != 'None':
            print("Loading checkpoint: ", checkpoint)
            self.model = SemsegCustom(checkpoint=checkpoint)
        else:
            print("Loading default checkpoint")
            self.model = SemsegCustom()
        self.model.to(self.device)

        self.criterion = FocalLoss(ignore_index=-1).to(self.device)


        # optimizer
        lr = float(os.getenv('LR'))
        momentum = float(os.getenv('MOMENTUM'))
        decay = float(os.getenv('WEIGHT_DECAY'))
        # self.optimizer = torch.optim.SGD(self.model.parameters(),
        #                                  lr=lr,
        #                                  momentum=momentum,
        #                                  weight_decay=decay)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=decay)

        # lr scheduling
        epochs = int(os.getenv("EPOCHS"))
        self.lr_scheduler = LRScheduler(mode='poly', base_lr=lr, nepochs=epochs,
                                        iters_per_epoch=len(self.train_loader), power=0.9)

        # evaluation metrics
        self.metric = SegmentationMetric(8)

        self.best_pred = 0.0

    def train(self):
        cur_iters = 0
        start_time = time.time()
        start_epoch = int(os.getenv("START_EPOCH"))
        epochs = int(os.getenv("EPOCHS"))

        for epoch in range(start_epoch, epochs):
            self.model.train()
            for i, (images, targets) in enumerate(self.train_loader):
                cur_lr = self.lr_scheduler(cur_iters)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = cur_lr

                images = images.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                cur_iters += 1
                print('Epoch: [%2d/%2d] Iter [%4d/%4d] || Time: %4.4f sec || lr: %.8f || Loss: %.4f' % (
                    epoch, epochs, i + 1, len(self.train_loader),
                    time.time() - start_time, cur_lr, loss.item()))


            save_every_epoch = int(os.getenv('SAVE_EVERY_EPOCH'))
            if save_every_epoch > 0:
                ckpt_name = f"8class_epoch_{epoch}"
                save_checkpoint(self.model, ckpt_name, is_best=False)

            self.validation(epoch)

        save_checkpoint(self.model, "8class_final", is_best=False)

    def validation(self, epoch):
        is_best = False
        self.metric.reset()
        self.model.eval()

        for i, (image, target) in enumerate(self.val_loader):
            image = image.to(self.device)

            outputs = self.model(image)
            pred = torch.argmax(outputs, dim=1)
            pred = pred.cpu().data.numpy()


            self.metric.update(pred, target.numpy())
            pixAcc, mIoU = self.metric.get()
            print('Epoch %d, Sample %d, validation pixAcc: %.3f%%, mIoU: %.3f%%' % (
                epoch, i + 1, pixAcc * 100, mIoU * 100))

        new_pred = (pixAcc + mIoU) / 2
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            save_checkpoint(self.model, is_best)


def save_checkpoint(model, name, is_best=False):
    """Save Checkpoint"""
    save_folder = os.getenv('SAVE_FOLDER')
    directory = os.path.expanduser(save_folder)
    dataset = os.getenv('PRE_TRAINING_DATASET')
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = '{}_{}.pth'.format('deeplabv3_resnet50', name)
    save_path = os.path.join(directory, filename)
    torch.save(model.deeplabv3.classifier[4].state_dict(), save_path)
    if is_best:
        best_filename = '{}_{}_best_model.pth'.format('deeplabv3_resnet50', name)
        best_filename = os.path.join(directory, best_filename)
        shutil.copyfile(filename, best_filename)


if __name__ == '__main__':
    load_dotenv()

    trainer = Trainer()
    resume = os.getenv('RESUME')
    if resume == "":
        resume = None

    eval = int(os.getenv('EVAL_ONLY'))
    start_epoch = int(os.getenv('START_EPOCH'))
    if eval > 0:
        print('Evaluation model: ', resume)
        trainer.validation(start_epoch)
    else:
        epochs = int(os.getenv("EPOCHS"))
        print('Starting Epoch: %d, Total Epochs: %d' % (start_epoch, epochs))
        trainer.train()
