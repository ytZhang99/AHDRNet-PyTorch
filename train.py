import os
import cv2
import time
import torch
import random
import imageio
import matplotlib
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.utils.data as data
import torchvision.transforms as transforms

from option import args
from logger import Logger
from utils import *
from dataset import TrainData, ValData ,TestData
from model import AHDRNet
from torch.optim import Adam, lr_scheduler


class Trainer(object):
    def __init__(self):
        # Training Settings
        self.num_epochs = args.epochs
        self.lr = args.lr
        self.train_set = TrainData(args.dir_train)
        self.train_loader = data.DataLoader(self.train_set, batch_size=args.batch_size,
                                            shuffle=True, num_workers=0, pin_memory=False)
        self.batch_sum = len(self.train_loader)

        self.model = AHDRNet().cuda()
        self.optimizer = Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=200, gamma=0.5)
        self.criterion = nn.L1Loss()
        self.train_loss = []

        # Validation Settings
        self.ep = None
        self.val_set = ValData(args.dir_test)
        self.val_loader = data.DataLoader(self.val_set)
        self.val_num = len(self.val_loader)
        self.val_psnr = 0
        self.curr_psnr = [0.]

        # Test Settings
        if args.test_only:
            self.test_set = TestData(args.dir_test)
            self.test_loader = data.DataLoader(self.test_set, batch_size=1, shuffle=False,
                                               num_workers=0, pin_memory=False)
            self.test_num = len(self.test_loader)

    def train(self):
        logger = Logger(args.logger_file)
        seed = args.seed
        random.seed(seed)
        torch.manual_seed(seed)
        print('# Model parameters:', sum(param.numel() for param in self.model.parameters()))

        if os.path.exists(args.model_path + args.model):
            print('===> Loading pre-trained model......')
            state = torch.load(args.model_path + args.model)
            self.model.load_state_dict(state['model'])
            # self.optimizer.load_state_dict(state['optimizer'])
        else:
            self.lr = args.lr

        for ep in range(self.num_epochs):
            ep_loss = 0.
            logger = Logger(args.logger_file, True)
            logger.append('Epoch: %d' % ep)
            for batch_idx, batch_data in enumerate(self.train_loader):
                batch_ldr0, batch_ldr1, batch_ldr2 = batch_data['input0'].cuda(), batch_data['input1'].cuda(), \
                                                     batch_data['input2'].cuda()
                label = batch_data['label'].cuda()

                torch.cuda.synchronize()
                start_time = time.time()

                pred = self.model(batch_ldr0, batch_ldr1, batch_ldr2)
                pred = torch.clamp(pred, 0., 1.)
                pred = range_compressor_tensor(pred)
                
                self.optimizer.zero_grad()
                loss = self.criterion(pred, label)
                loss.backward()
                self.optimizer.step()

                torch.cuda.synchronize()
                end_time = time.time()

                if batch_idx % args.log_interval == 0 and batch_idx != 0:
                    print(
                        'Epoch:{}\tcur/all:{}/{}\tLoss_D:{:.4f}\tTime:{:.2f}s '
                        .format(ep + 1, batch_idx, len(self.train_loader),
                                loss.item(),
                                end_time - start_time))

                # accumulate loss for each batch
                ep_loss += loss.item()
            self.scheduler.step()
            self.train_loss.append(ep_loss / self.batch_sum)

            #  save loss for each epoch
            logger = Logger(args.logger_file, True)
            logger.append('loss:{:.4f}'.format(ep_loss / self.batch_sum))

            # save models
            state = {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict()
            }

            torch.save(state, args.model_path + args.model)
            if ep % 20 == 0:
                torch.save(state, args.model_path + str(ep) + '.pth')

            # plot loss curve
            matplotlib.use('Agg')
            fig1 = plt.figure()
            plt.plot(self.train_loss)
            plt.savefig('loss_curve.png')
            plt.close('all')

            if (ep + 1) % args.val_interval == 0:
                self.validation(ep)
        print('===> Finished Training!')

    def validation(self, ep):
        self.ep = ep
        self.model.eval()
        state = torch.load(args.model_path + args.model)
        self.model.load_state_dict(state['model'])
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(self.val_loader):
                batch_ldr0, batch_ldr1, batch_ldr2 = batch_data['input0'].cuda(), batch_data['input1'].cuda(), \
                                                     batch_data['input2'].cuda()

                label = batch_data['label'].cuda()

                pred = self.model(batch_ldr0, batch_ldr1, batch_ldr2)
                pred = torch.clamp(pred, 0., 1.)
                pred = range_compressor_tensor(pred)

                psnr_pred = torch.squeeze(pred.clone())
                psnr_pred = psnr_pred.data.cpu().numpy().astype(np.float32)
                psnr_label = torch.squeeze(label.clone())
                psnr_label = psnr_label.data.cpu().numpy().astype(np.float32)
                psnr_mu = normalized_psnr(psnr_pred, psnr_label, psnr_label.max())
                self.val_psnr += psnr_mu

        self.val_psnr /= self.val_num
        print('Average PSNR_mu: {:.4f} dB'.format(self.val_psnr))
        if self.val_psnr > max(self.curr_psnr):
            torch.save(state, args.model_path + 'best_checkpoint.pth')
            with open('./best_ckp.json', 'w') as f:
                f.write('best epoch:' + str(self.ep) + '\n')
                f.write('Validation set: Average PSNR_mu: {:.4f}\n'.format(self.val_psnr))
        self.curr_psnr.append(self.val_psnr)
        matplotlib.use('Agg')
        fig2 = plt.figure()
        plt.plot(self.curr_psnr)
        plt.savefig('val_curve.png')
        plt.close('all')

    def test(self, ep):
        self.ep = ep
        self.model.eval()
        state = torch.load(args.model_path + args.model)
        self.model.load_state_dict(state['model'])
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(self.val_loader):
                print('Processing picture No.{}'.format(batch_idx + 1))
                batch_ldr0, batch_ldr1, batch_ldr2 = batch_data['input0'].cuda(), batch_data['input1'].cuda(), \
                                                     batch_data['input2'].cuda()

                pred = self.model(batch_ldr0, batch_ldr1, batch_ldr2)
                pred = torch.clamp(pred, 0., 1.)

                if 0 <= self.ep < args.epochs:
                    save_path = args.save_dir + str(self.ep) + '_epoch/'
                else:
                    save_path = args.save_dir
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                pred_save = torch.squeeze(pred.clone())
                pred_save = pred_save.data.cpu().numpy().astype(np.float32)
                # [C, H, W] -> [H, W, C]
                pred_save = np.transpose(pred_save, (1, 2, 0))
                # BGR -> RGB
                pred_save = pred_save[:, :, [2, 1, 0]]
                imageio.imwrite(save_path + str(batch_idx) + '.hdr', pred_save, 'hdr')
        print('Finished Testing!')
