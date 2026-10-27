"""
SCMA-Net training / testing controller.
Mirrors src/HINT.py but uses SCMAInpaintingModel.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataset import Dataset
from .scma_models import SCMAInpaintingModel
from .utils import Progbar, create_dir, stitch_images, imsave
from .metrics import PSNR

from PIL import Image
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import torchvision
import lpips
import time


class SCMA:
    def __init__(self, config):
        self.config = config
        self.model_name = 'inpaint'

        self.inpaint_model = SCMAInpaintingModel(config).to(config.DEVICE)

        self.transf = torchvision.transforms.Normalize(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.loss_fn_vgg = lpips.LPIPS(net='vgg').to(config.DEVICE)
        self.psnr = PSNR(255.0).to(config.DEVICE)
        self.cal_mae = nn.L1Loss(reduction='sum')

        if config.MODE == 1:
            self.train_dataset = Dataset(
                config,
                config.TRAIN_INPAINT_IMAGE_FLIST,
                config.TRAIN_MASK_FLIST,
                augment=True, training=True,
            )
        elif config.MODE == 2:
            self.test_dataset = Dataset(
                config,
                config.TEST_INPAINT_IMAGE_FLIST,
                config.TEST_MASK_FLIST,
                augment=False, training=False,
            )

        self.samples_path = os.path.join(config.PATH, 'samples')
        self.results_path = os.path.join(config.PATH, 'results')
        if config.RESULTS is not None:
            self.results_path = config.RESULTS

        self.debug = bool(config.DEBUG)
        self.log_file = os.path.join(config.PATH, 'log_' + self.model_name + '.dat')

    # ------------------------------------------------------------------
    def load(self):
        self.inpaint_model.load()

    def save(self):
        self.inpaint_model.save()

    # ------------------------------------------------------------------
    def train(self):
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.BATCH_SIZE,
            num_workers=4,
            drop_last=True,
            shuffle=True,
        )

        epoch = 0
        keep_training = True
        max_iteration = int(float(self.config.MAX_ITERS))
        total = len(self.train_dataset)

        while keep_training:
            epoch += 1
            print('\n\nTraining epoch: %d' % epoch)
            progbar = Progbar(total, width=20,
                              stateful_metrics=['epoch', 'iter'])

            for items in train_loader:
                self.inpaint_model.train()
                images, masks = self.cuda(*items)

                outputs, gen_loss, dis_loss, logs, \
                    gen_adv, gen_rec, gen_perc, gen_sty = \
                    self.inpaint_model.process(images, masks)

                outputs_merged = outputs * masks + images * (1 - masks)

                psnr = self.psnr(self.postprocess(images),
                                 self.postprocess(outputs_merged))
                mae  = (torch.sum(torch.abs(images - outputs_merged))
                        / torch.sum(images)).float()
                logs += [('psnr', psnr.item()), ('mae', mae.item())]

                self.inpaint_model.backward(gen_loss, dis_loss)
                iteration = self.inpaint_model.iteration

                if iteration >= max_iteration:
                    keep_training = False
                    break

                logs = [('epoch', epoch), ('iter', iteration)] + logs
                progbar.add(
                    len(images),
                    values=logs if self.config.VERBOSE
                    else [x for x in logs if not x[0].startswith('l_')],
                )

                # Visualisation
                if iteration % 40 == 0:
                    create_dir(self.results_path)
                    path_joint  = os.path.join(self.results_path,
                                               self.model_name, 'joint')
                    path_masked = os.path.join(self.results_path,
                                               self.model_name, 'masked')
                    path_result = os.path.join(self.results_path,
                                               self.model_name, 'result')
                    create_dir(path_joint)
                    create_dir(path_masked)
                    create_dir(path_result)

                    name = self.train_dataset.load_name(epoch - 1)[:-4] + '.png'
                    images_joint = stitch_images(
                        self.postprocess(images),
                        self.postprocess(images * (1 - masks)),
                        self.postprocess(outputs),
                        self.postprocess(outputs_merged),
                        img_per_row=1,
                    )
                    images_joint.save(os.path.join(path_joint, name))
                    imsave(self.postprocess(images * (1 - masks) + masks)[0],
                           os.path.join(path_masked, name))
                    imsave(self.postprocess(outputs_merged)[0],
                           os.path.join(path_result, name))
                    print(name + ' complete!')

                if self.config.LOG_INTERVAL and \
                        iteration % self.config.LOG_INTERVAL == 0:
                    self.log(logs)

                if self.config.SAVE_INTERVAL and \
                        iteration % self.config.SAVE_INTERVAL == 0:
                    self.save()

        print('\nEnd training....')

    # ------------------------------------------------------------------
    def test(self):
        self.inpaint_model.eval()
        create_dir(self.results_path)

        test_loader = DataLoader(
            dataset=self.test_dataset, batch_size=1)

        psnr_list, ssim_list, l1_list, lpips_list = [], [], [], []
        index = 0

        for items in test_loader:
            images, masks = self.cuda(*items)
            index += 1
            inputs = images * (1 - masks)

            with torch.no_grad():
                t0 = int(round(time.time() * 1000))
                outputs = self.inpaint_model(images, masks)
                print('test time elapsed {}ms'.format(
                    int(round(time.time() * 1000)) - t0))

            outputs_merged = outputs * masks + images * (1 - masks)

            psnr, ssim = self.metric(images, outputs_merged)
            psnr_list.append(psnr)
            ssim_list.append(ssim)

            if torch.cuda.is_available():
                pl = self.loss_fn_vgg(
                    self.transf(outputs_merged[0].cpu()).unsqueeze(0).cuda(),
                    self.transf(images[0].cpu()).unsqueeze(0).cuda(),
                ).item()
            else:
                pl = self.loss_fn_vgg(
                    self.transf(outputs_merged[0].cpu()).unsqueeze(0),
                    self.transf(images[0].cpu()).unsqueeze(0),
                ).item()
            lpips_list.append(pl)

            l1 = F.l1_loss(outputs_merged, images, reduction='mean').item()
            l1_list.append(l1)

            print("psnr:{:.4f}/{:.4f}  ssim:{:.4f}/{:.4f}  "
                  "l1:{:.6f}/{:.6f}  lpips:{:.4f}/{:.4f}  n={}".format(
                      psnr, np.mean(psnr_list),
                      ssim, np.mean(ssim_list),
                      l1,   np.mean(l1_list),
                      pl,   np.mean(lpips_list),
                      index))

            # Save results
            name = self.test_dataset.load_name(index - 1)[:-4] + '.png'
            path_joint  = os.path.join(self.results_path,
                                       self.model_name, 'joint')
            path_masked = os.path.join(self.results_path,
                                       self.model_name, 'masked')
            path_result = os.path.join(self.results_path,
                                       self.model_name, 'result')
            create_dir(path_joint)
            create_dir(path_masked)
            create_dir(path_result)

            images_joint = stitch_images(
                self.postprocess(images),
                self.postprocess(inputs),
                self.postprocess(outputs),
                self.postprocess(outputs_merged),
                img_per_row=1,
            )
            images_joint.save(os.path.join(path_joint, name))
            imsave(self.postprocess(images * (1 - masks) + masks)[0],
                   os.path.join(path_masked, name))
            imsave(self.postprocess(outputs_merged)[0],
                   os.path.join(path_result, name))
            print(name + ' complete!')

        print('\nEnd Testing')
        print('psnr_avg:{:.4f}  ssim_avg:{:.4f}  '
              'l1_avg:{:.6f}  lpips_avg:{:.4f}'.format(
                  np.mean(psnr_list), np.mean(ssim_list),
                  np.mean(l1_list),   np.mean(lpips_list)))

    # ------------------------------------------------------------------
    def log(self, logs):
        with open(self.log_file, 'a') as f:
            f.write('%s\n' % ' '.join([str(item[1]) for item in logs]))

    def cuda(self, *args):
        return (item.to(self.config.DEVICE) for item in args)

    def postprocess(self, img):
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()

    def metric(self, gt, pre):
        pre = pre.clamp_(0, 1) * 255.0
        pre = pre.permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.uint8)[0]
        gt  = gt.clamp_(0, 1) * 255.0
        gt  = gt.permute(0, 2, 3, 1).cpu().detach().numpy().astype(np.uint8)[0]
        psnr = min(100, compare_psnr(gt, pre))
        ssim = compare_ssim(gt, pre, multichannel=True, data_range=255)
        return psnr, ssim
