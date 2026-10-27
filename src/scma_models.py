"""
SCMA-Net model wrapper (generator + discriminator + losses + optimizers).
Mirrors src/models.py but uses SCMANet instead of HINT.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .scma_networks import SCMANet, Discriminator
from .loss import AdversarialLoss, PerceptualLoss, StyleLoss


class BaseModel(nn.Module):
    def __init__(self, name, config):
        super().__init__()
        self.name = name
        self.config = config
        self.iteration = 0
        self.gen_weights_path = os.path.join(config.PATH, name + '_gen.pth')
        self.dis_weights_path = os.path.join(config.PATH, name + '_dis.pth')

    def load(self):
        if os.path.exists(self.gen_weights_path):
            print('Loading %s generator...' % self.name)
            if torch.cuda.is_available():
                data = torch.load(self.gen_weights_path)
            else:
                data = torch.load(self.gen_weights_path,
                                  map_location=lambda storage, loc: storage)
            self.generator.load_state_dict(data['generator'], strict=False)
            self.iteration = data['iteration']

        if self.config.MODE == 1 and os.path.exists(self.dis_weights_path):
            print('Loading %s discriminator...' % self.name)
            if torch.cuda.is_available():
                data = torch.load(self.dis_weights_path)
            else:
                data = torch.load(self.dis_weights_path,
                                  map_location=lambda storage, loc: storage)
            self.discriminator.load_state_dict(data['discriminator'])

    def save(self):
        print('\nsaving %s...\n' % self.name)
        torch.save({'iteration': self.iteration,
                    'generator': self.generator.state_dict()},
                   self.gen_weights_path)
        torch.save({'discriminator': self.discriminator.state_dict()},
                   self.dis_weights_path)


class SCMAInpaintingModel(BaseModel):
    """
    Training / inference wrapper for SCMA-Net.

    Loss weights follow svip10.tex Section 2.4:
      λ_rec=1, λ_style=250, λ_perc=0.1, λ_adv=0.001
    (These can be overridden via config.)
    """
    def __init__(self, config):
        super().__init__('SCMAInpaintingModel', config)

        generator = SCMANet(
            inp_channels=4,
            out_channels=3,
            dim=getattr(config, 'SCMA_DIM', 48),
            num_blocks=getattr(config, 'SCMA_BLOCKS', [4, 6, 6, 8, 6, 6, 4]),
            ffn_expansion_factor=getattr(config, 'SCMA_FFN_EXP', 2.66),
            bias=False,
        )
        discriminator = Discriminator(
            in_channels=3,
            use_sigmoid=config.GAN_LOSS != 'hinge',
        )

        if len(config.GPU) > 1:
            generator     = nn.DataParallel(generator, config.GPU)
            discriminator = nn.DataParallel(discriminator, config.GPU)

        self.add_module('generator', generator)
        self.add_module('discriminator', discriminator)
        self.add_module('l1_loss', nn.L1Loss())
        self.add_module('perceptual_loss', PerceptualLoss())
        self.add_module('style_loss', StyleLoss())
        self.add_module('adversarial_loss', AdversarialLoss(type=config.GAN_LOSS))

        self.gen_optimizer = optim.Adam(
            generator.parameters(),
            lr=float(config.LR),
            betas=(config.BETA1, config.BETA2),
        )
        self.dis_optimizer = optim.Adam(
            discriminator.parameters(),
            lr=float(config.LR) * float(config.D2G_LR),
            betas=(config.BETA1, config.BETA2),
        )

        # Loss weights (paper defaults; override via config)
        self.w_rec   = getattr(config, 'L1_LOSS_WEIGHT',      1.0)
        self.w_style = getattr(config, 'STYLE_LOSS_WEIGHT',   250.0)
        self.w_perc  = getattr(config, 'CONTENT_LOSS_WEIGHT', 0.1)
        self.w_adv   = getattr(config, 'INPAINT_ADV_LOSS_WEIGHT', 0.001)

    # ------------------------------------------------------------------
    def forward(self, images, masks):
        """Return inpainted output tensor."""
        images_masked = images * (1 - masks) + masks  # white-fill masked region
        return self.generator(images_masked, masks)

    # ------------------------------------------------------------------
    def process(self, images, masks):
        self.iteration += 1
        self.gen_optimizer.zero_grad()
        self.dis_optimizer.zero_grad()

        outputs = self(images, masks)

        # Discriminator losses
        dis_real, _ = self.discriminator(images)
        dis_fake, _ = self.discriminator(outputs.detach())
        dis_real_loss = self.adversarial_loss(dis_real, True,  True)
        dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
        dis_loss = (dis_real_loss + dis_fake_loss) / 2

        # Generator losses
        gen_fake, _ = self.discriminator(outputs)
        gen_adv  = self.adversarial_loss(gen_fake, True, False) * self.w_adv
        gen_rec  = self.l1_loss(outputs, images) * self.w_rec / torch.mean(masks)
        gen_perc = self.perceptual_loss(outputs, images) * self.w_perc
        gen_sty  = self.style_loss(outputs * masks, images * masks) * self.w_style
        gen_loss = gen_adv + gen_rec + gen_perc + gen_sty

        logs = [
            ('gLoss', gen_loss.item()),
            ('dLoss', dis_loss.item()),
        ]
        return outputs, gen_loss, dis_loss, logs, gen_adv, gen_rec, gen_perc, gen_sty

    # ------------------------------------------------------------------
    def backward(self, gen_loss, dis_loss):
        dis_loss.backward(retain_graph=True)
        gen_loss.backward()
        self.dis_optimizer.step()
        self.gen_optimizer.step()
