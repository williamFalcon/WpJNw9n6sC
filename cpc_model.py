import torch
import torch.optim as optim
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torch.nn.functional as F
from test_tube import HyperOptArgumentParser
import pytorch_lightning as pl
from torch.optim.lr_scheduler import MultiStepLR
from fisherman.models.submodels import cpc_loss, cpc_transforms, cpc_nets

import math


class CPCSelfSupervised(pl.LightningModule):

    # ------------------------------
    # INIT
    # ------------------------------
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams

        # encoder network (Z vectors)
        dummy_batch = torch.zeros((2, 3, hparams.patch_size, hparams.patch_size))
        self.encoder = cpc_nets.CPCResNet101(dummy_batch)

        # context network (C vectors)
        c, h = self.__compute_final_nb_c(hparams.patch_size)
        self.context_network = cpc_nets.MaskedConv2d(c)

        # W transforms (k > 0)
        self.W_list = {}
        for k in range(1, h):
            w = torch.nn.Linear(c, c)
            self.W_list[str(k)] = w

        self.W_list = torch.nn.ModuleDict(self.W_list)

        # loss (has cached sampling layers, no params)
        self.nce_loss = cpc_loss.CPCLossNCE()

    def __compute_final_nb_c(self, patch_size):
        dummy_batch = torch.zeros((2*49, 3, patch_size, patch_size))
        dummy_batch = self.encoder(dummy_batch)
        dummy_batch = self.__recover_z_shape(dummy_batch, 2)
        b, c, h, w = dummy_batch.size()
        return c, h

    def __recover_z_shape(self, Z, b):
        # recover shape
        Z = Z.squeeze(-1)
        nb_feats = int(math.sqrt(Z.size(0) // b))
        Z = Z.view(b, -1, Z.size(1))
        Z = Z.permute(0, 2, 1).contiguous()
        Z = Z.view(b, -1, nb_feats, nb_feats)

        return Z

    # ------------------------------
    # FWD
    # ------------------------------
    def forward(self, img_1):
        # put all patches on the batch dim for simultaneous processing
        b, p, c, w, h = img_1.size()
        img_1 = img_1.view(-1, c, w, h)

        # Z are the latent vars
        Z = self.encoder(img_1)
        Z = self.__recover_z_shape(Z, b)

        return Z

    def training_step(self, batch, batch_nb):
        img_1, _ = batch

        # Latent features
        Z = self.forward(img_1.half())
        Z = Z.half()

        # generate the context vars
        C = self.context_network(Z)

        # apply masked context network

        # ------------------
        # NCE LOSS
        loss = self.nce_loss(Z, C, self.W_list)
        unsupervised_loss = loss
        if self.trainer.use_amp:
            unsupervised_loss = unsupervised_loss.half()

        # ------------------
        # FULL LOSS
        total_loss = unsupervised_loss
        result = {
            'loss': total_loss
        }

        return result

    def validation_step(self, batch, batch_nb):
        img_1, labels = batch

        if self.trainer.use_amp:
            img_1 = img_1.half()

        # generate features
        # Latent features
        Z = self.forward(img_1)
        Z = Z.half()

        # generate the context vars
        C = self.context_network(Z.half())

        # NCE LOSS
        loss = self.nce_loss(Z, C, self.W_list)
        unsupervised_loss = loss

        result = {
            'val_nce': unsupervised_loss
        }
        return result

    def validation_end(self, outputs):
        val_nce = 0
        for output in outputs:
            val_nce += output['val_nce']

        val_nce = val_nce / len(outputs)
        return {'val_nce': val_nce}

    def optimizer_step(self, epoch_nb, batch_nb, optimizer, optimizer_i):
        if self.trainer.global_step < 500:
            lr_scale = min(1., float(self.trainer.global_step + 1) / 500.)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.hparams.learning_rate

        optimizer.step()
        optimizer.zero_grad()

    def configure_optimizers(self):
        opt = optim.Adam(
            params=self.parameters(),
            lr=self.hparams.learning_rate,
            betas=(0.8, 0.999),
            weight_decay=1e-5,
            eps=1e-8
        )

        if self.hparams.dataset_name == 'CIFAR10': # Dataset.C100, Dataset.STL10
            lr_scheduler = MultiStepLR(opt, milestones=[250, 280], gamma=0.2)
        else:
            lr_scheduler = MultiStepLR(opt, milestones=[30, 45], gamma=0.2)

        return [opt], [lr_scheduler]

    @pl.data_loader
    def tng_dataloader(self):
        if self.hparams.dataset_name == 'CIFAR10':
            train_transform = cpc_transforms.CPCTransformsC10()
            dataset = CIFAR10(root=self.hparams.cifar10_root, train=True, transform=train_transform, download=True)

            dist_sampler = None
            if self.trainer.use_ddp:
                dist_sampler = torch.utils.data.distributed.DistributedSampler(dataset)

            loader = DataLoader(
                dataset=dataset,
                batch_size=self.hparams.batch_size,
                pin_memory=True,
                drop_last=True,
                num_workers=16,
                sampler=dist_sampler
            )

            return loader

    @pl.data_loader
    def val_dataloader(self):
        if self.hparams.dataset_name == 'CIFAR10':
            train_transform = cpc_transforms.CPCTransformsC10()
            dataset = CIFAR10(root=self.hparams.cifar10_root, train=False, transform=train_transform, download=True)

            dist_sampler = None
            if self.trainer.use_ddp:
                dist_sampler = torch.utils.data.distributed.DistributedSampler(dataset)

            loader = DataLoader(
                dataset=dataset,
                batch_size=self.hparams.batch_size,
                pin_memory=True,
                drop_last=True,
                num_workers=16,
                sampler=dist_sampler
            )

            return loader

    @staticmethod
    def add_model_specific_args(parent_parser, root_dir):
        parser = HyperOptArgumentParser(strategy=parent_parser.strategy, parents=[parent_parser])

        parser.set_defaults(nb_hopt_trials=1000)
        parser.set_defaults(min_nb_epochs=1000)
        parser.set_defaults(max_nb_epochs=1100)
        parser.set_defaults(early_stop_metric='val_nce')
        parser.set_defaults(model_save_monitor_value='val_nce')
        parser.set_defaults(model_save_monitor_mode='min')
        parser.set_defaults(early_stop_mode='min')

        # CIFAR 10
        dataset_name = 'CIFAR10'
        image_height = 32
        nb_classes = 10
        patch_size = 8
        patch_overlap = 4

        # dataset options
        parser.opt_list('--nb_classes', default=nb_classes, type=int, options=[10], tunable=False)
        parser.opt_list('--patch_size', default=patch_size, type=int, options=[10], tunable=False)
        parser.opt_list('--patch_overlap', default=patch_overlap, type=int, options=[10], tunable=False)

        # network params
        parser.add_argument('--image_height', type=int, default=image_height)

        # trainin params
        parser.add_argument('--dataset_name', type=str, default=dataset_name)
        parser.add_argument('--batch_size', type=int, default=200, help='input batch size (default: 200)')
        parser.opt_list('--learning_rate', type=float, default=0.0002, options=[
            2e-4*(1/64), 2e-4*(1/32),
            2e-4*(1/16), 2e-4*(1/8),
            2e-4*(1/4), 2e-4*(1/2),
            2e-4*(1/4),
            2e-4,
            2e-4*4, #2e-4*4,
            2e-4*8,
            ], tunable=False)

        # data
        parser.opt_list('--cifar10_root', default=f'{root_dir}/fisherman/datasets', type=str, tunable=False)
        return parser
