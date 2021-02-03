### Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


class Pix2PixHDModel(BaseModel):
    def name(self):
        return "Pix2PixHDModel"

    def init_loss_filter(self, use_gan_feat_loss, use_vgg_loss):
        flags = (True, use_gan_feat_loss, use_vgg_loss, True, True)

        def loss_filter(g_gan, g_gan_feat, g_vgg, d_real, d_fake):
            return [
                l for (l, f) in zip((g_gan, g_gan_feat, g_vgg, d_real,
                                     d_fake), flags) if f
            ]

        return loss_filter

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        if (opt.resize_or_crop != "none" or
                not opt.isTrain):  # when training at full res this causes OOM
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain

        ##### define networks
        # Generator network
        netG_input_nc = opt.label_nc + opt.image_nc
        if not opt.no_instance:
            netG_input_nc += 1

        self.netG = networks.define_G(
            netG_input_nc,
            opt.output_nc,
            opt.ngf,
            opt.netG,
            opt.n_downsample_global,
            opt.n_blocks_global,
            opt.n_local_enhancers,
            opt.n_blocks_local,
            opt.norm,
            gpu_ids=self.gpu_ids,
        )

        # Discriminator network
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            netD_input_nc = 2 * (opt.label_nc + opt.image_nc)
            if not opt.no_instance:
                netD_input_nc += 1
            self.netD = networks.define_D(
                netD_input_nc,
                opt.ndf,
                opt.n_layers_D,
                opt.norm,
                use_sigmoid,
                opt.num_D,
                not opt.no_ganFeat_loss,
                gpu_ids=self.gpu_ids,
            )

        if self.opt.verbose:
            print("---------- Networks initialized -------------")

        # load networks
        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            pretrained_path = "" if not self.isTrain else opt.load_pretrain
            self.load_network(self.netG, "G", opt.which_epoch, pretrained_path)
            if self.isTrain:
                self.load_network(self.netD, "D", opt.which_epoch,
                                  pretrained_path)
            # if self.gen_features:
            #     self.load_network(self.netE, "E", opt.which_epoch, pretrained_path)

        # set loss functions and optimizers
        if self.isTrain:
            if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
                raise NotImplementedError(
                    "Fake Pool Not Implemented for MultiGPU")
            self.fake_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr

            # define loss functions
            self.loss_filter = self.init_loss_filter(not opt.no_ganFeat_loss,
                                                     not opt.no_vgg_loss)

            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan,
                                                 tensor=self.Tensor)
            self.criterionFeat = torch.nn.L1Loss()
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss(self.gpu_ids)

            # Names so we can breakout loss
            self.loss_names = [
                'G_GAN', 'G_GAN_Feat', 'G_VGG', 'D_real', 'D_fake',
                'G_GANface', 'D_realface', 'D_fakeface'
            ]

            # initialize optimizers
            # optimizer G
            if opt.niter_fix_global > 0:
                import sys

                if sys.version_info >= (3, 0):
                    finetune_list = set()
                else:
                    from sets import Set

                    finetune_list = Set()

                params_dict = dict(self.netG.named_parameters())
                params = []
                for key, value in params_dict.items():
                    if key.startswith("model" + str(opt.n_local_enhancers)):
                        params += [value]
                        finetune_list.add(key.split(".")[0])
                print(
                    "------------- Only training the local enhancer network (for %d epochs) ------------"
                    % opt.niter_fix_global)
                print("The layers that are finetuned are ",
                      sorted(finetune_list))
            else:
                params = list(self.netG.parameters())
            # if self.gen_features:
            #     params += list(self.netE.parameters())
            self.optimizer_G = torch.optim.Adam(params,
                                                lr=opt.lr,
                                                betas=(opt.beta1, 0.999))

            # optimizer D
            params = list(self.netD.parameters())
            self.optimizer_D = torch.optim.Adam(params,
                                                lr=opt.lr,
                                                betas=(opt.beta1, 0.999))

    def encode_input(
        self,
        label_map,
        real_image=None,
        next_label=None,
        next_image=None,
        zeroshere=None,
        infer=False,
    ):

        input_label = label_map.float().cuda()

        if next_label is not None:
            next_label = next_label.float().cuda()
        # real images for training
        if real_image is not None:
            real_image = real_image.cuda()

        # real images for training
        if next_image is not None:
            next_image = next_image.float().cuda()

        if zeroshere is not None:
            zeroshere = zeroshere.float().cuda()

        return input_label, real_image, next_label, next_image, zeroshere

    def discriminate(self, input_label, test_image, use_pool=False):
        input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        if use_pool:
            fake_query = self.fake_pool.query(input_concat)
            return self.netD.forward(fake_query)
        else:
            return self.netD.forward(input_concat)

    def discriminate_4(self, s0, s1, i0, i1, use_pool=False):
        input_concat = torch.cat((s0, s1, i0.detach(), i1.detach()), dim=1)
        if use_pool:
            fake_query = self.fake_pool.query(input_concat)
            return self.netD.forward(fake_query)
        else:
            return self.netD.forward(input_concat)

    def forward(self,
                label,
                next_label,
                image,
                next_image,
                face_coords,
                zeroshere,
                infer=False):
        # Encode Inputs
        input_label, real_image, next_label, next_image, zeroshere = self.encode_input(
            label,
            image,
            next_label=next_label,
            next_image=next_image,
            zeroshere=zeroshere,
        )

        initial_I_0 = 0

        # Fake Generation I_0
        input_concat = torch.cat((input_label, zeroshere), dim=1)
        I_0 = self.netG.forward(input_concat)
        # Fake Generation I_1
        input_concat = torch.cat((next_label, I_0), dim=1)
        I_1 = self.netG.forward(input_concat)

        loss_D_fake_face = loss_D_real_face = loss_G_GAN_face = 0
        fake_face_0 = fake_face_1 = real_face_0 = real_face_1 = 0
        fake_face = real_face = face_residual = 0

        # Fake Detection and Loss
        pred_fake_pool = self.discriminate_4(input_label,
                                             next_label,
                                             I_0,
                                             I_1,
                                             use_pool=True)
        loss_D_fake = self.criterionGAN(pred_fake_pool, False)

        # Real Detection and Loss
        pred_real = self.discriminate_4(input_label, next_label, real_image,
                                        next_image)
        loss_D_real = self.criterionGAN(pred_real, True)

        # GAN loss (Fake Passability Loss)
        pred_fake = self.netD.forward(
            torch.cat((input_label, next_label, I_0, I_1), dim=1))
        loss_G_GAN = self.criterionGAN(pred_fake, True)

        # GAN feature matching loss
        loss_G_GAN_Feat = 0
        if not self.opt.no_ganFeat_loss:
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D
            for i in range(self.opt.num_D):
                for j in range(len(pred_fake[i]) - 1):
                    loss_G_GAN_Feat += (
                        D_weights * feat_weights * self.criterionFeat(
                            pred_fake[i][j], pred_real[i][j].detach()) *
                        self.opt.lambda_feat)

        # VGG feature matching loss
        loss_G_VGG = 0
        if not self.opt.no_vgg_loss:
            loss_G_VGG0 = self.criterionVGG(I_0,
                                            real_image) * self.opt.lambda_feat
            loss_G_VGG1 = self.criterionVGG(I_1,
                                            next_image) * self.opt.lambda_feat
            loss_G_VGG = loss_G_VGG0 + loss_G_VGG1
            if (self.opt.netG == "global"
                ):  # need 2x VGG for artifacts when training local
                loss_G_VGG *= 0.5
            # if self.opt.face_discrim:
            #     loss_G_VGG += (
            #         0.5
            #         * self.criterionVGG(fake_face_0, real_face_0)
            #         * self.opt.lambda_feat
            #     )
            #     loss_G_VGG += (
            #         0.5
            #         * self.criterionVGG(fake_face_1, real_face_1)
            #         * self.opt.lambda_feat
            #     )

        # if self.opt.use_l1:
        #     loss_G_VGG += (self.criterionL1(I_1, next_image)) * self.opt.lambda_A

        # Only return the fake_B image if necessary to save BW
        return [
            [
                loss_G_GAN,
                loss_G_GAN_Feat,
                loss_G_VGG,
                loss_D_real,
                loss_D_fake,
                loss_G_GAN_face,
                loss_D_real_face,
                loss_D_fake_face,
            ],
            None if not infer else [
                torch.cat(
                    (I_0, I_1), dim=3), fake_face, face_residual, initial_I_0
            ],
        ]

    def inference(self, label, prevouts, face_coords):

        # Encode Inputs
        input_label, _, _, _, prevouts = self.encode_input(label,
                                                           zeroshere=prevouts,
                                                           infer=True)

        if self.opt.face_generator:
            miny = face_coords[0][0]
            maxy = face_coords[0][1]
            minx = face_coords[0][2]
            maxx = face_coords[0][3]
        """ new face """
        I_0 = 0
        # Fake Generation

        input_concat = torch.cat((input_label, prevouts), dim=1)
        initial_I_0 = self.netG.forward(input_concat)

        # if self.opt.face_generator:
        #     face_label_0 = input_label[:, :, miny:maxy, minx:maxx]
        #     face_residual_0 = self.faceGen.forward(
        #         torch.cat(
        #             (face_label_0, initial_I_0[:, :, miny:maxy, minx:maxx]),
        #             dim=1))
        #     I_0 = initial_I_0.clone()
        #     I_0[:, :, miny:maxy,
        #         minx:maxx] = initial_I_0[:, :, miny:maxy,
        #                                  minx:maxx] + face_residual_0
        #     fake_face_0 = I_0[:, :, miny:maxy, minx:maxx]
        #     return I_0
        return initial_I_0

    def sample_features(self, inst):
        # read precomputed feature clusters
        cluster_path = os.path.join(self.opt.checkpoints_dir, self.opt.name,
                                    self.opt.cluster_path)
        features_clustered = np.load(cluster_path, encoding="latin1").item()

        # randomly sample from the feature clusters
        inst_np = inst.cpu().numpy().astype(int)
        feat_map = self.Tensor(inst.size()[0], self.opt.feat_num,
                               inst.size()[2],
                               inst.size()[3])
        for i in np.unique(inst_np):
            label = i if i < 1000 else i // 1000
            if label in features_clustered:
                feat = features_clustered[label]
                cluster_idx = np.random.randint(0, feat.shape[0])

                idx = (inst == int(i)).nonzero()
                for k in range(self.opt.feat_num):
                    feat_map[idx[:, 0], idx[:, 1] + k, idx[:, 2],
                             idx[:, 3]] = feat[cluster_idx, k]
        if self.opt.data_type == 16:
            feat_map = feat_map.half()
        return feat_map

    def encode_features(self, image, inst):
        image = Variable(image.cuda(), volatile=True)
        feat_num = self.opt.feat_num
        h, w = inst.size()[2], inst.size()[3]
        block_num = 32
        feat_map = self.netE.forward(image, inst.cuda())
        inst_np = inst.cpu().numpy().astype(int)
        feature = {}
        for i in range(self.opt.label_nc):
            feature[i] = np.zeros((0, feat_num + 1))
        for i in np.unique(inst_np):
            label = i if i < 1000 else i // 1000
            idx = (inst == int(i)).nonzero()
            num = idx.size()[0]
            idx = idx[num // 2, :]
            val = np.zeros((1, feat_num + 1))
            for k in range(feat_num):
                val[0, k] = feat_map[idx[0], idx[1] + k, idx[2],
                                     idx[3]].data[0]
            val[0, feat_num] = float(num) / (h * w // block_num)
            feature[label] = np.append(feature[label], val, axis=0)
        return feature

    def get_edges(self, t):
        edge = torch.cuda.ByteTensor(t.size()).zero_()
        edge[:, :, :,
             1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] !=
                                                   t[:, :, :, :-1])
        edge[:, :,
             1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] !=
                                                   t[:, :, :-1, :])
        if self.opt.data_type == 16:
            return edge.half()
        else:
            return edge.float()

    def save(self, which_epoch):
        self.save_network(self.netG, "G", which_epoch, self.gpu_ids)
        self.save_network(self.netD, "D", which_epoch, self.gpu_ids)
        # if self.gen_features:
        #     self.save_network(self.netE, "E", which_epoch, self.gpu_ids)

    def update_fixed_params(self):
        # after fixing the global generator for a number of iterations, also start finetuning it
        params = list(self.netG.parameters())
        # if self.gen_features:
        #     params += list(self.netE.parameters())
        self.optimizer_G = torch.optim.Adam(params,
                                            lr=self.opt.lr,
                                            betas=(self.opt.beta1, 0.999))
        if self.opt.verbose:
            print(
                "------------ Now also finetuning global generator -----------"
            )

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D.param_groups:
            param_group["lr"] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group["lr"] = lr
        if self.opt.verbose:
            print("update learning rate: %f -> %f" % (self.old_lr, lr))
        self.old_lr = lr


class InferenceModel(Pix2PixHDModel):
    def forward(self, inp):
        label, inst = inp
        return self.inference(label, inst)
