import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image


class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        ### input A (label maps)
        self.dir_label = os.path.join(opt.dataroot, opt.phase + "_label")
        self.label_paths = sorted(make_dataset(self.dir_label))

        ### input B (real images)
        if opt.isTrain or opt.use_encoded_image:
            self.dir_image = os.path.join(opt.dataroot, opt.phase + "_img")
            self.image_paths = sorted(make_dataset(self.dir_image))

        ### instance maps
        if not opt.no_instance:
            self.dir_inst = os.path.join(opt.dataroot, opt.phase + "_inst")
            self.inst_paths = sorted(make_dataset(self.dir_inst))

        ### load precomputed instance-wise encoded features
        if opt.load_features:
            self.dir_feat = os.path.join(opt.dataroot, opt.phase + "_feat")
            print("----------- loading features from %s ----------" % self.dir_feat)
            self.feat_paths = sorted(make_dataset(self.dir_feat))

        ### load face bounding box coordinates size 128x128
        # if opt.face_discrim or opt.face_generator:
        #     self.dir_facetext = os.path.join(opt.dataroot, opt.phase + '_facetexts128')
        #     print('----------- loading face bounding boxes from %s ----------' % self.dir_facetext)
        #     self.facetext_paths = sorted(make_dataset(self.dir_facetext))

        self.dataset_size = len(self.label_paths)

    def __getitem__(self, index):
        ### input A (label maps)
        label_path = self.label_paths[index]
        label = Image.open(label_path).convert("RGB")
        params = get_params(self.opt, label.size)
        transform_label = get_transform(
            self.opt, params, method=Image.NEAREST, normalize=False
        )
        label_tensor = transform_label(label).float()
        original_label_path = label_path

        # image_tensor = inst_tensor = feat_tensor = 0
        image_tensor = next_label = next_image = face_tensor = 0

        ### input B (real images)
        if self.opt.isTrain or self.opt.use_encoded_image:
            image_path = self.image_paths[index]
            image = Image.open(image_path).convert("RGB")
            transform_image = get_transform(self.opt, params)
            image_tensor = transform_image(image).float()

        ### if using instance maps
        # if not self.opt.no_instance:
        #     inst_path = self.inst_paths[index]
        #     inst = Image.open(inst_path)
        #     inst_tensor = transform_A(inst)

        #     if self.opt.load_features:
        #         feat_path = self.feat_paths[index]
        #         feat = Image.open(feat_path).convert('RGB')
        #         norm = normalize()
        #         feat_tensor = norm(transform_A(feat))

        is_next = index < len(self) - 1
        # if self.opt.gestures:
        #     is_next = is_next and (index % 64 != 63)

        """ Load the next label, image pair """
        if is_next:

            paths = self.label_paths
            label_path = paths[index + 1]
            label = Image.open(label_path).convert("RGB")
            # params = get_params(self.opt, label.size)
            transform_label = get_transform(
                self.opt, params, method=Image.NEAREST, normalize=False
            )
            next_label = transform_label(label).float()

            if self.opt.isTrain:
                image_path = self.image_paths[index + 1]
                image = Image.open(image_path).convert("RGB")
                transform_image = get_transform(self.opt, params)
                next_image = transform_image(image).float()

        """ If using the face generator and/or face discriminator """
        # if self.opt.face_discrim or self.opt.face_generator:
        #     facetxt_path = self.facetext_paths[index]
        #     facetxt = open(facetxt_path, "r")
        #     face_tensor = torch.IntTensor(list([int(coord_str) for coord_str in facetxt.read().split()]))

        input_dict = {
            "label": label_tensor,
            "image": image_tensor,
            "path": original_label_path,
            "face_coords": face_tensor,
            "next_label": next_label,
            "next_image": next_image,
        }
        # input_dict = {'label': label_tensor, 'inst': inst_tensor, 'image': image_tensor,
        #               'feat': feat_tensor, 'path': label_path}

        return input_dict

    def __len__(self):
        return len(self.label_paths)

    def name(self):
        return "AlignedDataset"
