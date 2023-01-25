# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from numpy.lib import pad
import torch
from torch import nn
from torch.nn import functional as F
from random import randint

from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.structures import ImageList, Instances, Boxes
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n

from ..backbone import Backbone, build_backbone
from ..postprocessing import detector_postprocess
from ..proposal_generator import build_proposal_generator
from ..roi_heads import build_roi_heads
from .build import META_ARCH_REGISTRY
from ..backbone.clipcap.clipcap import unsupervised_loss, unsupervised_feature_loss, generate_feature_caption

__all__ = ["GeneralizedRCNN", "ProposalNetwork"]

from torchvision.transforms import Resize


@META_ARCH_REGISTRY.register()
class GeneralizedRCNN(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    @configurable
    def __init__(
            self,
            *,
            backbone: Backbone,
            proposal_generator: nn.Module,
            roi_heads: nn.Module,
            pixel_mean: Tuple[float],
            pixel_std: Tuple[float],
            input_format: Optional[str] = None,
            vis_period: int = 0,
            use_clip_c4: False,
            use_clip_attpool: False,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super().__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
                self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"
        if np.sum(pixel_mean) < 3.0:  # converrt pixel value to range [0.0, 1.0] by dividing 255.0
            assert input_format == 'RGB'
            self.div_pixel = True
        else:  # default setting
            self.div_pixel = False
        self.use_clip_c4 = use_clip_c4  # if True, use C4 mode where roi_head uses the last resnet layer from backbone
        self.use_clip_attpool = use_clip_attpool  # if True (C4+text_emb_as_classifier), use att_pool to replace default mean pool

        # self.clipcap_model = ClipCaptionModel(40, 40)
        # p = torch.load('/Users/sinamalakouti/Desktop/test-regionclip/transformer_weights.pt', 'cpu')
        # self.clipcap_model.load_state_dict(p)

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "use_clip_c4": cfg.MODEL.BACKBONE.NAME == "build_clip_resnet_backbone",
            "use_clip_attpool": cfg.MODEL.ROI_HEADS.NAME == 'CLIPRes5ROIHeads' and cfg.MODEL.CLIP.USE_TEXT_EMB_CLASSIFIER,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 top-scoring predicted
        object proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch

    def preprocess_image_train(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)

        images_t = [x["image_trgt"].to(self.device) for x in batched_inputs]
        images_t = [(x - self.pixel_mean) / self.pixel_std for x in images_t]
        images_t = ImageList.from_tensors(images_t, self.backbone.size_divisibility)

        resizer = Resize((224, 224))
        images = resizer(images.tensor)
        images_t = resizer(images_t.tensor)
        return images, images_t

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]], clipcap_model=None, branch='supervised'):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        torch.cuda.empty_cache()
        if not self.training:
            return self.inference(batched_inputs)
        if branch == 'caption_consistency':
            images_src, images_target = self.preprocess_image_train(batched_inputs)

            if images_src.shape != images_target.shape:
                print("gooz11")
                print(images_src.shape)
                print(images_target.shape)
                print(self.training)



            prefix_src = self.backbone.attnpool(self.backbone(images_src)['res5'])
            teacher_features = generate_feature_caption(prefix_src, clipcap_model.to(self.device), 40).detach()
            teacher_features = torch.stack(teacher_features, 0)

            prefix_trgt = self.backbone.attnpool(self.backbone(images_target)['res5'])
            student_features = generate_feature_caption(prefix_trgt, clipcap_model.to(self.device), 40)
            student_features = torch.stack(student_features, 0)

            # loss, captions = unsupervised_loss(prefix_src, prefix_trgt, clipcap_model.to(self.device), 40)
            # loss, captions = unsupervised_feature_loss(prefix_src, prefix_trgt, clipcap_model.to(self.device), 40)


            teacher_features = teacher_features.squeeze(1).detach()
            student_features = student_features.squeeze(1)
            del images_src
            # del prefix_src
            del images_target
            del batched_inputs




            teacher_features = (teacher_features / teacher_features.norm(dim=1, keepdim=True)).detach()
            student_features = student_features / student_features.norm(dim=1, keepdim=True)

            if student_features.shape != teacher_features.shape:
                print("jizzzzzz")
                print(student_features.shape)
                print(teacher_features.shape)
                print(self.training)
            joint_features = student_features @ teacher_features.t()
            n = 4
            ground_truth = torch.arange(n, dtype=torch.long, device=self.device)
            # print("n isssssssssssss   ", n)
            # print(ground_truth)
            # print(joint_features.shape)

            loss_fn = nn.CrossEntropyLoss()
            # print("EROORR" * 10)
            # print(ground_truth.shape)
            # print(joint_features.shape)
            loss = loss_fn(joint_features, ground_truth)

            return loss

        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
        # eg: {'p2': torch.Size([b, c, 200, 304]), 'p3': torch.Size([b, c, 100, 152]), 'p4': torch.Size([b, c, 50, 76]), 'p5': torch.Size([b, c, 25, 38]), 'p6': torch.Size([b, c, 13, 19])}
        features = self.backbone(images.tensor)
        if self.proposal_generator is not None:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        if self.use_clip_c4:  # use C4 + resnet weights from CLIP
            if self.use_clip_attpool:  # use att_pool from CLIP to match dimension
                _, detector_losses = self.roi_heads(images, features, proposals, gt_instances,
                                                    res5=self.backbone.layer4, attnpool=self.backbone.attnpool)
            else:  # use default mean pool
                _, detector_losses = self.roi_heads(images, features, proposals, gt_instances,
                                                    res5=self.backbone.layer4)
        else:  # default setting
            _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def inference(
            self,
            batched_inputs: List[Dict[str, torch.Tensor]],
            detected_instances: Optional[List[Instances]] = None,
            do_postprocess: bool = True,
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator is not None:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            if self.use_clip_c4:  # use C4 + resnet weights from CLIP
                if self.use_clip_attpool:  # use att_pool from CLIP to match dimension
                    results, _ = self.roi_heads(images, features, proposals, None, res5=self.backbone.layer4,
                                                attnpool=self.backbone.attnpool)
                else:  # use default mean pool
                    results, _ = self.roi_heads(images, features, proposals, None, res5=self.backbone.layer4)
            else:  # default setting
                results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]

            if self.use_clip_c4:  # use C4 + resnet weights from CLIP
                if self.use_clip_attpool:  # use att_pool from CLIP to match dimension
                    results = self.roi_heads.forward_with_given_boxes(features, detected_instances,
                                                                      res5=self.backbone.layer4,
                                                                      attnpool=self.backbone.attnpool)
                else:  # use default mean pool
                    results = self.roi_heads.forward_with_given_boxes(features, detected_instances,
                                                                      res5=self.backbone.layer4)
            else:  # default setting
                results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        # visualize_proposals(batched_inputs, proposals, self.input_format)
        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results

    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        if self.div_pixel:
            images = [((x / 255.0) - self.pixel_mean) / self.pixel_std for x in images]
        else:
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    @staticmethod
    def _postprocess(instances, batched_inputs: List[Dict[str, torch.Tensor]], image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
                instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results


@META_ARCH_REGISTRY.register()
class ProposalNetwork(nn.Module):
    """
    A meta architecture that only predicts object proposals.
    """

    @configurable
    def __init__(
            self,
            *,
            backbone: Backbone,
            proposal_generator: nn.Module,
            pixel_mean: Tuple[float],
            pixel_std: Tuple[float],
            input_format: Optional[str] = None,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
        """
        super().__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        if np.sum(pixel_mean) < 3.0:  # converrt pixel value to range [0.0, 1.0] by dividing 255.0
            assert input_format == 'RGB'
            self.div_pixel = True
        else:  # default setting
            self.div_pixel = False

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            Same as in :class:`GeneralizedRCNN.forward`

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "proposals" whose value is a
                :class:`Instances` with keys "proposal_boxes" and "objectness_logits".
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        if self.div_pixel:
            images = [((x / 255.0) - self.pixel_mean) / self.pixel_std for x in images]
        else:
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        features = self.backbone(images.tensor)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
        proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        # In training, the proposals are not useful at all but we generate them anyway.
        # This makes RPN-only models about 5% slower.
        if self.training:
            return proposal_losses

        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
                proposals, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"proposals": r})
        return processed_results
