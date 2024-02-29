import math
from typing import List, Optional, Tuple, Union

import albumentations as A
import numpy as np
import PIL
from PIL import Image

from ._utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, STR_TO_PIL_INTERP


class ImageNetTransform:
    """ImageNetTransform class

    Args:
        train (bool): whether to use training transform or not
        img_size (Union[int, Tuple[int, int]], optional): the image size. Defaults to 224.
        scale (Optional[Tuple[float, float]], optional): the scale range. Defaults to None.
        ratio (Optional[Tuple[float, float]], optional): the ratio range. Defaults to None.
        crop_mode (str, optional): the crop mode. Defaults to "center".
        crop_pct (float, optional): the crop percentage. Defaults to 0.875.
        hflip (float, optional): the horizontal flip probability. Defaults to 0.5.
        vflip (float, optional): the vertical flip probability. Defaults to 0.
        color_jitter (Optional[Union[float, Tuple[float, float, float, float]]], optional): the color jitter. Defaults to 0.4.
        auto_augment (Optional[str], optional): the auto augment technique. Defaults to None.
        interpolation (str, optional): the interpolation method. Defaults to 'bilinear'.
        mean (Tuple[float, ...], optional): the mean value. Defaults to IMAGENET_DEFAULT_MEAN.
        std (Tuple[float, ...], optional): the standard deviation value. Defaults to IMAGENET_DEFAULT_STD.
    """

    def __init__(
        self,
        train: bool,
        img_size: Union[int, Tuple[int, int]] = 224,
        scale: Optional[Tuple[float, float]] = None,
        ratio: Optional[Tuple[float, float]] = None,
        crop_mode: str = "center",
        crop_pct: float = 0.875,
        hflip: float = 0.5,
        vflip: float = 0.0,
        color_jitter: Optional[Union[float, Tuple[float, float, float, float]]] = 0.4,
        auto_augment: Optional[str] = None,
        interpolation: str = "bilinear",
        mean: Tuple[float, ...] = IMAGENET_DEFAULT_MEAN,
        std: Tuple[float, ...] = IMAGENET_DEFAULT_STD,
    ):
        assert crop_mode in {"center", "squash", "border"}
        self.img_size = img_size if isinstance(img_size, (tuple, list)) else (img_size, img_size)
        self.scale = scale
        self.ratio = ratio
        self.crop_mode = crop_mode
        self.crop_pct = crop_pct
        self.hflip = hflip
        self.vflip = vflip
        self.color_jitter = color_jitter
        self.auto_augment = auto_augment
        self.interpolation = STR_TO_PIL_INTERP[interpolation]
        self.mean = mean
        self.std = std

        if train:
            self.transform = self._train_transform()
        else:
            self.transform = self._val_transform()

    def _primary_transform(self) -> List[A.BasicTransform]:
        """Compose primary transform

        Returns:
            List[A.BasicTransform]: list of albumentations transforms
        """
        _transforms = []
        scale = self.scale if self.scale is not None else (0.08, 1.0)  # default imagenet scale range
        ratio = self.ratio if self.ratio is not None else (3.0 / 4.0, 4.0 / 3.0)  # default imagenet ratio range
        _transforms.append(
            A.RandomResizedCrop(
                height=self.img_size[0],
                width=self.img_size[1],
                scale=scale,
                ratio=ratio,
                interpolation=self.interpolation,
                p=1.0,
            )
        )
        if self.hflip > 0:
            _transforms.append(A.HorizontalFlip(p=self.hflip))
        if self.vflip > 0:
            _transforms.append(A.VerticalFlip(p=self.vflip))

        return _transforms

    def _secondary_transform(self) -> List[A.BasicTransform]:
        """Compose secondary transform

        Returns:
            List[A.BasicTransform]: list of albumentations transforms
        """
        _transforms = []
        disable_color_jitter = False
        # TODO: implement auto_augment techniques: randaugment, augmix, autoagument
        if self.auto_augment is not None:
            pass
        if self.color_jitter is not None and not disable_color_jitter:
            if isinstance(self.color_jitter, (tuple, list)):
                assert len(self.color_jitter) == 4
            else:
                self.color_jitter = (self.color_jitter,) * 4
            _transforms.append(
                A.ColorJitter(
                    brightness=self.color_jitter[0],
                    contrast=self.color_jitter[1],
                    saturation=self.color_jitter[2],
                    hue=self.color_jitter[3],
                    always_apply=True,
                )
            )

        return _transforms

    def _final_transform(self) -> List[A.BasicTransform]:
        """Compose final transform

        Returns:
            List[A.BasicTransform]: list of albumentations transforms
        """
        _transforms = []
        _transforms.append(
            A.Normalize(mean=self.mean, std=self.std, max_pixel_value=255.0, always_apply=True),
            # TODO: add RandomErase
        )

        return _transforms

    def _train_transform(self) -> A.Compose:
        """Setup training transform

        Returns:
            A.Compose: composition of transforms
        """

        _transforms: List[A.BasicTransform] = []
        _primary_transform = self._primary_transform()
        _secondary_transform = self._secondary_transform()
        _final_transform = self._final_transform()

        # TODO: implement separate
        return A.Compose(_primary_transform + _secondary_transform + _final_transform)

    def _val_transform(self) -> A.Compose:
        """Setup validation transform

        Raises:
            NotImplementedError: squash mode not implemented
            NotImplementedError: border mode not implemented

        Returns:
            A.Compose: composition of transforms
        """
        # TODO: verify scaled_img_size
        if isinstance(self.img_size, (tuple, list)):
            assert len(self.img_size) == 2
            scaled_img_size = tuple([math.floor(x / self.crop_pct) for x in self.img_size])
        else:
            scaled_img_size = math.floor(self.img_size / self.crop_pct)  # type: ignore
            scaled_img_size = (scaled_img_size, scaled_img_size)

        # squash mode scales each edge to 1/pct of target, then crops
        # aspect ration is not preserved, no img lost if crop_pct == 1.0

        _transforms = []
        if self.crop_mode == "squash":
            _transforms.append(A.SmallestMaxSize(max_size=max(scaled_img_size), interpolation=self.interpolation))
            _transforms.append(A.CenterCrop(self.img_size[0], self.img_size[1]))
        elif self.crop_mode == "border":
            _transforms.append(
                A.LongestMaxSize(max_size=max(scaled_img_size), interpolation=self.interpolation),
            )
            _transforms.append(
                A.PadIfNeeded(
                    self.img_size[0], self.img_size[1], value=[round(255 * v) for v in self.mean], border_mode=0
                )
            )
            _transforms.append(
                A.CenterCrop(self.img_size[0], self.img_size[1]),
            )
        else:
            if scaled_img_size[0] == scaled_img_size[1]:
                _transforms.append(
                    A.Resize(height=scaled_img_size[0], width=scaled_img_size[1], interpolation=self.interpolation),
                )
            else:
                _transforms.append(
                    A.SmallestMaxSize(max(scaled_img_size), interpolation=self.interpolation),
                )
            _transforms.append(A.CenterCrop(self.img_size[0], self.img_size[1]))
        _transforms.append(A.Normalize())
        return A.Compose(_transforms)

    def __call__(self, image: np.array) -> np.array:  # type: ignore
        """Apply transform to image

        Args:
            image (np.array): the input image

        Returns:
            np.array: the transformed image
        """
        return self.transform(image=image)["image"]
