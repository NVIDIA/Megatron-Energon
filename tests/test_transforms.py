# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

"""This module tests image transforms."""

import logging
import random
import sys
from typing import Any, Tuple

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

from megatron.energon.transforms import CustomTransform, MergeTransform


@pytest.fixture(autouse=True)
def setup_logging():
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)


def reset_rng(seed: int = 42):
    """Creates a PIL image with random noise."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def get_test_image(width: int, height: int):
    """Creates a PIL image with random noise."""

    arr = np.zeros((width, height, 3), dtype=np.uint8)

    # Some colorful borders
    arr[0, :, :] = [255, 0, 0]
    arr[:, 0, :] = [255, 255, 0]
    arr[-1, :, :] = [255, 255, 255]
    arr[:, -1, :] = [0, 255, 0]

    # A single white pixel
    if width > 3 and height > 3:
        arr[3, 3, :] = [255, 255, 255]

    # And in the middle some noise
    if width > 10 and height > 10:
        arr[5:-5, 5:-5, :] = np.random.randint(0, 255, (width - 10, height - 10, 3))

    return Image.fromarray(arr)


def get_test_image_soft(width: int, height: int):
    """Creates a PIL image smooth content"""

    arr = np.zeros((width, height, 3), dtype=np.uint8)

    # Fill red channel the image with a smooth gradient from left to right.
    arr[:, :, 0] = np.arange(width)[:, None] / width * 255
    # The same for green from top to bottom:
    arr[:, :, 1] = np.arange(height)[None, :] / height * 255

    return Image.fromarray(arr)


def _apply_and_compare(testable_transform, img, atol=2, seed=42, msg=None, only_nonblack=False):
    # Then transform using our method
    merge_transform = MergeTransform([testable_transform])

    reset_rng(seed=seed)
    test_result = merge_transform(img)

    # And also transform using torchvision directly
    reset_rng(seed=seed)
    ref_result = testable_transform(img)

    # Then compare the sizes and the images contents
    assert test_result.size == ref_result.size

    # Check that image contents are close
    np_test = np.array(test_result)
    np_ref = np.array(ref_result)

    if only_nonblack:
        nonblack_mask = (np_test > 0) & (np_ref > 0)
        np_test = np_test[nonblack_mask]
        np_ref = np_ref[nonblack_mask]

    # The maximum allowed difference between pixel values is 2 (uint8)
    assert np.allclose(np_test, np_ref, atol=atol), msg


def test_resize():
    """Tests ResizeMapper"""

    MAX_SIZE = 150
    # These are the different setups we test. Each entry is a tuple of
    # (source size, resize_kwargs)

    size_list = [  # source size (w, h), resize_kwargs
        [(100, 100), {"size": (100, 100)}],
        [(200, 50), {"size": (100, 100)}],
        [(50, 50), {"size": (100, 100)}],
        [(500, 500), {"size": (10, 10)}],
        [(1, 2), {"size": (1, 3)}],  # Scale width by 1.5x
        [(50, 100), {"size": 100, "max_size": MAX_SIZE}],  # Test max_size
    ]

    for source_size, resize_kwargs in size_list:
        logging.info(
            f"Testing Resize with source size {source_size} and resize_kwargs {resize_kwargs}"
        )

        # Create a test image of the given source size
        img = get_test_image(*source_size)
        transform = T.Resize(**resize_kwargs, interpolation=InterpolationMode.NEAREST)

        _apply_and_compare(
            transform,
            img,
            msg=f"Resize: source_size={source_size}, resize_kwargs={resize_kwargs}",
        )


def test_random_resized_crop():
    """Tests RandomResizedCropMapper"""

    randcrop = T.RandomResizedCrop(
        90, scale=(0.3, 0.7), ratio=(0.75, 1.3), interpolation=InterpolationMode.BILINEAR
    )
    source_size = (50, 60)

    logging.info(f"Testing RandomResizedCrop with source size {source_size}")

    # Create a test image of the given source size
    img = get_test_image_soft(*source_size)

    _apply_and_compare(randcrop, img, msg="RandomResizedCrop")


def test_random_flip():
    source_size = (55, 33)
    img = get_test_image(*source_size)

    logging.info("Testing RandomHorizontalFlip 5 times")
    for idx in range(5):
        randhflip = T.RandomHorizontalFlip(p=0.8)
        _apply_and_compare(randhflip, img, seed=idx, msg="RandomHorizontalFlip")

    logging.info("Testing RandomVerticalFlip 5 times")
    for idx in range(5):
        randvflip = T.RandomVerticalFlip(p=0.8)
        _apply_and_compare(randvflip, img, seed=idx, msg="RandomVerticalFlip")


def test_random_rotation():
    source_size = (55, 33)
    img = get_test_image_soft(*source_size)

    logging.info("Testing RandomRotation without expand")
    for idx in range(5):
        randrot = T.RandomRotation((-90, 269), interpolation=InterpolationMode.BILINEAR)
        _apply_and_compare(
            randrot,
            img,
            seed=idx,
            msg="RandomRotation without expand",
        )

    logging.info("Testing RandomRotation with expand")
    for idx in range(5):
        randrot = T.RandomRotation(
            (-180, 269), interpolation=InterpolationMode.BILINEAR, expand=True
        )
        _apply_and_compare(
            randrot,
            img,
            seed=idx,
            msg="RandomRotation with expand",
        )


def test_random_crop():
    source_size = (155, 120)
    img = get_test_image(*source_size)

    size_list = [  # crop size (w, h)
        (155, 120),  # Same size
        (100, 50),
        3,  # Single int as size
        120,
        (155, 8),  # One dimension same size
    ]

    logging.info("Testing RandomCrop")
    for idx, size in enumerate(size_list):
        randcrop = T.RandomCrop(size)
        _apply_and_compare(
            randcrop,
            img,
            seed=idx,
            msg=f"RandomCrop: crop size={size}",
        )

    # Test `pad_if_needed` (Crop size larger than image size)
    randcrop = T.RandomCrop((500, 500), pad_if_needed=True)
    _apply_and_compare(randcrop, img)


def test_random_perspective():
    source_size = (128, 133)
    img = get_test_image_soft(*source_size)

    logging.info("Testing RandomPerspective")
    for idx in range(5):
        randpersp = T.RandomPerspective(interpolation=InterpolationMode.BILINEAR)
        _apply_and_compare(
            randpersp,
            img,
            seed=idx,
            msg=f"RandomPerspective: source_size={source_size}",
            only_nonblack=True,  # Sometimes one pixel is off
        )


def test_center_crop():
    source_size_list = [  # source size (w, h)
        (155, 120),
        (154, 119),
    ]

    crop_size_list = [  # crop size (w, h)
        (155, 120),  # Same size
        (100, 50),
        3,  # Single int as size
        120,
        (200, 50),  # Large than image in x direction
        (50, 200),  # Large than image in y direction
        (200, 200),  # Large than image in both directions
    ]

    logging.info("Testing CenterCrop")

    for source_size in source_size_list:
        img = get_test_image(*source_size)

        for idx, crop_size in enumerate(crop_size_list):
            centcrop = T.CenterCrop(crop_size)
            _apply_and_compare(
                centcrop,
                img,
                seed=idx,
                msg=f"CenterCrop: source_size={source_size}, crop_size={crop_size}",
            )


def test_custom():
    """Tests if a custom transform works"""

    source_size = (128, 133)

    class FixedTranslate(CustomTransform):
        """Translates the image by 5 pixels in both x and y direction"""

        def __init__(self):
            pass

        def apply_transform(self, matrix: np.ndarray, dst_size: np.ndarray) -> Tuple[Any, Any, Any]:
            matrix = self.translate(5, 5) @ matrix
            return matrix, dst_size, (self.__class__.__name__, (5, 5))

    img = get_test_image(*source_size)

    merge_transform = MergeTransform([FixedTranslate()])
    test_result = merge_transform(img)

    reference_img = Image.new(img.mode, img.size, (0, 0, 0))
    reference_img.paste(img, (5, 5))

    assert np.allclose(np.array(test_result), np.array(reference_img), atol=1), "FixedTranslate"


def test_merge():
    """Tests if two merged transforms yield the same result.
    Merging RandomCrop and RandomPerspective."""

    source_size = (128, 133)
    img = get_test_image_soft(*source_size)

    randcrop = T.RandomCrop((70, 70))
    randrot = T.RandomRotation((45, 269), interpolation=InterpolationMode.BILINEAR)

    merge_transform = MergeTransform([randrot, randcrop])
    reset_rng(1)
    test_result = merge_transform(img)

    reset_rng(1)
    ref_result = randcrop(randrot(img))

    assert np.allclose(np.array(test_result), np.array(ref_result), atol=1), (
        "MergeTransform of RandomRotation and RandomCrop"
    )
