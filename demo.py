# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.models._shared.body_detection.app import BodyDetectionApp
from qai_hub_models.models._shared.body_detection.demo import BodyDetectionDemo
from qai_hub_models.models.gear_guard_net.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    GearGuardNet,
)
# from qai_hub_models.utils.asset_loaders import CachedWebModelAsset
import numpy as np
# print(f"Model ID: {MODEL_ID}, Model Asset Version: {MODE L_ASSET_VERSION}")

INPUT_IMAGE_ADDRESS = "cropped_worker.jpg"
# print(f"Input Image Address: {INPUT_IMAGE_ADDRESS}\n\n")
# Preprocess the image to fit the input resolution: 320x192
from PIL import Image

def preprocess_image(image_path: str, target_size: tuple) -> np.ndarray:
    image = Image.open(image_path)
    image = image.resize(target_size)
    image_array = np.array(image)
    return image_array

preprocessed_image = preprocess_image(INPUT_IMAGE_ADDRESS, (192, 320))
Image.fromarray(preprocessed_image).save("known_face_preprocessed.jpg")

def main(is_test: bool = False):
    BodyDetectionDemo(
        is_test,
        GearGuardNet,
        MODEL_ID,
        BodyDetectionApp,
        "known_face_preprocessed.jpg",
        320,
        192,
        0.9,
    )


if __name__ == "__main__":
    main()