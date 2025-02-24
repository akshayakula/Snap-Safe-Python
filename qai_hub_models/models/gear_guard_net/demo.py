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
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset
print(f"Model ID: {MODEL_ID}, Model Asset Version: {MODEL_ASSET_VERSION}")

INPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "test_image.jpg"
)

# INPUT_IMAGE_ADDRESS = "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/apidoc/input_image1.jpg"
print(f"Input Image Address: {INPUT_IMAGE_ADDRESS}\n\n")

def main(is_test: bool = False):
    BodyDetectionDemo(
        is_test,
        GearGuardNet,
        MODEL_ID,
        BodyDetectionApp,
        INPUT_IMAGE_ADDRESS,
        320,
        192,
        0.9,
    )


if __name__ == "__main__":
    main()
