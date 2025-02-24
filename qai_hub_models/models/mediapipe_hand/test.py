# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import numpy as np

from qai_hub_models.models.mediapipe_hand.app import MediaPipeHandApp
from qai_hub_models.models.mediapipe_hand.demo import INPUT_IMAGE_ADDRESS
from qai_hub_models.models.mediapipe_hand.demo import main as demo_main
from qai_hub_models.models.mediapipe_hand.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    MediaPipeHand,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
from qai_hub_models.utils.testing import skip_clone_repo_check

OUTPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "hand_output.png"
)

# Because we have not made a modification to the pytorch source network,
# no numerical tests are included for the model; only for the app.


@skip_clone_repo_check
def test_hand_app():
    input = load_image(
        INPUT_IMAGE_ADDRESS,
    )
    expected_output = load_image(
        OUTPUT_IMAGE_ADDRESS,
    ).convert("RGB")
    app = MediaPipeHandApp(MediaPipeHand.from_pretrained())
    assert np.allclose(
        app.predict_landmarks_from_image(input)[0], np.asarray(expected_output)
    )


@skip_clone_repo_check
def test_demo():
    demo_main(is_test=True)
