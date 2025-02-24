# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import json
from pathlib import Path
import cv2
import numpy as np

from qai_hub_models.models.face_det_lite.app import FaceDetLiteApp
from qai_hub_models.models.face_det_lite.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    FaceDetLite_model,
)
from qai_hub_models.utils.args import (
    demo_model_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image

INPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "test_640x480_Rooney.jpg"
)


# Run face_det_lite model end-to-end on a sample image.
# The demo will output the face bounding boxes in json files
# the bounding box represented by left, top, width, and height.
def main(
    model_cls: type[FaceDetLite_model] = FaceDetLite_model,
    model_id: str = MODEL_ID,
    is_test: bool = False,
):
    # Demo parameters
    parser = get_model_cli_parser(model_cls)
    parser = get_on_device_demo_parser(parser, add_output_dir=True)
    parser.add_argument(
        "--image",
        type=str,
        default=INPUT_IMAGE_ADDRESS,
        help="image file path or URL",
    )
    # args = parser.parse_args([] if is_test else None)
    args = parser.parse_args([])
    model = demo_model_from_cli_args(model_cls, MODEL_ID, args)
    validate_on_device_demo_args(args, MODEL_ID)

    # Load image
    (_, _, height, width) = FaceDetLite_model.get_input_spec()["input"][0]
    orig_image = load_image(args.image)

    # Ensure the image is a NumPy array
    if not isinstance(orig_image, np.ndarray):
        orig_image = np.array(orig_image)

    print("Model Loaded")

    app = FaceDetLiteApp(model)
    res = app.run_inference_on_image(orig_image)
    out_dict = {}

    # Assuming `res` is a list of bounding boxes, each represented as (left, top, width, height)
    for bbox in res:
        left, top, width, height, confidence = bbox
        # Draw the bounding box on the image
        cv2.rectangle(orig_image, (left, top), (left + width, top + height), (0, 255, 0), 2)

    out_dict["bounding box"] = str(res)

    # Convert the image from BGR to RGB
    orig_image_rgb = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)

    # Save or display the image with bounding boxes
    output_image_path = "output_image_with_bboxes.jpg"
    cv2.imwrite(output_image_path, orig_image_rgb)
    print(f"Image with bounding boxes saved at: {output_image_path}")

    if not is_test:
        output_path = (
            args.output_dir or str(Path() / "build")
        ) + "/FaceDetLitebNet_output.json"

        with open(output_path, "w", encoding="utf-8") as wf:
            json.dump(out_dict, wf, ensure_ascii=False, indent=4)
        print(f"Model outputs are saved at: {output_path}")
