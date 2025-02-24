# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from copy import deepcopy

import numpy as np
import PIL.Image as Image
import cv2

from qai_hub_models.models._shared.body_detection.app import BodyDetectionApp
from qai_hub_models.models.protocols import FromPretrainedTypeVar
from qai_hub_models.utils.args import (
    demo_model_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import load_image
from qai_hub_models.utils.display import display_or_save_image
from qai_hub_models.utils.draw import draw_box_from_corners


def plot_result(img: np.ndarray, result: np.ndarray):
    """
    Plot detection result with class numbers.

    Inputs:
        img: np.ndarray
            Input image.
        result: np.ndarray
            Detection result.
    """
    box_color = ((255, 0, 0), (0, 255, 0))
    
    # Define class names
    class_names = {0: "Helmet", 1: "Vest"}
    
    for r in result:
        corners = np.array(
            [[r[1], r[2]], [r[1], r[4]], [r[3], r[2]], [r[3], r[4]]]
        ).astype(int)
        class_id = int(r[0])
        class_name = class_names.get(class_id, f"Class {class_id}")
        draw_box_from_corners(img, corners, box_color[class_id])
        
        # Calculate text position
        text_position = (corners[0][0], max(corners[0][1] - 10, 10))
        
        # Define text and background properties
        text = class_name
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        
        # Calculate background rectangle coordinates
        background_topleft = (text_position[0], text_position[1] - text_size[1] - 5)
        background_bottomright = (text_position[0] + text_size[0] + 5, text_position[1] + 5)
        
        # Draw translucent white rectangle
        overlay = img.copy()
        cv2.rectangle(overlay, background_topleft, background_bottomright, (255, 255, 255), -1)
        alpha = 0.6  # Transparency factor
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        
        # Add black text on top
        cv2.putText(img, text, text_position, font, font_scale, (0, 0, 0), font_thickness)
    return img


def BodyDetectionDemo(
    is_test: bool,
    model_name: type[FromPretrainedTypeVar],
    model_id: str,
    app_name: type[BodyDetectionApp],
    imgfile: str,
    height: int,
    width: int,
    conf: float,
) -> None:
    """
    Object detection demo.

    Input:
        is_test: bool.
            Is test
        model_name: nn.Module
            Object detection model.
        model_id: str.
            Model ID
        app_name: BodyDetectionApp
            Object detection app.
        imgfile: str:
            Image file path.
        height: int
            Input image height.
        width: int
            Input image width.
        conf: float
            Detection confidence.
    """
    parser = get_model_cli_parser(model_name)
    parser = get_on_device_demo_parser(parser, add_output_dir=True)
    parser.add_argument(
        "--image",
        type=str,
        default=imgfile,
        help="image file path or URL",
    )
    args = parser.parse_args([] if is_test else None)
    model = demo_model_from_cli_args(model_name, model_id, args)
    validate_on_device_demo_args(args, model_id)

    app = app_name(model)  # type: ignore[arg-type]
    result = app.detect(args.image, height, width, conf)
    
    # Print the result with spacing
    print("\nDetection Result:")
    # Check if both helmet (class 0) and vest (class 1) are present in the result
    classes_detected = result[:, 0]
    if 0 in classes_detected and 1 in classes_detected:
        print("Both helmet and vest are present!")
    elif 0 in classes_detected:
        print("Only helmet is present!")
    elif 1 in classes_detected:
        print("Only vest is present!")
    else:
        print("Neither helmet nor vest is present!")

    if is_test:
        img = np.array(load_image(args.image))
        image_annotated = plot_result(deepcopy(img), result)
        display_or_save_image(
            Image.fromarray(image_annotated), args.output_dir, "result.jpg"
        )
