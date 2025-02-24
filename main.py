import cv2
import argparse
import warnings
import numpy as np
import pickle
import os
import subprocess

from models import SCRFD, ArcFace
from utils.helpers import Face, draw_face_info

warnings.filterwarnings("ignore")


def load_models(detection_model_path: str, attribute_model_path: str):
    """Loads the detection and attribute models.
    Args:
        detection_model_path (str): Path to the detection model file.
        attribute_model_path (str): Path to the attribute model file.
    Returns
        tuple: A tuple containing the detection model and the attribute model.

    """
    try:
        detection_model = SCRFD(model_path=detection_model_path)
        attribute_model = ArcFace(model_path=attribute_model_path)
    except Exception as e:
        print(f"Error loading models: {e}")
        raise
    return detection_model, attribute_model


def inference_image(detection_model, attribute_model, image_path, save_output, new_user=None):
    """Processes a single image for face detection and attributes.
    Args:
        detection_model (SCRFD): The face detection model.
        attribute_model (Attribute): The attribute detection model.
        image_path (str): Path to the input image.
        save_output (str): Path to save the output image.
    """
    frame = cv2.imread(image_path)
    if frame is None:
        print("Failed to load image")
        return

    process_frame(detection_model, attribute_model, frame, new_user)
    if save_output:
        # print(frame)
        cv2.imwrite(save_output, frame)
    # cv2.imshow("FaceDetection", frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def inference_video(detection_model, attribute_model, video_source, save_output):
    """Processes a video source for face detection and attributes.
    Args:
        detection_model (SCRFD): The face detection model.
        attribute_model (Attribute): The attribute detection model.
        video_source (str or int): Path to the input video file or camera index.
        save_output (str): Path to save the output video.
    """
    if video_source.isdigit() or video_source == '0':
        cap = cv2.VideoCapture(int(video_source))
    else:
        cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print("Failed to open video source")
        return

    out = None
    if save_output:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(save_output, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        process_frame(detection_model, attribute_model, frame)
        if save_output:
            out.write(frame)

        cv2.imshow("FaceDetection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if save_output:
        out.release()
    cv2.destroyAllWindows()


def save_embeddings(embeddings, file_path):
    """Saves embeddings to a file using pickle.
    Args:
        embeddings (list): The list of embeddings to save.
        file_path (str): The path to the file where embeddings will be saved.
    """
    with open(file_path, 'wb') as f:
        pickle.dump(embeddings, f)


def process_frame(detection_model, attribute_model, frame, new_user=None):
    """Detects faces and attributes in a frame and draws the information.
    Args:
        detection_model (SCRFD): The face detection model.
        attribute_model (Attribute): The attribute detection model.
        frame (np.ndarray): The image frame to process.
    """
    boxes_list, points_list = detection_model.detect(frame)
    embeddings_list = []  # List to store embeddings

    for boxes, keypoints in zip(boxes_list, points_list):
        *bbox, conf_score = boxes
        embedding = attribute_model(frame, keypoints)
        embeddings_list.append(embedding)  # Append embedding to list
        face = Face(kps=keypoints, bbox=bbox, name=new_user)
        draw_face_info(frame, face)

    # Save embeddings to a file
    if new_user and len(embeddings_list) == 1:
        # Create a dictionary with new_user as the key
        user_embeddings = {new_user: embeddings_list}
        save_embeddings(user_embeddings, 'embeddings.pkl')
    # Load known embeddings from file
    if os.path.exists('embeddings.pkl'):
        with open('embeddings.pkl', 'rb') as f:
            known_embeddings = pickle.load(f)
    else:
        known_embeddings = {}

    # Update known_embeddings with new user's embeddings
    if new_user and len(embeddings_list) == 1:
        known_embeddings[new_user] = embeddings_list
        save_embeddings(known_embeddings, 'embeddings.pkl')
    # Calculate cosine similarity between each embedding and known embeddings
    for user, embeddings in known_embeddings.items():
        for embedding in embeddings_list:
            similarities = [cosine_similarity(embedding, known_emb) for known_emb in embeddings]
            for idx, similarity in enumerate(similarities):
                if similarity > 0.5:
                    print(f"{user} is checked in")
                else:
                    print(f"Cosine similarity with {user}'s embedding {idx}: {similarity}")


def cosine_similarity(test_embedding, known_embeddings):
    """Calculates the cosine similarity between two embeddings.
    Args:
        test_embedding (np.ndarray): The first embedding vector.
        known_embeddings (np.ndarray): The second embedding vector.
    Returns:
        float: The cosine similarity between the two embeddings.
    """
    dot_product = np.dot(test_embedding, known_embeddings)
    norm1 = np.linalg.norm(test_embedding)
    norm2 = np.linalg.norm(known_embeddings)
    similarity = dot_product / (norm1 * norm2)
    return similarity


def run_demo(image_path):
    """Runs demo.py with the specified image."""
    try:
        subprocess.run(['python', 'demo.py', '--image', image_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running demo.py: {e}")


def run_face_analysis(detection_weights, attribute_weights, input_source, save_output=None, new_user=None):
    """Runs face detection on the given input source."""
    # Run demo.py with the image argument
    if isinstance(input_source, str) and input_source.lower().endswith(('.jpg', '.png', '.jpeg')):
        run_demo(input_source)

    detection_model, attribute_model = load_models(detection_weights, attribute_weights)

    if isinstance(input_source, str) and input_source.lower().endswith(('.jpg', '.png', '.jpeg')):
        inference_image(detection_model, attribute_model, input_source, save_output, new_user)
    else:
        inference_video(detection_model, attribute_model, input_source, save_output)


def main():
    """Main function to run face detection from command line."""
    parser = argparse.ArgumentParser(description="Run face detection on an image or video")
    parser.add_argument(
        '--detection-weights',
        type=str,
        default="weights/optimized_det_10g_onnx_mqy17r97q.onnx",
        help='Path to the detection model weights file'
    )
    parser.add_argument(
        '--attribute-weights',
        type=str,
        default="weights/optimized_arc_onnx_mn06vdp9n.onnx",
        help='Path to the attribute model weights file'
    )
    parser.add_argument(
        '--source',
        type=str,
        default="assets/worker.jpg",
        help='Path to the input image or video file or camera index (0, 1, ...)'
    )
    parser.add_argument('--output', type=str, help='Path to save the output image or video')
    parser.add_argument(
        '--new-user',
        type=str,
        help='Name of the new user'
    )
    args = parser.parse_args()
    run_face_analysis(args.detection_weights, args.attribute_weights, args.source, args.output, args.new_user)

if __name__ == "__main__":
    main()
