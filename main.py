import os
import random
import time
import concurrent.futures
from os import system
from helpers.MediaConverter import MediaConverter
from helpers.AdversarialAttack import AdversarialAttack
from helpers.AttackDetector import AttackDetector
from helpers.DataVisualizer import DataVisualizer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Clear the terminal screen
system("clear")


def process_frame(
    i: int,
    image_file_path: str,
    disturbed_image_file_paths: list,
    disturbed_decorated_image_file_paths: list,
    actual_attack_indexes: list,
    media_converter: MediaConverter,
    adversarial_attack: AdversarialAttack,
):
    """
    Process each frame of the video, apply attacks if necessary, and decorate the images.

    Args:
    - i: Index of the frame
    - image_file_path: Path of the original image
    - disturbed_image_file_paths: List to store paths of disturbed images
    - disturbed_decorated_image_file_paths: List to store paths of disturbed and decorated images
    - actual_attack_indexes: List to store indexes of frames where attacks were applied
    - media_converter: Instance of the MediaConverter class
    - adverserial_attack: Instance of the AdverserialAttack class

    Returns:
    - Tuple containing index, disturbed image file path, and disturbed decorated image file path
    """
    # Progress indicator
    if i % 50 == 0:
        print("")
    else:
        print(".", end="")

    if i >= 50 and i <= 100:
        # Apply adversarial attack
        disturbed_image_file_path = adversarial_attack.fgsm_attack(
            image_path=image_file_path,
            epsilon=5,
            output_dir=disturbed_output_frames_dir,
        )

        # Decorate the disturbed image
        disturbed_decorated_image_file_path = media_converter.decorate_image(
            image_path=disturbed_image_file_path,
            output_dir=disturbed_decorated_output_frames_dir,
            color=(0, 0, 255),
            count=i,
        )

        disturbed_decorated_image_file_paths.append(disturbed_decorated_image_file_path)
        disturbed_image_file_paths.append(disturbed_image_file_path)
        actual_attack_indexes.append(i)
    else:
        # Copy and paste the original image
        disturbed_image_file_path = media_converter.copy_and_paste_file(
            original_filepath=image_file_path,
            destination_directory=disturbed_output_frames_dir,
        )

        # Decorate the disturbed image
        disturbed_decorated_image_file_path = media_converter.decorate_image(
            image_path=image_file_path,
            output_dir=disturbed_decorated_output_frames_dir,
            color=(0, 255, 0),
            count=i,
        )

        disturbed_decorated_image_file_paths.append(disturbed_decorated_image_file_path)
        disturbed_image_file_paths.append(disturbed_image_file_path)

    return i, disturbed_image_file_path, disturbed_decorated_image_file_path


def detect_attack(
    i: int,
    original_image_path: str,
    disturbed_image_path: str,
    detected_attack_indexes: list,
    generated_detection_file_paths: list,
    media_converter: MediaConverter,
    attack_detector: AttackDetector,
):
    """
    Detect attacks between original and disturbed frames.

    Args:
    - i: Index of the frame
    - original_image_path: Path of the original image
    - disturbed_image_path: Path of the disturbed image
    - detected_attack_indexes: List to store indexes of frames where attacks were detected
    - generated_detection_file_paths: List to store paths of generated detection images
    - media_converter: Instance of the MediaConverter class
    - attack_detector: Instance of the AttackDetector class
    """
    # Detect attack
    attacked = attack_detector.detect_attack_given_two_paths(
        main_image_path=original_image_path, second_image_path=disturbed_image_path
    )

    # Progress indicator
    if i % 50 == 0:
        print("")
    else:
        print(".", end="")

    if attacked:
        # If attack detected, decorate the image in red
        detected_attack_indexes.append(i)
        file_path = media_converter.decorate_image(
            image_path=disturbed_image_path,
            output_dir=generated_decorated_detection_frames_dir,
            color=(0, 0, 255),
            count=i,
        )
        generated_detection_file_paths.append(file_path)
    else:
        # If no attack detected, decorate the image in green
        file_path = media_converter.decorate_image(
            image_path=disturbed_image_path,
            output_dir=generated_decorated_detection_frames_dir,
            color=(0, 255, 0),
            count=i,
        )
        generated_detection_file_paths.append(file_path)


if __name__ == "__main__":
    start_time = time.time()

    # Directories
    result_dir = "results/"
    output_videos_dir = os.path.join(result_dir, "output_videos")
    original_output_frames_dir = os.path.join(result_dir, "original_output_frames")
    disturbed_output_frames_dir = os.path.join(result_dir, "disturbed_output_frames")
    disturbed_decorated_output_frames_dir = os.path.join(
        result_dir, "disturbed_decorated_output_frames"
    )
    generated_decorated_detection_frames_dir = os.path.join(
        result_dir, "generated_decorated_detection_frames"
    )

    # Input video
    original_video_filepath = "sample_video/video.avi"

    # Initialize instances
    media_converter = MediaConverter()
    adversarial_attack = AdversarialAttack()
    attack_detector = AttackDetector()
    data_visualizer = DataVisualizer()

    # Process original video
    print("\nProcessing original video...")
    images_vectors = media_converter.convert_video_to_frames(
        video_filepath=original_video_filepath
    )
    original_images_file_paths = media_converter.save_frames_to_folder(
        frames=images_vectors, output_dir=original_output_frames_dir
    )
    resulting_original_video_filepath = media_converter.convert_images_to_video(
        image_paths=original_images_file_paths,
        output_dir=output_videos_dir,
        file_name="resulting_original_video.mp4",
    )
    print(f"Original video saved to {resulting_original_video_filepath}")

    # Process disturbed video
    disturbed_image_file_paths = []
    disturbed_decorated_image_file_paths = []
    generated_detection_file_paths = []
    detected_attack_indexes = []
    actual_attack_indexes = []

    print("\nProcessing disturbed video...")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                process_frame,
                i,
                image_file_path,
                disturbed_image_file_paths,
                disturbed_decorated_image_file_paths,
                actual_attack_indexes,
                media_converter,
                adversarial_attack,
            )
            for i, image_file_path in enumerate(original_images_file_paths)
        ]

    # Ensure correct order after parallel processing
    results = sorted(
        [future.result() for future in concurrent.futures.as_completed(futures)],
        key=lambda x: x[0],
    )

    disturbed_image_file_paths = [result[1] for result in results]
    disturbed_decorated_image_file_paths = [result[2] for result in results]
    print(f"\nSaving disturbed video")

    disturbed_video_filepath = media_converter.convert_images_to_video(
        image_paths=disturbed_image_file_paths,
        output_dir=output_videos_dir,
        file_name="disturbed_video.mp4",
    )
    disturbed_decorated_video_filepath = media_converter.convert_images_to_video(
        image_paths=disturbed_decorated_image_file_paths,
        output_dir=output_videos_dir,
        file_name="disturbed_decorated_video.mp4",
    )

    print("Disturbed video saved to " + disturbed_video_filepath)

    # Detect attacks
    if len(disturbed_image_file_paths) != len(original_images_file_paths):
        print("\nUnable to run detection. Unequal amount of frames.")
    else:
        print("\nProcessing detection video...")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    detect_attack,
                    i,
                    original_images_file_paths[i],
                    disturbed_image_file_paths[i],
                    detected_attack_indexes,
                    generated_detection_file_paths,
                    media_converter,
                    attack_detector,
                )
                for i in range(len(original_images_file_paths))
            ]

        print("\nDetected attacks:", detected_attack_indexes)

        # Save detection video
        generated_detection_video_filepath = media_converter.convert_images_to_video(
            image_paths=generated_detection_file_paths,
            output_dir=output_videos_dir,
            file_name="generated_detection_video.mp4",
        )

        print(f"Detection video saved to {generated_detection_video_filepath}")

        print("\nSaving data into pdf.")
        # Save all plots in a single PDF
        data_visualizer.visualize_data(
            detected_attack_indexes=detected_attack_indexes,
            actual_attack_indexes=actual_attack_indexes,
            length_of_all_indexes=len(original_images_file_paths),
            output_dir=result_dir,
            output_file_name="parallel_detection_results.pdf",
        )
        print("Done saving data into pdf.")

    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"\nElapsed time for calculations: {elapsed_time} seconds")
    print("Done.")
