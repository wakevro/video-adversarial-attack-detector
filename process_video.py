import time
from os import system
from helpers.AttackDetector import AttackDetector
from helpers.DataVisualizer import DataVisualizer
from helpers.MediaConverter import MediaConverter

# Clear the terminal screen
system("clear")

if __name__ == "__main__":
    start_time = time.time()

    # Directories for storing results
    result_dir = "results/"
    generated_disturbed_output_frames_dir = (
        result_dir + "generated_disturbed_output_frames"
    )
    generated_disturbed_decorated_output_frames_dir = (
        result_dir + "generated_disturbed_decorated_output_frames"
    )
    output_videos_dir = result_dir + "output_videos/"
    disturbed_video_filepath = output_videos_dir + "disturbed_video.mp4"

    # Initialize helper instances
    media_converter = MediaConverter()
    attack_detector = AttackDetector(contamination=0.2)
    data_visualizer = DataVisualizer()

    # Process disturbed video
    print("\nProcessing disturbed video...")
    generated_disturbed_images_vectors = media_converter.convert_video_to_frames(
        video_filepath=disturbed_video_filepath
    )
    generated_disturbed_images_file_paths = media_converter.save_frames_to_folder(
        frames=generated_disturbed_images_vectors,
        output_dir=generated_disturbed_output_frames_dir,
    )

    # Detect attacks in the processed frames
    print("\nProcessing detection ...")
    with open("attacked_indexes.txt") as f:
        actual_attack_indexes = [int(x) for x in f.readlines()]

    detected, attacked_images_indexes, threshold_list = (
        attack_detector.detect_attack_from_image_paths(
            generated_disturbed_images_file_paths
        )
    )
    print(f"\nAttacked images indexes: {actual_attack_indexes}")
    print(f"\nDetected images indexes: {attacked_images_indexes}")

    # Visualize the detection results
    data_visualizer.visualize_data(
        detected_attack_indexes=attacked_images_indexes,
        actual_attack_indexes=actual_attack_indexes,
        length_of_all_indexes=len(generated_disturbed_images_file_paths),
        threshold_list=threshold_list,
        output_dir=result_dir,
        output_file_name="approach_2_detection_results.pdf",
    )

    # Generate decorated detected video
    print("\nGenerating decorated detected video...")
    generated_disturbed_decorated_image_file_paths = []

    for i in range(len(generated_disturbed_images_file_paths)):
        if i % 50 == 0:
            print("")
        else:
            print(".", end="")

        image_file_path = generated_disturbed_images_file_paths[i]

        if i in attacked_images_indexes and i in actual_attack_indexes:
            color = (0, 255, 0)  # Green for correct detection
        elif (i in attacked_images_indexes and i not in actual_attack_indexes) or (
            i not in attacked_images_indexes and i in actual_attack_indexes
        ):
            color = (0, 0, 255)  # Red for false positive or false negative
        else:
            color = (0, 255, 0)  # Green for no attack detected

        generated_disturbed_decorated_image_file_path = media_converter.decorate_image(
            image_path=image_file_path,
            output_dir=generated_disturbed_decorated_output_frames_dir,
            color=color,
            count=i,
        )
        generated_disturbed_decorated_image_file_paths.append(
            generated_disturbed_decorated_image_file_path
        )

    # Save the detected decorated video
    print("\nSaving detected decorated video")
    generated_disturbed_decorated_video_filepath = (
        media_converter.convert_images_to_video(
            image_paths=generated_disturbed_decorated_image_file_paths,
            output_dir=output_videos_dir,
            file_name="generated_decorated_detected_video.mp4",
        )
    )
    print(
        "Detected decorated video saved to "
        + generated_disturbed_decorated_video_filepath
    )

    # Print elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nElapsed time for calculations: {elapsed_time} seconds")
