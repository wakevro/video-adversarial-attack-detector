import os
import random
import time
from helpers.AdversarialAttack import AdversarialAttack
from helpers.MediaConverter import MediaConverter


def clear_terminal():
    """
    Clears the terminal screen.
    """
    os.system("clear")


def main():
    start_time = time.time()
    result_dir = "results/"
    output_videos_dir = os.path.join(result_dir, "output_videos")
    original_output_frames_dir = os.path.join(result_dir, "original_output_frames")
    generated_disturbed_output_frames_dir = os.path.join(
        result_dir, "generated_disturbed_output_frames"
    )
    disturbed_output_frames_dir = os.path.join(result_dir, "disturbed_output_frames")
    disturbed_decorated_output_frames_dir = os.path.join(
        result_dir, "disturbed_decorated_output_frames"
    )
    generated_decorated_detection_frames_dir = os.path.join(
        result_dir, "generated_decorated_detection_frames"
    )
    original_video_filepath = "sample_video/video.avi"

    media_converter = MediaConverter()
    adversarial_attack = AdversarialAttack()

    print("\nProcessing original video...")

    # Convert video to frames
    image_vectors = media_converter.convert_video_to_frames(
        video_filepath=original_video_filepath
    )
    original_images_file_paths = media_converter.save_frames_to_folder(
        frames=image_vectors, output_dir=original_output_frames_dir
    )
    resulting_original_video_filepath = media_converter.convert_images_to_video(
        image_paths=original_images_file_paths,
        output_dir=output_videos_dir,
        file_name="resulting_original_video.mp4",
    )

    print(f"Original video saved to {resulting_original_video_filepath}")

    disturbed_image_file_paths = []
    disturbed_decorated_image_file_paths = []
    generated_detection_file_paths = []
    detected_attack_indexes = []
    actual_attack_indexes = []

    print("\nGenerating disturbed video...")

    with open("attacked_indexes.txt", "w") as file:
        for i in range(len(original_images_file_paths[:60])):
            if i % 50 == 0:
                print("")
            else:
                print(".", end="")

            image_file_path = original_images_file_paths[i]

            # Generate disturbances for specific frames
            if True and (i >= 20) and (i <= 30):
                disturbed_image_file_path = adversarial_attack.fgsm_attack(
                    image_path=image_file_path,
                    epsilon=10,
                    output_dir=disturbed_output_frames_dir,
                )
                disturbed_decorated_image_file_path = media_converter.decorate_image(
                    image_path=disturbed_image_file_path,
                    output_dir=disturbed_decorated_output_frames_dir,
                    color=(0, 0, 255),
                    count=i,
                )
                actual_attack_indexes.append(i)
                file.write(str(i) + "\n")
            else:
                disturbed_image_file_path = media_converter.copy_and_paste_file(
                    original_filepath=image_file_path,
                    destination_directory=disturbed_output_frames_dir,
                )
                disturbed_decorated_image_file_path = media_converter.decorate_image(
                    image_path=image_file_path,
                    output_dir=disturbed_decorated_output_frames_dir,
                    color=(0, 255, 0),
                    count=i,
                )

            disturbed_decorated_image_file_paths.append(
                disturbed_decorated_image_file_path
            )
            disturbed_image_file_paths.append(disturbed_image_file_path)

        print(f"\nAttacked images indexes: {actual_attack_indexes}")

    print("Saving disturbed video")

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

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nElapsed time for calculations: {elapsed_time} seconds")


if __name__ == "__main__":
    clear_terminal()
    main()
