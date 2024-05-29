import cv2
import os
import shutil
from natsort import natsorted


class MediaConverter:
    """
    A class to handle various media conversion tasks such as converting videos to frames,
    converting images to videos, adding borders to images, and copying files.
    """

    def __init__(self):
        self.current_fps = 30

    def log(self, message):
        """
        Logs a message with the class name.

        Args:
            message (str): The message to log.
        """
        print(f"{self.__class__.__name__}: {message}")

    def newLine(self):
        """
        Prints a new line for better readability in logs.
        """
        print("\n")

    def convert_video_to_frames(self, video_filepath):
        """
        Converts a video to individual frames.

        Args:
            video_filepath (str): Path to the input video file.

        Returns:
            list: A list of frames (as numpy arrays) extracted from the video.
        """
        cam = cv2.VideoCapture(video_filepath)
        fps = cam.get(cv2.CAP_PROP_FPS)
        self.current_fps = fps
        self.log(f"Current video FPS: {self.current_fps}")
        frames = []

        while True:
            ret, frame = cam.read()
            if ret:
                frames.append(frame)
            else:
                break

        cam.release()
        return frames

    def convert_images_to_video(self, image_paths, output_dir, file_name):
        """
        Converts a list of images to a video.

        Args:
            image_paths (list): List of paths to the input images.
            output_dir (str): Directory to save the output video.
            file_name (str): Name of the output video file.

        Returns:
            str: The name of the output video file.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        images = [cv2.imread(img_path) for img_path in image_paths]
        height, width, _ = images[0].shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video = cv2.VideoWriter(
            os.path.join(output_dir, file_name),
            fourcc,
            self.current_fps,
            (width, height),
        )

        for image in images:
            video.write(image)

        cv2.destroyAllWindows()
        video.release()

        return file_name

    def save_frames_to_folder(self, frames, output_dir):
        """
        Saves a list of frames to a specified directory.

        Args:
            frames (list): List of frames (as numpy arrays) to save.
            output_dir (str): Directory to save the frames.

        Returns:
            list: List of file paths to the saved frames.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        file_paths = [
            os.path.join(output_dir, f"frame{i+1}.jpg") for i in range(len(frames))
        ]

        for i in range(len(frames)):
            cv2.imwrite(file_paths[i], frames[i])

        return file_paths

    def add_border(self, image_filepath, border_color):
        """
        Adds a border to an image.

        Args:
            image_filepath (str): Path to the input image file.
            border_color (tuple): Color of the border (B, G, R).

        Returns:
            numpy.ndarray: Image with the added border.
        """
        image = cv2.imread(image_filepath)
        original_height, original_width = image.shape[:2]
        border_thickness = 10

        image_with_border = cv2.copyMakeBorder(
            image,
            top=border_thickness,
            bottom=border_thickness,
            left=border_thickness,
            right=border_thickness,
            borderType=cv2.BORDER_CONSTANT,
            value=border_color,
        )

        resized_image = cv2.resize(image_with_border, (original_width, original_height))

        return resized_image

    def decorate_image(self, image_path, output_dir, color, count):
        """
        Adds a border to an image and saves it to a specified directory.

        Args:
            image_path (str): Path to the input image file.
            output_dir (str): Directory to save the decorated image.
            color (tuple): Color of the border (B, G, R).
            count (int): Index for the decorated image file name.

        Returns:
            str: Path to the saved decorated image.
        """
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
            except OSError:
                pass

        updated_image = self.add_border(image_path, color)
        file_path = os.path.join(output_dir, f"decorated_frame{count + 1}.jpg")
        cv2.imwrite(file_path, updated_image)

        return file_path

    def copy_and_paste_file(self, original_filepath, destination_directory):
        """
        Copies a file to a specified directory.

        Args:
            original_filepath (str): Path to the original file.
            destination_directory (str): Directory to copy the file to.

        Returns:
            str: Path to the copied file.
        """
        if not os.path.exists(destination_directory):
            try:
                os.makedirs(destination_directory)
            except OSError:
                pass

        try:
            filename = os.path.basename(original_filepath)
            new_filepath = os.path.join(destination_directory, filename)
            shutil.copy(original_filepath, destination_directory)
            return new_filepath
        except FileNotFoundError:
            self.log(f"Error: File not found at {original_filepath}")
        except Exception as e:
            self.log(f"Error: {e}")
            return None

    def list_image_paths(self, folder_path):
        """
        Lists the paths of all .jpg images in a specified folder, sorted naturally.

        Args:
            folder_path (str): Path to the folder.

        Returns:
            list: List of image file paths.
        """
        image_paths = []
        if os.path.exists(folder_path):
            for filename in natsorted(os.listdir(folder_path)):
                if filename.lower().endswith(".jpg"):
                    image_path = os.path.join(folder_path, filename)
                    image_paths.append(image_path)

        return image_paths
