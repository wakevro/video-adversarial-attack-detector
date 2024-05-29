import numpy as np
import cv2
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


class AttackDetector:
    """
    A class to detect adversarial attacks on images using Isolation Forest.
    """

    def __init__(self, contamination=0.1) -> None:
        """
        Initializes the AttackDetector with a specified contamination level.

        Args:
            contamination (float, optional): The proportion of outliers in the data set. Defaults to 0.1.
        """
        self.contamination = contamination
        self.model = IsolationForest(contamination=self.contamination, random_state=0)

    def log(self, message):
        """
        Logs a message with the class name.

        Args:
            message (str): The message to log.
        """
        print(f"{self.__class__.__name__}: {message}")

    def newLine(self):
        """
        Prints a new line.
        """
        print("\n")

    def detect_attack_given_two_paths(self, main_image_path, second_image_path):
        """
        Detects the difference in image statistics between two images.

        Args:
            main_image_path (str): The path to the main image.
            second_image_path (str): The path to the second image for comparison.

        Returns:
            float: The difference in the sum of squared differences from the mean of the images.
        """
        image_one = cv2.imread(main_image_path)
        image_one_mean = np.mean(image_one)
        image_one_ssq = np.sum((image_one - image_one_mean) ** 2)

        image_two = cv2.imread(second_image_path)
        image_two_mean = np.mean(image_two)
        image_two_ssq = np.sum((image_two - image_two_mean) ** 2)

        difference = round(image_one_ssq - image_two_ssq, 14)
        return difference

    def extract_features(self, image_path):
        """
        Extracts features from an image by flattening it into a 1D array.

        Args:
            image_path (str): The path to the image.

        Returns:
            np.ndarray: The flattened image as a feature vector.
        """
        img = cv2.imread(image_path)
        features = img.flatten()
        return features

    def detect_attack_from_image_paths(self, image_paths: list):
        """
        Detects adversarial attacks from a list of image paths using Isolation Forest.

        Args:
            image_paths (list): A list of paths to the images.

        Returns:
            tuple: A tuple containing a boolean indicating success, a list of indexes of attacked images,
                   and a list of prediction scores.
        """
        self.log("Started detecting attacks...")
        if len(image_paths) < 30:
            self.log("Not enough images to detect outliers")
            return False, [], []

        self.model = IsolationForest(contamination=self.contamination, random_state=0)

        features = [self.extract_features(path) for path in image_paths]

        self.model.fit(features)
        predictions = self.model.predict(features)

        # Identify attacked images based on predictions
        threshold_list = predictions.tolist()
        attacked_images_indexes = [
            i for i in range(len(image_paths)) if predictions[i] == -1 and (i > 20)
        ]

        self.log(f"Finished detection of outliers... {len(attacked_images_indexes)}")
        return True, attacked_images_indexes, threshold_list
