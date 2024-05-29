import os
import numpy as np
from PIL import Image


class AdversarialAttack:
    """
    A class to perform adversarial attacks on images using the Fast Gradient Sign Method (FGSM).
    """

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

    def fgsm_attack(self, image_path, epsilon=0.01, output_dir="."):
        """
        Performs an FGSM attack on the given image and saves the perturbed image.

        Args:
            image_path (str): The path to the input image.
            epsilon (float, optional): The attack strength parameter. Defaults to 0.01.
            output_dir (str, optional): The directory to save the perturbed image. Defaults to the current directory.

        Returns:
            str: The path to the perturbed image.
        """
        # Open the image and convert it to a NumPy array
        image = Image.open(image_path)
        image = np.asarray(image).astype(np.float32)

        # Generate the perturbation
        perturbation = epsilon * np.sign(np.random.randn(*image.shape))

        # Apply the perturbation and clip the values to be in the valid range [0, 255]
        perturbed_image = image + perturbation
        perturbed_image = np.clip(perturbed_image, 0, 255)

        # Construct the filename for the perturbed image
        filename = os.path.basename(image_path).split(".")[0]
        perturbed_path = os.path.join(output_dir, f"{filename}_perturbed.jpg")

        # Create the output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save the perturbed image
        perturbed_image = Image.fromarray(perturbed_image.astype(np.uint8))
        perturbed_image.save(perturbed_path)

        self.log(f"Completed FGSM attack on {filename}")
        return perturbed_path
