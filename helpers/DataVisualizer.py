import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


class DataVisualizer:
    """
    A class to visualize data for adversarial attack detection.
    """

    def log(self, message):
        """
        Logs a message with the class name.

        Args:
            message (str): The message to log.
        """
        print(f"{self.__class__.__name__}: {message}")

    def visualize_data(
        self,
        detected_attack_indexes: list,
        actual_attack_indexes: list,
        length_of_all_indexes: int,
        threshold_list: list = [],
        output_dir: str = "./",
        output_file_name: str = "all_plots.pdf",
    ):
        """
        Visualizes the attack detection results, including distribution plots, timeline plots, and metrics.

        Args:
            detected_attack_indexes (list): List of indexes where attacks were detected.
            actual_attack_indexes (list): List of indexes where attacks actually occurred.
            length_of_all_indexes (int): Total number of frames/images.
            threshold_list (list, optional): List of threshold values. Defaults to an empty list.
            output_dir (str, optional): Directory to save the output PDF. Defaults to the current directory.
            output_file_name (str, optional): Name of the output PDF file. Defaults to "all_plots.pdf".
        """
        with PdfPages(output_dir + output_file_name) as pdf:
            # Distribution plot
            plt.figure(figsize=(12, 6))
            plt.bar(
                [
                    "Actual\n Attack Frames",
                    "Detected\n Attack Frames",
                    "Undetected\n Attack Frames",
                    "Non-Attacked Frames\n by Detection",
                    "Non-Attacked Frames\n by Actual",
                ],
                [
                    len(actual_attack_indexes),
                    len(detected_attack_indexes),
                    len(set(actual_attack_indexes) - set(detected_attack_indexes)),
                    length_of_all_indexes - len(detected_attack_indexes),
                    length_of_all_indexes - len(actual_attack_indexes),
                ],
                color=["blue", "red", "yellow", "green", "grey"],
            )
            for index, value in enumerate(
                [
                    len(actual_attack_indexes),
                    len(detected_attack_indexes),
                    len(set(actual_attack_indexes) - set(detected_attack_indexes)),
                    length_of_all_indexes - len(detected_attack_indexes),
                    length_of_all_indexes - len(actual_attack_indexes),
                ]
            ):
                plt.text(index, value, str(value), ha="center")
            plt.title("Distribution of Attack Detection Results")
            plt.xlabel("Frames")
            plt.ylabel("Count")
            pdf.savefig()
            plt.close()

            # Timeline plot
            plt.figure(figsize=(12, 6))
            plt.scatter(
                detected_attack_indexes,
                [1] * len(detected_attack_indexes),
                color="red",
                label="Detected Attacks",
                marker="x",
            )
            plt.scatter(
                [
                    i
                    for i in range(length_of_all_indexes)
                    if i not in detected_attack_indexes
                ],
                [0] * (length_of_all_indexes - len(detected_attack_indexes)),
                color="green",
                label="Non-Attacked Frames",
                marker="o",
            )
            plt.title("Timeline of Detected Attacks")
            plt.xlabel("Frame Index")
            plt.yticks([0, 1], ["Detected\nNon-Attacked", "Detected Attacked"])
            plt.legend()
            pdf.savefig()
            plt.close()

            # Calculate metrics
            true_positives = len(
                set(detected_attack_indexes).intersection(set(actual_attack_indexes))
            )
            false_positives = len(
                set(detected_attack_indexes) - set(actual_attack_indexes)
            )
            false_negatives = len(
                set(actual_attack_indexes) - set(detected_attack_indexes)
            )

            cm = confusion_matrix(
                [
                    -1 if i in actual_attack_indexes else 1
                    for i in range(length_of_all_indexes)
                ],
                [
                    -1 if i in detected_attack_indexes else 1
                    for i in range(length_of_all_indexes)
                ],
                labels=[1, -1],
            )

            accuracy = (cm[0][0] + cm[1][1]) / cm.sum() * 100

            self.log(f"Accuracy: {accuracy:.2f}%")

            # Actual vs detected attack plot with metrics
            plt.figure(figsize=(12, 6))
            plt.scatter(
                actual_attack_indexes,
                [1] * len(actual_attack_indexes),
                color="blue",
                label="Actual Attacks",
                marker="x",
            )
            plt.scatter(
                detected_attack_indexes,
                [1] * len(detected_attack_indexes),
                color="red",
                label="Detected Attacks",
                marker="o",
                alpha=0.5,
            )
            plt.title("Actual vs Detected Attacks")
            plt.xlabel("Frame Index")
            plt.yticks([1], ["Actual Attack vs Detected Attack"])
            plt.legend()

            # Display metrics on the plot
            plt.text(
                0.02,
                0.92,
                f"Actual attacked frames: {len(actual_attack_indexes)}",
                transform=plt.gca().transAxes,
            )
            plt.text(
                0.02,
                0.88,
                f"Accurate detected frames: {true_positives}",
                transform=plt.gca().transAxes,
            )
            plt.text(
                0.02,
                0.84,
                f"False detected frames: {false_positives}",
                transform=plt.gca().transAxes,
            )
            plt.text(
                0.02,
                0.80,
                f"Undetected frames: {false_negatives}",
                transform=plt.gca().transAxes,
            )
            plt.text(
                0.02, 0.76, f"Accuracy: {accuracy:.2f}%", transform=plt.gca().transAxes
            )

            pdf.savefig()
            plt.close()

            # Threshold values plot
            if threshold_list:
                plt.figure(figsize=(12, 6))
                plt.plot(threshold_list)

                false_detect = plt.plot([], [], "ro", label="False Detection")[0]
                accurate_detect = plt.plot([], [], "go", label="Accurate Detection")[0]
                undetected = plt.plot([], [], "yo", label="Undetected")[0]

                for i in range(len(threshold_list)):
                    if i in detected_attack_indexes:
                        if i not in actual_attack_indexes:
                            plt.plot(i, threshold_list[i], "ro")
                        else:
                            plt.plot(i, threshold_list[i], "go")
                    if i not in detected_attack_indexes and i in actual_attack_indexes:
                        plt.plot(i, threshold_list[i], "yo")

                plt.title("Threshold Values")
                plt.xlabel("Image Index")
                plt.ylabel("Threshold")

                plt.axvline(20, color="green")

                plt.legend(
                    handles=[false_detect, accurate_detect, undetected],
                    loc="upper left",
                )

                pdf.savefig()
                plt.close()

            # Confusion matrix plot
            plt.figure(figsize=(12, 6))
            disp_cm = ConfusionMatrixDisplay(
                cm, display_labels=["Non-Attacked", "Attacked"]
            )
            disp_cm.plot()
            plt.title("Confusion Matrix")
            plt.grid(False)
            plt.tight_layout()

            pdf.savefig()
            plt.close()
