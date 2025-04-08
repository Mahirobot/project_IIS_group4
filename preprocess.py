import pandas as pd
import os
import cv2
from feat import Detector


def main():
    # Path to the dataset containing cropped images
    data_path = "./DiffusionFER/DiffusionEmotion_S/cropped/"

    # Initialize variables for storing data
    image_paths = []
    labels = []
    image_files = []

    # Loop through the subfolders in the dataset directory
    for folder_name in os.listdir(data_path):
        image_folder = data_path + folder_name
        images = os.listdir(image_folder)

        # Add image paths and labels to respective lists
        for image in images:
            labels.append(folder_name)
            image_paths.append(image_folder + f'/{image}')
            image_files.append(cv2.imread(image_paths[-1]))

    # Initialize the facial expression detector
    detector = Detector(device="cpu")

    # List to store Action Unit (AU) data for each image
    aus_data = []

    # Create a DataFrame to store image paths and labels
    data = pd.DataFrame(
        {'image_paths': image_paths,
         'labels': labels,
         })

    # Process each image and extract AU data
    for image_path in data['image_paths']:
        # Detect facial features and action units in the image
        result = detector.detect_image(image_path)

        # Number of faces detected in the image
        num_rows = len(result)

        # Extract the AU data
        new_df = result.aus

        # Modify the image path to a simplified format
        image_path = image_path.replace(".png", "")
        image_path = image_path.replace("./DiffusionFER/DiffusionEmotion_S/cropped/", "")

        # Add file and face ID columns to the AU data
        new_df.insert(0, "file", image_path)
        new_df.insert(1, "face", range(num_rows))

        # Append the modified AU data to the list
        aus_data.append(new_df)

    # Merge all AU data from the list into a single DataFrame
    merged_df = pd.concat(aus_data, ignore_index=True)

    # Save the merged AU data to a CSV file
    aus_df = pd.DataFrame(merged_df)
    aus_df.to_csv("aus.csv", index=False)


# Call the main function
if __name__ == "__main__":
    main()
