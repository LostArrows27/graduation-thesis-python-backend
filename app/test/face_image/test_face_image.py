import os
import json
from io import BytesIO
import traceback

from app.libs.logger.log import log_error, log_info
from app.models.model import FaceCategoryModel, save_image_with_faces

# Now import app modules


def process_face_images():
    # Initialize face detection model
    face_model = FaceCategoryModel()

    # Get the current directory where test_face_image.py is located
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Define paths
    image_dir = os.path.join(current_dir, "images")
    image_list_path = os.path.join(current_dir, "image_list.json")
    output_dir = os.path.join(current_dir, "test_face_results")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Read the image list from JSON file
    with open(image_list_path, 'r') as f:
        image_list = json.load(f)

    # Initialize counters
    total_faces_detected = 0
    total_faces_expected = sum(item.get('faces', 0) for item in image_list)

    # Process each image in the list
    for item in image_list:
        image_name = item['name']
        image_path = os.path.join(image_dir, image_name)

        try:
            log_info(f"Processing image: {image_name}")

            # Open the image file
            with open(image_path, 'rb') as img_file:
                # Create a BytesIO object to hold the image data
                image_bytes = BytesIO(img_file.read())
                image_bytes.name = image_name  # Set name attribute for save_image_with_faces

            # Detect faces
            face_locations, face_encodings = face_model.category_image(
                image_bytes)
            faces_found = len(face_locations)

            # Add the found attribute to the image item
            item['found'] = faces_found

            # Add to total count
            total_faces_detected += faces_found

            # Save image with highlighted faces
            if faces_found > 0:
                # Reset the position to the beginning of the file
                image_bytes.seek(0)
                save_image_with_faces(image_bytes, face_locations, output_dir)

            log_info(f"Found {faces_found} faces in {image_name}")

        except Exception as e:
            log_error(f"Error processing {image_name}: {str(e)}")
            log_error(traceback.format_exc())
            item['found'] = 0
            item['error'] = str(e)

    # Create results summary
    results = {
        'total_faces_detected': total_faces_detected,
        'total_faces_expected': total_faces_expected,
        'images': image_list
    }

    # Save results to JSON file
    results_file = os.path.join(output_dir, 'face_detection_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    log_info(f"Face detection complete. Results saved to {results_file}")
    log_info(
        f"Detected {total_faces_detected} faces out of {total_faces_expected} expected")
