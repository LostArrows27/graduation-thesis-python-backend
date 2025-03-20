import json
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from app.models.inference import AILabelTestService
from app.models.model import AIModel

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.absolute()


def draw_text_on_image(image, text, save_path):
    # Get original image dimensions
    orig_width, orig_height = image.size

    # Define new dimensions with better proportions
    max_width = 800  # Set a reasonable max width
    text_height = 250  # More space for text

    # Calculate new image dimensions while maintaining aspect ratio
    if orig_width > max_width:
        ratio = max_width / orig_width
        new_width = max_width
        new_height = int(orig_height * ratio)
    else:
        new_width = orig_width
        new_height = orig_height

    # Resize the original image
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)

    # Create a new image with space for text
    result_image = Image.new(
        "RGB", (new_width, new_height + text_height), (255, 255, 255))
    # Paste original image below text area
    result_image.paste(resized_image, (0, text_height))

    # Draw text
    draw = ImageDraw.Draw(result_image)

    # Try to use a larger font
    try:
        font = ImageFont.truetype("arial.ttf", 24)  # Even larger font size
    except IOError:
        try:
            # Try system fonts if arial isn't available
            system_fonts = [
                "DejaVuSans.ttf",
                "Arial.ttf",
                "Verdana.ttf",
                "Tahoma.ttf",
                "Calibri.ttf"
            ]
            for font_name in system_fonts:
                try:
                    font = ImageFont.truetype(font_name, 24)
                    break
                except IOError:
                    continue
        except:
            font = ImageFont.load_default()

    # Format text for better display if it's JSON
    if text.startswith("{"):
        # It's JSON text - try to make it more readable
        try:
            json_data = json.loads(text)
            formatted_text = []

            # For each category, format the top labels
            for category, labels in json_data.items():
                category_line = f"{category.replace('_labels', '').title()}: "
                label_texts = []

                for label_obj in labels:
                    for label, value in label_obj.items():
                        label_texts.append(f"{label} ({value}%)")

                formatted_text.append(
                    f"{category_line}{', '.join(label_texts)}")

            # Join all category lines
            text = "\n".join(formatted_text)
        except:
            # If JSON parsing fails, use the original text
            pass

    # Center the text horizontally and position it vertically
    lines = text.split('\n')
    y_position = 20  # Start position from top

    for line in lines:
        # Get text dimensions - compatible with newer PIL versions
        try:
            # Method 1: For newer Pillow versions
            left, top, right, bottom = font.getbbox(line)
            text_width = right - left
            text_height = bottom - top
        except AttributeError:
            try:
                # Method 2: Alternative for newer versions
                text_width, text_height = draw.textsize(line, font=font)
            except AttributeError:
                # Method 3: Fallback to a reasonable estimate
                text_width = len(line) * (font.size // 2)
                text_height = font.size + 4

        x_position = (new_width - text_width) // 2  # Center horizontally

        # Draw the text - bolder and bigger
        draw.text((x_position, y_position), line, fill=(0, 0, 0), font=font)
        y_position += text_height + 15  # More spacing between lines

    # Save result
    result_image.save(save_path)
    print(f"Saved result image to: {save_path}")


def format_label_json(labels):
    """Format label dictionary with 2 decimal places for percentages"""
    formatted = {}
    for category, label_list in labels.items():
        formatted[category] = []
        for label_dict in label_list:
            for label, value in label_dict.items():
                formatted[category].append({label: round(value, 2)})

    return json.dumps(formatted, indent=2)


def process_test_open_clip():
    try:
        print("\n==== STARTING IMAGE LABELING TEST ====")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Script directory: {SCRIPT_DIR}")

        # Create result directories if they don't exist
        results_dir = SCRIPT_DIR / "results"
        relate_results_dir = results_dir / "relate"
        unrelate_results_dir = results_dir / "unrelate"

        relate_results_dir.mkdir(parents=True, exist_ok=True)
        unrelate_results_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created results directories in: {results_dir}")

        try:
            model = AIModel()
            label_service = AILabelTestService(model)
            print("Initialized AIModel and AILabelTestService")
        except Exception as e:
            print(f"ERROR initializing model: {str(e)}")
            return

        # Process "relate" images - using script directory
        relate_path = SCRIPT_DIR / "image" / "relate"
        print(f"Looking for related images in: {relate_path}")

        if relate_path.exists():
            print(f"Found relate directory: {relate_path}")
            image_files = list(relate_path.glob("*.*"))
            print(f"Found {len(image_files)} files in relate directory")

            for img_file in image_files:
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.jfif']:
                    print(f"Processing relate image: {img_file.name}")

                    try:
                        # Load and process image
                        image = Image.open(img_file)
                        print(
                            f"  Loaded image: {img_file.name} ({image.size})")

                        name, is_relate, image_features = label_service.return_relate_status_with_name(
                            image)
                        print(
                            f"  Classification result: {name}, is_relate={is_relate}")

                        if is_relate:
                            # Get all labels for related images
                            labels = label_service.return_all_labels(
                                image_features)
                            print(f"  Got labels: {len(labels)} categories")

                            # Format and save result
                            labels_text = format_label_json(labels)
                            save_path = relate_results_dir / \
                                f"{img_file.stem}_result{img_file.suffix}"
                            draw_text_on_image(image, labels_text, save_path)
                        else:
                            print(
                                f"  Warning: Image {img_file.name} marked as unrelated, but in 'relate' folder")
                    except Exception as e:
                        print(
                            f"  ERROR processing relate image {img_file.name}: {str(e)}")
        else:
            print(f"ERROR: Relate directory not found at {relate_path}")
            # Try to help user - create the directory structure
            print(f"Creating directory structure at {relate_path}")
            relate_path.mkdir(parents=True, exist_ok=True)
            print(f"Please put related images in {relate_path}")

        # Process "unrelate" images - using script directory
        unrelate_path = SCRIPT_DIR / "image" / "unrelate"
        print(f"Looking for unrelated images in: {unrelate_path}")

        if unrelate_path.exists():
            print(f"Found unrelate directory: {unrelate_path}")
            image_files = list(unrelate_path.glob("*.*"))
            print(f"Found {len(image_files)} files in unrelate directory")

            for img_file in image_files:
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.jfif']:
                    print(f"Processing unrelate image: {img_file.name}")

                    try:
                        # Load and process image
                        image = Image.open(img_file)
                        print(
                            f"  Loaded image: {img_file.name} ({image.size})")

                        name, is_relate, _ = label_service.return_relate_status_with_name(
                            image)
                        print(
                            f"  Classification result: {name}, is_relate={is_relate}")

                        # Save result with unrelate name
                        save_path = unrelate_results_dir / \
                            f"{img_file.stem}_result{img_file.suffix}"
                        draw_text_on_image(
                            image, f"Unrelated: {name}", save_path)
                    except Exception as e:
                        print(
                            f"  ERROR processing unrelate image {img_file.name}: {str(e)}")
        else:
            print(f"ERROR: Unrelate directory not found at {unrelate_path}")
            # Try to help user - create the directory structure
            print(f"Creating directory structure at {unrelate_path}")
            unrelate_path.mkdir(parents=True, exist_ok=True)
            print(f"Please put unrelated images in {unrelate_path}")

        print("==== COMPLETED IMAGE LABELING TEST ====\n")
    except Exception as e:
        print(f"ERROR in process_test_open_clip: {str(e)}")
