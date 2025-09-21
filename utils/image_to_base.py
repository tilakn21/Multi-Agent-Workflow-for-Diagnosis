import base64

def image_to_base64(image_path: str) -> str:
    """
    Converts an image file to a base64-encoded string.

    Args:
        image_path (str): Path to the image file.

    Returns:
        str: Base64-encoded string of the image with data URI prefix.
    """
    try:
        with open(image_path, "rb") as f:
            b64_data = base64.b64encode(f.read()).decode("utf-8")
        # Return with data URI prefix for PNG (change MIME type if needed)
        return f"data:image/png;base64,{b64_data}"
    except Exception as e:
        print(f"Error converting image to base64: {e}")
        return ""
image_path = "image.png"  # Relative path in utils folder
image_base64 = image_to_base64(image_path)

# Store the base64 string in a text file (relative path)
with open("image_base64.txt", "w") as txt_file:
    txt_file.write(image_base64)