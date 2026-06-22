import io
from PIL import Image
import numpy as np

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Preprocess an uploaded image for the CIFAR-10 model.

    Steps:
    1. Open the bytes with Pillow.
    2. Convert to RGB (handles RGBA or grayscale).
    3. Resize to 32x32 using LANCZOS filtering.
    4. Convert to NumPy array of type float32 and normalise to [0, 1].
    5. Reshape to (1, 32, 32, 3) as required by the model.
    """
    # Open image from bytes
    with Image.open(io.BytesIO(image_bytes)) as img:
        img = img.convert("RGB")
        
        # Center crop to a square to preserve aspect ratio
        width, height = img.size
        min_dim = min(width, height)
        left = (width - min_dim) / 2
        top = (height - min_dim) / 2
        right = (width + min_dim) / 2
        bottom = (height + min_dim) / 2
        img = img.crop((left, top, right, bottom))
        
        img = img.resize((32, 32), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        # Ensure shape (32, 32, 3)
        if arr.shape != (32, 32, 3):
            raise ValueError(f"Unexpected image shape {arr.shape}, expected (32, 32, 3)")
        return arr.reshape((1, 32, 32, 3))

def get_class_names() -> list[str]:
    """Return Turkish class names for CIFAR-10 in order.
    """
    return [
        "uçak",
        "araba",
        "kuş",
        "kedi",
        "geyik",
        "köpek",
        "kurbağa",
        "at",
        "gemi",
        "kamyon",
    ]
