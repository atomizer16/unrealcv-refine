import numpy as np
from PIL import Image

# 1. Load .npy file
arr = np.load(r'C:/Users/ASUS/Documents/unrealcv_script/rgb_depth_dataset/camera_006_CineCameraActor_6/depth/000000.npy')

# 2. [Preprocessing] If it's float, convert to uint8 type in range 0-255
if arr.dtype in [np.float32, np.float64]:
    # Assuming data range is 0.0-1.0, adjust if not the case
    arr = (arr * 255).astype(np.uint8)
else:
    arr = arr.astype(np.uint8)

# 3. Convert to image based on shape
if arr.ndim == 2:  # Grayscale image
    img = Image.fromarray(arr, mode='L')
elif arr.ndim == 3 and arr.shape[2] in [3, 4]:  # Color or RGBA image
    img = Image.fromarray(arr)
else:
    print(f"Cannot process array shape: {arr.shape}")

# 4. Save as PNG
img.save('output.png')