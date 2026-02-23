from PIL import Image, ImageDraw, ImageFilter, ImageFont
import numpy as np

W, H = 1024, 768
# Create gradient background (ocean -> land)
arr = np.zeros((H, W, 3), dtype=np.uint8)
for y in range(H):
    t = y / H
    # mix blue (ocean) and green/brown (land)
    ocean = np.array([20, 60, 160])
    land = np.array([120, 170, 70])
    color = (1 - t) * ocean + t * land
    arr[y, :, :] = color.astype(np.uint8)

# Add synthetic urban areas (bright patches) and pollution hotspots (red blobs)
rng = np.random.default_rng(1)
for _ in range(60):
    cx = rng.integers(100, W-100)
    cy = rng.integers(100, H-100)
    r = rng.integers(8, 40)
    y, x = np.ogrid[-cy:H-cy, -cx:W-cx]
    mask = x*x + y*y <= r*r
    # urban: lighter gray-green
    arr[mask] = np.clip(arr[mask] + rng.integers(30, 80), 0, 255)

# pollution heatmap blobs
for _ in range(8):
    cx = rng.integers(200, W-200)
    cy = rng.integers(150, H-150)
    r = rng.integers(30, 120)
    intensity = rng.uniform(0.5, 1.0)
    y, x = np.ogrid[-cy:H-cy, -cx:W-cx]
    dist2 = x*x + y*y
    blob = np.exp(-dist2 / (2*(r**2))) * 255 * intensity
    # add red-yellow heat color
    arr[:, :, 0] = np.clip(arr[:, :, 0] + blob * 0.6, 0, 255)
    arr[:, :, 1] = np.clip(arr[:, :, 1] + blob * 0.3, 0, 255)

img = Image.fromarray(arr)
img = img.filter(ImageFilter.GaussianBlur(radius=1.2))

draw = ImageDraw.Draw(img)
# add small coastline/road lines
for i in range(150):
    x0 = rng.integers(0, W)
    y0 = rng.integers(0, H)
    x1 = x0 + rng.integers(-40, 40)
    y1 = y0 + rng.integers(-40, 40)
    draw.line((x0, y0, x1, y1), fill=(200,200,180), width=1)

# overlay title
try:
    font = ImageFont.truetype("DejaVuSans-Bold.ttf", 28)
except Exception:
    font = ImageFont.load_default()

text = "Synthetic Satellite — PM2.5 Sample"
try:
    bbox = draw.textbbox((0,0), text, font=font)
    w_text = bbox[2] - bbox[0]
    h_text = bbox[3] - bbox[1]
except Exception:
    try:
        w_text, h_text = font.getsize(text)
    except Exception:
        w_text, h_text = (300, 18)
draw.rectangle(((10, H-40), (12 + w_text + 8, H-10)), fill=(0,0,0,140))
draw.text((14, H-36), text, fill=(255,255,255), font=font)

img.save("datasets_images/train/satellite_sample.png")
img.save("datasets_images/val/satellite_sample.png")
img.save("datasets_images/test/satellite_sample.png")
print("Saved sample images to train/, val/, test/")
