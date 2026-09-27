import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

def create_sample_images():
    sample_dir = os.path.join("static", "samples")
    os.makedirs(sample_dir, exist_ok=True)

    configs = {
        "adenocarcinoma.jpg": ("adenocarcinoma", (55, 65, 95, 105), 230),
        "large_cell_carcinoma.jpg": ("large_cell", (125, 75, 175, 130), 245),
        "squamous_cell_carcinoma.jpg": ("squamous", (60, 110, 100, 150), 210),
        "normal.jpg": ("normal", None, 0)
    }

    for fname, (cls_type, bbox, val) in configs.items():
        fpath = os.path.join(sample_dir, fname)
        # Create base CT-scan background
        img = Image.new('L', (512, 512), color=15)
        draw = ImageDraw.Draw(img)

        # Draw thoracic wall & ribcage outline
        draw.ellipse([50, 40, 462, 472], outline=110, width=16)

        # Draw left and right lung fields
        draw.ellipse([80, 80, 220, 420], fill=45)
        draw.ellipse([292, 80, 432, 420], fill=45)

        # Draw spine and mediastinum (central heart area)
        draw.rectangle([236, 120, 276, 400], fill=160)
        draw.ellipse([216, 200, 296, 340], fill=140)

        # Draw pulmonary vascular markings
        for i in range(5):
            draw.line([150, 150 + i*40, 180 + i*10, 170 + i*40], fill=90, width=3)
            draw.line([360, 150 + i*40, 330 - i*10, 170 + i*40], fill=90, width=3)

        if bbox is not None:
            # Scale bbox to 512x512
            b = [int(v * 512 / 224) for v in bbox]
            draw.ellipse(b, fill=val)

        # Apply slight blur to simulate realistic soft tissue CT density
        img = img.filter(ImageFilter.GaussianBlur(radius=1.5))
        img = img.convert('RGB')
        img.save(fpath, quality=95)
        print(f"Generated sample image: {fpath}")

if __name__ == '__main__':
    create_sample_images()
