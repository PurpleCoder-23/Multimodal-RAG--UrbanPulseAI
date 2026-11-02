
import json
from pathlib import Path

from PIL import Image
from rapidocr_onnxruntime import RapidOCR

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_PATH = (
    PROJECT_ROOT
    / "data"
    / "Advisories"
    / "images"
    / "durga_matha_immersion_2026.jpeg"
)
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = OUTPUT_DIR / "advisory_image_ocr.json"


def main():
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Advisory image not found: {IMAGE_PATH}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Validate the image and obtain its dimensions.
    with Image.open(IMAGE_PATH) as image:
        image_width, image_height = image.size
        image_format = image.format

    print(f"Image: {IMAGE_PATH.name}")
    print(f"Dimensions: {image_width} x {image_height}")
    print(f"Format: {image_format}")

    # Extract text and its bounding boxes.
    ocr = RapidOCR()
    result, _ = ocr(str(IMAGE_PATH))

    extracted_lines = []

    if result:
        for item in result:
            text = item[1]
            confidence = float(item[2])

            extracted_lines.append(
                {
                    "text": text,
                    "confidence": confidence,
                    "bounding_box": item[0],
                }
            )

    full_text = "\n".join(
        item["text"] for item in extracted_lines
    )

    document = {
        "document_id": "durga_matha_immersion_2026",
        "source_type": "traffic_advisory_image",
        "source_file": str(
            IMAGE_PATH.relative_to(PROJECT_ROOT)
        ),
        "image_metadata": {
            "width": image_width,
            "height": image_height,
            "format": image_format,
        },
        "extraction_method": "RapidOCR",
        "extracted_text": full_text,
        "lines": extracted_lines,
        "line_count": len(extracted_lines),
    }

    OUTPUT_PATH.write_text(
        json.dumps(document, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nOCR lines extracted: {len(extracted_lines)}")
    print(f"Saved OCR output: {OUTPUT_PATH}")

    print("\n--- Extracted text preview ---")
    print(full_text[:2500] if full_text else "No text detected.")


if __name__ == "__main__":
    main()
