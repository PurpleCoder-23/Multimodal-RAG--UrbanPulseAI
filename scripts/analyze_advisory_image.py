
import os
import json
import base64
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

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
OUTPUT_PATH = OUTPUT_DIR / "advisory_image_vlm.json"

MODEL_NAME = "qwen/qwen3.8-27b"


def encode_image(image_path):
    """Convert local image into Base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(
            image_file.read()
        ).decode("utf-8")


def analyze_image():
    # Load environment variables from project root
    load_dotenv(PROJECT_ROOT / ".env")

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. Check your .env file."
        )

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Image not found: {IMAGE_PATH}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading advisory image...")

    base64_image = encode_image(IMAGE_PATH)

    client = Groq(api_key=api_key)

    prompt = """
    Analyze this Hyderabad traffic police advisory image.

    Extract the following information:

    1. Document title and issuing authority.
    2. Event and advisory validity period.
    3. Main visual sections and their arrangement.
    4. Traffic diversion instructions.
    5. Junctions and roads mentioned.
    6. Symbols, map elements, arrows or visual relationships.
    7. Any information that is visually unclear.

    Return a valid JSON object with these keys:
    document_title,
    issuing_authority,
    event,
    validity_period,
    visual_layout,
    traffic_diversions,
    junctions,
    visual_elements,
    uncertainty_notes.

    Do not invent missing information.
    Preserve uncertain names as uncertain rather than guessing.
    """

    print("Sending image to vision-language model...")

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": (
                                "data:image/jpeg;base64,"
                                + base64_image
                            )
                        }
                    }
                ]
            }
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
        max_completion_tokens=3000
    )

    result_text = response.choices[0].message.content

    if not result_text:
        raise RuntimeError("VLM returned an empty response.")

    result = json.loads(result_text)

    output = {
        "document_id": "durga_matha_immersion_2026",
        "source_type": "vlm_image_analysis",
        "source_file": str(
            IMAGE_PATH.relative_to(PROJECT_ROOT)
        ),
        "model": MODEL_NAME,
        "analysis": result
    }

    OUTPUT_PATH.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("\nVLM analysis completed.")
    print(f"Saved output: {OUTPUT_PATH}")
    print("\n--- VLM Analysis ---")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    analyze_image()
