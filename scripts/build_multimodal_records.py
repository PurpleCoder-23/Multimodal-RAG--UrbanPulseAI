
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

OCR_PATH = PROCESSED_DIR / "advisory_image_ocr.json"
VLM_PATH = PROCESSED_DIR / "advisory_image_vlm.json"

OUTPUT_PATH = PROCESSED_DIR / "multimodal_records.jsonl"


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main():
    ocr_data = load_json(OCR_PATH)
    vlm_data = load_json(VLM_PATH)

    analysis = vlm_data["analysis"]

    records = []

    common_metadata = {
        "document_id": ocr_data["document_id"],
        "source_image": ocr_data["source_file"],
        "source_type": "traffic_advisory_image",
    }

    # Record 1: OCR-extracted text
    records.append({
        "record_id": "durga_advisory_ocr",
        "modality": "text",
        "evidence_type": "ocr_extraction",
        "content": ocr_data["extracted_text"],
        "metadata": {
            **common_metadata,
            "extraction_method": "RapidOCR",
            "line_count": ocr_data["line_count"],
        },
    })

    # Record 2: VLM visual interpretation
    visual_summary = (
        f"Document: {analysis.get('document_title', '')}. "
        f"Issuing authority: {analysis.get('issuing_authority', '')}. "
        f"Event: {analysis.get('event', '')}. "
        f"Validity: {analysis.get('validity_period', '')}. "
        f"Visual layout: {analysis.get('visual_layout', '')}. "
        f"Visual elements: "
        f"{', '.join(analysis.get('visual_elements', []))}."
    )

    records.append({
        "record_id": "durga_advisory_visual_summary",
        "modality": "image",
        "evidence_type": "vlm_visual_interpretation",
        "content": visual_summary,
        "metadata": {
            **common_metadata,
            "model": vlm_data["model"],
        },
    })

    # Individual diversion records
    for diversion in analysis.get("traffic_diversions", []):
        records.append({
            "record_id": f"durga_diversion_{diversion['id']}",
            "modality": "image_derived_text",
            "evidence_type": "traffic_diversion",
            "content": diversion["instruction"],
            "metadata": {
                **common_metadata,
                "diversion_id": diversion["id"],
                "model": vlm_data["model"],
            },
        })

    # Individual junction records
    for index, junction in enumerate(
        analysis.get("junctions", []), start=1
    ):
        records.append({
            "record_id": f"durga_junction_{index}",
            "modality": "image_derived_text",
            "evidence_type": "junction",
            "content": f"Junction mentioned in the advisory: {junction}",
            "metadata": {
                **common_metadata,
                "model": vlm_data["model"],
            },
        })

    # Save records as JSON Lines
    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("Multimodal integration completed.")
    print(f"Total records: {len(records)}")
    print(f"Output: {OUTPUT_PATH}")

    print("\nRecord distribution:")

    for evidence_type in sorted(
        {record["evidence_type"] for record in records}
    ):
        count = sum(
            record["evidence_type"] == evidence_type
            for record in records
        )
        print(f"{evidence_type}: {count}")


if __name__ == "__main__":
    main()
