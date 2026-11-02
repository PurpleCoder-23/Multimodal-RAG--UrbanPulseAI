
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUTPUT_DIR = DATA / "processed"
OUTPUT_FILE = OUTPUT_DIR / "text_chunks.jsonl"

CHUNK_SIZE = 120
OVERLAP = 25


def load_json(path):
    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def split_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    """Split text into overlapping word-based chunks."""
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))

        if end == len(words):
            break

        start = end - overlap

    return chunks


def create_chunks(source_type, source_file, title, text, metadata):
    records = []

    for index, chunk in enumerate(split_text(text)):
        records.append({
            "chunk_id": f"{source_type}-{index + 1:04d}",
            "source_type": source_type,
            "source_file": str(source_file.relative_to(ROOT)),
            "title": title,
            "content": chunk,
            "metadata": metadata,
        })

    return records


def main():
    all_chunks = []

    # Traffic advisory: create searchable passages from individual sections.
    advisory_path = DATA / "Advisories" / "durga_matha_immersion_2026.json"
    advisory = load_json(advisory_path)

    sections = [
        ("Overview", {
            "title": advisory.get("title"),
            "source": advisory.get("source"),
            "published_notice_period": advisory.get("published_notice_period"),
            "status": advisory.get("status_as_of_2026_10_29"),
            "affected_locations": advisory.get("affected_locations"),
            "expected_congestion": advisory.get("expected_congestion"),
        }),
        ("Diversions", advisory.get("diversions", [])),
        ("Junctions to avoid", advisory.get("junctions_to_avoid", [])),
    ]

    for section_name, section_data in sections:
        text = f"{section_name}: {json.dumps(section_data, ensure_ascii=False)}"

        all_chunks.extend(
            create_chunks(
                source_type="advisory",
                source_file=advisory_path,
                title=f"{advisory.get('title', 'Traffic advisory')} — {section_name}",
                text=text,
                metadata={
                    "section": section_name,
                    "status": advisory.get("status_as_of_2026_10_29"),
                    "source_url": advisory.get("source_url"),
                },
            )
        )

    # Weather: turn measured values and units into readable text.
    weather_path = DATA / "Weather" / "hyderabad_weather.json"
    weather = load_json(weather_path)

    current = weather.get("current", {})
    units = weather.get("current_units", {})

    weather_text = (
        f"Hyderabad weather snapshot at {current.get('time')}. "
        f"Temperature: {current.get('temperature_2m')} "
        f"{units.get('temperature_2m', 'degrees Celsius')}. "
        f"Relative humidity: {current.get('relative_humidity_2m')} "
        f"{units.get('relative_humidity_2m', 'percent')}. "
        f"Precipitation: {current.get('precipitation')} "
        f"{units.get('precipitation', 'mm')}. "
        f"Wind speed: {current.get('wind_speed_10m')} "
        f"{units.get('wind_speed_10m', 'km/h')}."
    )

    all_chunks.extend(
        create_chunks(
            source_type="weather",
            source_file=weather_path,
            title="Hyderabad current weather",
            text=weather_text,
            metadata={
                "timestamp": current.get("time"),
                "timezone": weather.get("timezone"),
                "latitude": weather.get("latitude"),
                "longitude": weather.get("longitude"),
            },
        )
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        for record in all_chunks:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("Text chunking completed.")
    print("Chunks created:", len(all_chunks))
    print("Output:", OUTPUT_FILE.relative_to(ROOT))


if __name__ == "__main__":
    main()
