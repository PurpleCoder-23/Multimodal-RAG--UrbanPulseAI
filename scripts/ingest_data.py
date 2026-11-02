
import csv
import json
from pathlib import Path

# Project paths
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUTPUT_DIR = DATA / "processed"
OUTPUT_FILE = OUTPUT_DIR / "ingested_records.jsonl"


def read_json(path):
    """Read JSON files, including UTF-8 BOM files."""
    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def make_record(source_type, source_file, title, content, metadata=None):
    """Create one consistently structured record."""
    return {
        "source_type": source_type,
        "source_file": str(source_file.relative_to(ROOT)),
        "title": title,
        "content": content,
        "metadata": metadata or {},
    }


def ingest_advisory():
    path = DATA / "Advisories" / "durga_matha_immersion_2026.json"
    advisory = read_json(path)

    return [
        make_record(
            source_type="advisory",
            source_file=path,
            title=advisory.get("title", "Traffic advisory"),
            content=json.dumps(advisory, ensure_ascii=False, indent=2),
            metadata={
                "advisory_id": advisory.get("advisory_id"),
                "status": advisory.get("status_as_of_2026_10_29"),
                "source_url": advisory.get("source_url"),
            },
        )
    ]


def ingest_weather():
    path = DATA / "Weather" / "hyderabad_weather.json"
    weather = read_json(path)

    return [
        make_record(
            source_type="weather",
            source_file=path,
            title="Hyderabad weather snapshot",
            content=json.dumps(weather, ensure_ascii=False, indent=2),
            metadata={
                "latitude": weather.get("latitude"),
                "longitude": weather.get("longitude"),
                "timezone": weather.get("timezone"),
            },
        )
    ]


def ingest_traffic():
    path = DATA / "traffic" / "synthetic_traffic_records.csv"
    records = []

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            records.append(
                make_record(
                    source_type="traffic",
                    source_file=path,
                    title=f"Traffic record {row['record_id']}",
                    content=(
                        f"Road: {row['location']}. "
                        f"Time: {row['simulation_time']}. "
                        f"Speed: {row['speed_kmh']} km/h. "
                        f"Vehicles per minute: {row['vehicles_per_minute']}. "
                        f"Congestion: {row['congestion_level']}."
                    ),
                    metadata={
                        "record_id": row["record_id"],
                        "road": row["location"],
                        "timestamp": row["simulation_time"],
                        "record_origin": row["record_origin"],
                        "is_real_observation": row["is_real_observation"],
                    },
                )
            )

    return records


def ingest_roads():
    path = DATA / "Maps" / "hyderabad_road_segments.geojson"
    geojson = read_json(path)
    records = []

    for index, feature in enumerate(geojson.get("features", [])):
        properties = feature.get("properties") or {}
        geometry = feature.get("geometry") or {}

        road_type = properties.get("highway")
        if not road_type:
            continue

        road_name = properties.get("name") or f"Unnamed road segment {index + 1}"

        records.append(
            make_record(
                source_type="road",
                source_file=path,
                title=road_name,
                content=f"Road name: {road_name}. Road type: {road_type}.",
                metadata={
                    "road_type": road_type,
                    "geometry_type": geometry.get("type"),
                    "feature_index": index,
                    "record_origin": "OpenStreetMap extract",
                },
            )
        )

    return records


def main():
    documents = []
    documents.extend(ingest_advisory())
    documents.extend(ingest_weather())
    documents.extend(ingest_roads())
    documents.extend(ingest_traffic())

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        for document in documents:
            file.write(json.dumps(document, ensure_ascii=False) + "\n")

    counts = {}
    for document in documents:
        source_type = document["source_type"]
        counts[source_type] = counts.get(source_type, 0) + 1

    print("Ingestion completed successfully.")
    print("Records by source:")
    for source_type, count in counts.items():
        print(f"  {source_type}: {count}")

    print(f"Total records: {len(documents)}")
    print(f"Output file: {OUTPUT_FILE.relative_to(ROOT)}")
    print("Note: traffic records are synthetic, not real observations.")


if __name__ == "__main__":
    main()
