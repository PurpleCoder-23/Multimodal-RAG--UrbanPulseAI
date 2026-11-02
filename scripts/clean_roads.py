import json
from pathlib import Path
from collections import Counter

INPUT_FILE = Path("data/maps/hyderabad_roads.geojson")
OUTPUT_FILE = Path("data/maps/hyderabad_road_segments.geojson")

# Keep road-related features represented by line geometries.
ROAD_TYPES = {
    "motorway", "motorway_link",
    "trunk", "trunk_link",
    "primary", "primary_link",
    "secondary", "secondary_link",
    "tertiary", "tertiary_link",
    "residential", "living_street",
    "service", "unclassified", "road",
    "construction",
}

with INPUT_FILE.open("r", encoding="utf-8") as file:
    geojson = json.load(file)

clean_features = []

for feature in geojson.get("features", []):
    geometry = feature.get("geometry") or {}
    properties = feature.get("properties") or {}
    highway_type = properties.get("highway")

    if (
        geometry.get("type") in {"LineString", "MultiLineString"}
        and highway_type in ROAD_TYPES
    ):
        clean_features.append(feature)

clean_geojson = {
    "type": "FeatureCollection",
    "features": clean_features,
}

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT_FILE.open("w", encoding="utf-8") as file:
    json.dump(clean_geojson, file, ensure_ascii=False)

counts = Counter(
    feature["properties"]["highway"]
    for feature in clean_features
)

print("Road cleaning completed.")
print("Input features:", len(geojson.get("features", [])))
print("Road segments retained:", len(clean_features))
print("Output file:", OUTPUT_FILE)
print("Road categories:")

for category, count in counts.most_common():
    print(f"  {category}: {count}")
