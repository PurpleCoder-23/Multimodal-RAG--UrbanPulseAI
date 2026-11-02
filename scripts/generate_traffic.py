
import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

# Reproducible synthetic data: same seed, same generated records.
random.seed(42)

OUTPUT_FILE = Path("data/traffic/synthetic_traffic_records.csv")
RECORD_COUNT = 500

# These names came from our Hyderabad-area map inspection.
ROADS = [
    {"road": "HITEC City Road", "road_type": "primary", "base_speed": 40},
    {"road": "HITEC City-Kondapur Main Road", "road_type": "primary", "base_speed": 38},
    {"road": "Inorbit Mall Road", "road_type": "secondary", "base_speed": 30},
    {"road": "Durgam Cheruvu Road", "road_type": "secondary", "base_speed": 32},
    {"road": "Ayyappa Society Main Road", "road_type": "tertiary", "base_speed": 25},
    {"road": "Image Garden Road", "road_type": "residential", "base_speed": 20},
    {"road": "Road Number 13", "road_type": "residential", "base_speed": 18},
]

START = datetime(2026, 10, 26, 0, 0)
END = datetime(2026, 10, 29, 0, 0)
TOTAL_MINUTES = int((END - START).total_seconds() // 60)


def congestion_label(speed):
    if speed < 15:
        return "heavy"
    if speed < 25:
        return "moderate"
    return "light"


OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

fieldnames = [
    "record_id",
    "simulation_time",
    "location",
    "road_type",
    "speed_kmh",
    "vehicles_per_minute",
    "congestion_level",
    "record_origin",
    "is_real_observation",
]

with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()

    for i in range(1, RECORD_COUNT + 1):
        road = random.choice(ROADS)
        timestamp = START + timedelta(
            minutes=random.randrange(TOTAL_MINUTES)
        )

        # Simulated rush-hour effect, not measured Hyderabad traffic.
        hour = timestamp.hour
        rush_hour = hour in range(8, 11) or hour in range(17, 21)
        speed_adjustment = -random.uniform(5, 15) if rush_hour else 0

        speed = round(
            max(5, min(70, road["base_speed"] + speed_adjustment
                        + random.uniform(-8, 8))),
            1,
        )
        vehicles = random.randint(3, 35)

        writer.writerow({
            "record_id": f"SYN-{i:04d}",
            "simulation_time": timestamp.isoformat(),
            "location": road["road"],
            "road_type": road["road_type"],
            "speed_kmh": speed,
            "vehicles_per_minute": vehicles,
            "congestion_level": congestion_label(speed),
            "record_origin": "synthetic",
            "is_real_observation": "false",
        })

print("Synthetic traffic dataset created.")
print("Records:", RECORD_COUNT)
print("Output:", OUTPUT_FILE)
print("Source: generated simulation, not observed traffic")
