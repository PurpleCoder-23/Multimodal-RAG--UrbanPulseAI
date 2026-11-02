
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAFFIC_PATH = (
    PROJECT_ROOT / "data" / "traffic" / "synthetic_traffic_records.csv"
)

WEATHER_PATH = (
    PROJECT_ROOT / "data" / "Weather" / "hyderabad_weather.json"
)


class StructuredRetriever:

    def __init__(self):
        self.traffic = pd.read_csv(TRAFFIC_PATH)

        with WEATHER_PATH.open(
            "r", encoding="utf-8-sig"
        ) as file:
            self.weather = json.load(file)

        self.current_weather = self.weather["current"]

    def traffic_summary(self):
        """Calculate overall simulated traffic statistics."""

        return {
            "record_count": len(self.traffic),
            "average_speed_kmh": round(
                self.traffic["speed_kmh"].mean(), 2
            ),
            "average_vehicles_per_minute": round(
                self.traffic["vehicles_per_minute"].mean(), 2
            ),
            "congestion_distribution": (
                self.traffic["congestion_level"]
                .value_counts()
                .to_dict()
            ),
            "data_origin": "synthetic",
            "is_real_observation": False,
        }

    def filter_congestion(self, level):
        """Retrieve records for a congestion category."""

        level = level.lower()

        results = self.traffic[
            self.traffic["congestion_level"].str.lower() == level
        ]

        return {
            "query_type": "congestion_filter",
            "filter": level,
            "record_count": len(results),
            "records": results.to_dict(orient="records"),
            "data_origin": "synthetic",
            "is_real_observation": False,
        }

    def search_location(self, location):
        """Search traffic records by location name."""

        results = self.traffic[
            self.traffic["location"].str.contains(
                location, case=False, na=False
            )
        ]

        return {
            "query_type": "location_search",
            "location": location,
            "record_count": len(results),
            "records": results.to_dict(orient="records"),
            "data_origin": "synthetic",
            "is_real_observation": False,
        }

    def weather_summary(self):
        """Retrieve the saved weather snapshot."""

        return {
            "temperature_c": self.current_weather["temperature_2m"],
            "humidity_percent": (
                self.current_weather["relative_humidity_2m"]
            ),
            "precipitation_mm": (
                self.current_weather["precipitation"]
            ),
            "wind_speed_kmh": (
                self.current_weather["wind_speed_10m"]
            ),
            "snapshot_time": self.current_weather["time"],
            "timezone": self.weather["timezone"],
            "data_origin": "saved_weather_snapshot",
        }

    def retrieve(self, query):
        """Route simple structured queries to the appropriate method."""

        query = query.strip().lower()

        if any(word in query for word in ["weather", "temperature",
                                           "humidity", "rain",
                                           "precipitation", "wind"]):
            return self.weather_summary()

        if "heavy" in query:
            return self.filter_congestion("heavy")

        if "moderate" in query:
            return self.filter_congestion("moderate")

        if "light" in query:
            return self.filter_congestion("light")

        if any(word in query for word in ["average", "summary",
                                           "statistics", "overall"]):
            return self.traffic_summary()

        # Search for a location mentioned in the query.
        for location in self.traffic["location"].dropna().unique():
            if location.lower() in query:
                return self.search_location(location)

        return {
            "message": (
                "Query not recognized. Try weather, heavy congestion, "
                "traffic summary, or a location name."
            )
        }


def main():
    retriever = StructuredRetriever()

    print("UrbanPulse AI - Structured Retrieval")
    print("Available: weather, heavy congestion, traffic summary")
    print("You can also enter a location name.")
    print("Type 'exit' to stop.")

    while True:
        query = input("\nEnter your query: ").strip()

        if query.lower() in {"exit", "quit"}:
            break

        if not query:
            continue

        result = retriever.retrieve(query)

        print("\n--- Structured Retrieval Result ---")
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
