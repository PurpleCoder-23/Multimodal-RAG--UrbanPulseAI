
import json
from pathlib import Path

from shapely.geometry import shape, Point
from shapely.strtree import STRtree
from shapely.ops import transform
from pyproj import Transformer


class GeospatialRetriever:

    def __init__(self):

        project_root = Path(__file__).resolve().parent.parent

        geojson_path = (
            project_root
            / "data"
            / "Maps"
            / "hyderabad_road_segments.geojson"
        )

        with open(geojson_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        self.roads = []
        self.properties = []

        for feature in data["features"]:

            geometry = shape(feature["geometry"])

            if geometry.is_empty:
                continue

            self.roads.append(geometry)
            self.properties.append(feature.get("properties", {}))

        # Convert geographic coordinates (WGS84)
        # into metric coordinates (UTM Zone 44N).
        self.to_metric = Transformer.from_crs(
            "EPSG:4326",
            "EPSG:32644",
            always_xy=True
        ).transform

        self.metric_roads = [
            transform(self.to_metric, road)
            for road in self.roads
        ]

        self.spatial_index = STRtree(self.metric_roads)

        print(f"Loaded {len(self.metric_roads)} road segments.")

    def find_nearby_roads(
        self,
        latitude,
        longitude,
        radius_meters=500,
        top_k=10
    ):

        # Shapely Point expects (longitude, latitude).
        location = Point(longitude, latitude)

        # Convert the query location into metric coordinates.
        metric_location = transform(self.to_metric, location)

        # Find candidate roads using the spatial index.
        candidate_indices = self.spatial_index.query(
            metric_location.buffer(radius_meters)
        )

        results = []

        for index in candidate_indices:

            road = self.metric_roads[index]

            distance = metric_location.distance(road)

            if distance <= radius_meters:

                properties = self.properties[index]

                results.append({
                    "road_name": properties.get("name", "Unnamed road"),
                    "road_type": properties.get("highway", "Unknown"),
                    "distance_meters": round(distance, 2),
                    "maxspeed": properties.get("maxspeed"),
                    "lanes": properties.get("lanes"),
                    "oneway": properties.get("oneway"),
                    "surface": properties.get("surface")
                })

        results.sort(key=lambda item: item["distance_meters"])

        return results[:top_k]


if __name__ == "__main__":

    retriever = GeospatialRetriever()

    # Test coordinates near the road extract.
    results = retriever.find_nearby_roads(
        latitude=17.4526369,
        longitude=78.3800073,
        radius_meters=500,
        top_k=10
    )

    print("\nNearby roads:")

    for road in results:
        print(road)
