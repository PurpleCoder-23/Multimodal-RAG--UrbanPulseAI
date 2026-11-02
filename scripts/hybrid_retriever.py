
import json
import re
from pathlib import Path
from datetime import datetime

from geospatial_retriever import GeospatialRetriever
from sentence_transformers import SentenceTransformer
from answer_generator import AnswerGenerator
from intent_router import IntentRouter

from retrieve_multimodal import (
    MODEL_NAME,
    load_index,
    retrieve as semantic_retrieve,
)

from structured_retriever import StructuredRetriever


class HybridRetriever:

    def __init__(self, top_k=5):

        print("Loading UrbanPulse AI retrieval systems...")

        # 1. Load semantic retrieval components
        self.embeddings, self.metadata = load_index()
        self.model = SentenceTransformer(MODEL_NAME)

        # 2. Load structured retrieval components
        self.structured_retriever = StructuredRetriever()

        # 3. Load geospatial retrieval components
        self.geospatial_retriever = GeospatialRetriever()

        self.top_k = top_k

        # 4. Load traffic advisory metadata
        self.advisory_metadata = self.load_advisory_metadata()

        print("Hybrid retrieval system ready.")

    def needs_semantic_retrieval(self, query):

        keywords = [
            "liberty",
            "basheerbagh",
            "tank bund",
            "saifabad",
            "ravindra bharathi",
            "htp junction",
            "nallagutta",
            "ambedkar statue",
            "vv statue",
            "ar petrol pump",
            "bjr statue",
            "advisory",
            "diversion",
            "junction",
            "avoid",
            "road closure",
            "closed road",
            "immersion",
            "traffic police",
            "road restriction",
        ]

        return any(word in query.lower() for word in keywords)

    def extract_coordinates(self, query):

        pattern = (
            r"\blat(?:itude)?\s*[:=]\s*(-?\d+(?:\.\d+)?)"
            r"\s*[,;]\s*"
            r"lon(?:gitude)?\s*[:=]\s*(-?\d+(?:\.\d+)?)"
        )

        match = re.search(pattern, query, re.IGNORECASE)

        if not match:
            return None

        latitude = float(match.group(1))
        longitude = float(match.group(2))

        if not (-90 <= latitude <= 90):
            return None

        if not (-180 <= longitude <= 180):
            return None

        radius_match = re.search(
            r"\bwithin\s+(\d+(?:\.\d+)?)\s*(?:m|meters?)\b",
            query,
            re.IGNORECASE,
        )

        radius = float(radius_match.group(1)) if radius_match else 500

        return latitude, longitude, radius

    def retrieve(self, query):

        query = query.strip()

        if not query:
            return {"error": "Please enter a valid question."}

        # Initialize evidence dictionary
        evidence = {
            "query": query,
            "retrieval_methods": [],
            "semantic_evidence": [],
            "structured_evidence": [],
            "geospatial_evidence": [],
            "advisory_metadata": None,
        }

        query_lower = query.lower()

        # --------------------------------------------------
        # 1. SEMANTIC RETRIEVAL
        # --------------------------------------------------

        if self.needs_semantic_retrieval(query):

            semantic_results = semantic_retrieve(
                query,
                self.model,
                self.embeddings,
                self.metadata,
                top_k=self.top_k,
            )

            evidence["semantic_evidence"] = semantic_results
            evidence["retrieval_methods"].append("semantic")

            # Attach advisory metadata when relevant semantic
            # evidence has been retrieved.
            if semantic_results and self.advisory_metadata:

                evidence["advisory_metadata"] = (
                    self.advisory_metadata
                )

        # --------------------------------------------------
        # 2. STRUCTURED RETRIEVAL
        # --------------------------------------------------

        weather_keywords = [
            "weather",
            "temperature",
            "humidity",
            "rain",
            "precipitation",
            "wind",
        ]

        congestion_keywords = [
            "heavy",
            "moderate",
            "light",
            "congestion",
        ]

        needs_weather = any(
            word in query_lower for word in weather_keywords
        )

        needs_congestion = any(
            word in query_lower for word in congestion_keywords
        )

        # Both weather and congestion requested
        if needs_weather and needs_congestion:

            weather_result = (
                self.structured_retriever.weather_summary()
            )

            congestion_level = next(
                (
                    level
                    for level in ["heavy", "moderate", "light"]
                    if level in query_lower
                ),
                None,
            )

            if congestion_level:

                traffic_result = (
                    self.structured_retriever.filter_congestion(
                        congestion_level
                    )
                )

            else:

                traffic_result = (
                    self.structured_retriever.traffic_summary()
                )

            evidence["structured_evidence"] = {
                "weather": weather_result,
                "traffic": traffic_result,
            }

            evidence["retrieval_methods"].append("structured")

        # Only weather or congestion requested
        elif needs_weather or needs_congestion:

            structured_result = (
                self.structured_retriever.retrieve(query)
            )

            evidence["structured_evidence"] = structured_result
            evidence["retrieval_methods"].append("structured")

        # Check other supported structured queries
        else:

            structured_result = (
                self.structured_retriever.retrieve(query)
            )

            if (
                isinstance(structured_result, dict)
                and "message" not in structured_result
            ):

                evidence["structured_evidence"] = structured_result
                evidence["retrieval_methods"].append("structured")

        # --------------------------------------------------
        # 3. GEOSPATIAL RETRIEVAL
        # --------------------------------------------------

        coordinates = self.extract_coordinates(query)

        if coordinates:

            latitude, longitude, radius = coordinates

            nearby_roads = (
                self.geospatial_retriever.find_nearby_roads(
                    latitude=latitude,
                    longitude=longitude,
                    radius_meters=radius,
                    top_k=self.top_k,
                )
            )

            evidence["geospatial_evidence"] = nearby_roads
            evidence["retrieval_methods"].append("geospatial")

        # --------------------------------------------------
        # 4. CHECK RETRIEVAL ROUTES
        # --------------------------------------------------

        if not evidence["retrieval_methods"]:

            evidence["message"] = (
                "No matching retrieval route was identified. "
                "Try asking about weather, traffic congestion, "
                "traffic statistics, diversions or junctions."
            )

        # IMPORTANT:
        # This return belongs to retrieve(), not to the
        # geospatial if-block or metadata-loading method.
        return evidence

    # ======================================================
    # LOAD ADVISORY METADATA
    # ======================================================

    def load_advisory_metadata(self):

        project_root = Path(__file__).resolve().parent.parent

        advisory_path = (
            project_root
            / "data"
            / "Advisories"
            / "durga_matha_immersion_2026.json"
        )

        try:

            with open(
                advisory_path,
                "r",
                encoding="utf-8",
            ) as file:

                advisory = json.load(file)

            validity = advisory["published_notice_period"]

            valid_from = datetime.fromisoformat(
                validity["start"]
            )

            valid_until = datetime.fromisoformat(
                validity["end"]
            )

            # Calculate status using the notice's timezone
            current_time = datetime.now(valid_until.tzinfo)

            if current_time < valid_from:

                status = "upcoming"

            elif current_time <= valid_until:

                status = "active"

            else:

                status = "historical_expired"

            metadata = {
                "advisory_id": advisory["advisory_id"],
                "event": advisory["title"],
                "issuing_authority": advisory["source"],
                "valid_from": validity["start"],
                "valid_until": validity["end"],
                "status": status,
                "status_checked_at": current_time.isoformat(
                    timespec="minutes"
                ),
                "affected_locations": advisory["affected_locations"],
                "junctions_to_avoid": advisory["junctions_to_avoid"],
                "expected_congestion": advisory["expected_congestion"],
                "source_url": advisory["source_url"],
            }

            print("Advisory metadata loaded successfully.")
            print(f"Advisory status: {status}")

            return metadata

        except FileNotFoundError:

            print("Warning: Advisory JSON file not found.")
            return None

        except (
            KeyError,
            json.JSONDecodeError,
            ValueError,
        ) as error:

            print(
                f"Warning: Could not load advisory metadata: {error}"
            )

            return None


# ==========================================================
# MAIN CONVERSATIONAL APPLICATION
# ==========================================================

def main():

    retriever = HybridRetriever(top_k=5)

    answer_generator = AnswerGenerator()

    intent_router = IntentRouter()

    print("\nUrbanPulse AI - Conversational Hybrid Retrieval")
    print("Type 'exit' to stop.")

    while True:

        query = input("\nYou: ").strip()

        if query.lower() in {"exit", "quit"}:

            print("\nUrbanPulse AI: Goodbye!")
            break

        if not query:
            continue

        # 1. Identify user intent
        intent = intent_router.classify(query)

        # 2. Handle conversational queries without retrieval
        if intent in {
            "greeting",
            "identity",
            "capabilities",
            "out_of_domain",
        }:

            response = intent_router.response(intent)

            print(f"\nUrbanPulse AI: {response}")

            continue

        if intent == "empty":
            continue

        # 3. Retrieve evidence for domain-specific questions
        result = retriever.retrieve(query)

        # 4. Generate answer using retrieved evidence
        answer = answer_generator.generate_answer(result)

        # 5. Display retrieved evidence
        print("\n--- Retrieved Evidence ---")

        print(
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False,
            )
        )

        # 6. Display final answer
        print("\nUrbanPulse AI:")
        print(answer)


if __name__ == "__main__":
    main()
