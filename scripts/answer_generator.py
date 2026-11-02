
import os
import json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from groq import Groq


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_NAME = "qwen/qwen3.8-27b"


class AnswerGenerator:

    def __init__(self):

        # Load environment variables
        load_dotenv(PROJECT_ROOT / ".env")

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found. Check your .env file."
            )

        self.client = Groq(api_key=api_key)

        self.model = MODEL_NAME

        print("LLM answer generator initialized.")


    
    def compact_evidence(self, retrieved_evidence):

        compact = {
            "query": retrieved_evidence.get("query"),
            "retrieval_methods": retrieved_evidence.get(
                "retrieval_methods", []
            ),
            "advisory_metadata": retrieved_evidence.get(
                "advisory_metadata"
            ),
            "semantic_evidence": [],
            "structured_evidence": {},
            "geospatial_evidence": []
        }

        # 1. Keep only the top semantic results.
        for item in retrieved_evidence.get(
            "semantic_evidence", []
        )[:5]:

            compact["semantic_evidence"].append({
                "record_id": item.get("record_id"),
                "evidence_type": item.get("evidence_type"),
                "content": item.get("content"),
                "similarity": item.get("similarity")
            })

        # 2. Compress structured evidence.
        structured = retrieved_evidence.get(
            "structured_evidence", {}
        )

        def summarize(value):

            if isinstance(value, dict):

                result = {}

                for key, item in value.items():

                    if key == "records" and isinstance(item, list):

                        result["record_count"] = value.get(
                            "record_count", len(item)
                        )

                        result["sample_records"] = item[:3]

                    else:
                        result[key] = summarize(item)

                return result

            if isinstance(value, list):
                return [summarize(item) for item in value[:5]]

            return value

        compact["structured_evidence"] = summarize(structured)

        # 3. Keep the nearest road segments.
        for road in retrieved_evidence.get(
            "geospatial_evidence", []
        )[:5]:

            compact["geospatial_evidence"].append(road)

        return compact


    def generate_answer(self, retrieved_evidence):

        if not retrieved_evidence:
            return "No retrieved evidence was provided."

        query = retrieved_evidence.get("query", "")

        if not retrieved_evidence.get("retrieval_methods"):
            return (
                "I could not find sufficient relevant evidence "
                "to answer your question."
            )

        compact = self.compact_evidence(retrieved_evidence)

        evidence_text = json.dumps(
            compact,
            indent=2,
            ensure_ascii=False
        )
        # Current India time is temporal context only; it is not traffic evidence.
        now = datetime.now(ZoneInfo("Asia/Kolkata"))
        current_time = now.isoformat(timespec="minutes")
        current_date = now.date().isoformat()

        system_prompt = """

        You are UrbanPulse AI, a traffic information assistant.

        Answer using only the retrieved evidence. The current date/time
        supplied in the user message is trusted temporal context, not
        traffic, weather, advisory, or location evidence.

        Rules:
        1. Do not invent traffic conditions, closures, diversions,
           locations, neighbourhoods, or weather information.
        2. If evidence is insufficient, state what is missing.
        3. Clearly distinguish synthetic/simulated records from
           real-world observations. Never present synthetic records
           as live traffic conditions.
        4. Interpret timestamps relative to the supplied current date/time.
           Do not call a timestamp "future" unless it is later than now.
           If timezone or timestamp meaning is unclear, state that limitation.
        5. Identify historical advisories as historical. Never describe
           an expired advisory as a current restriction.
        6. If advisory validity metadata is provided, state its validity
           period and whether it has expired as of the supplied date.
           If metadata is absent, do not guess the advisory's current status.
        7. Do not infer the neighbourhood or geographic area of a named
           place unless retrieved evidence explicitly states it.
        8. Do not claim two datasets or locations are geographically
           connected unless evidence includes a spatial relationship
           or an explicit spatial join.
        9. Do not claim a road is congested based only on road geometry.
           Road geometry describes roads, not traffic.
        10. For coordinate-based results, report only the coordinates,
            radius, road names, and distances actually supplied. Do not
            overstate the geographic coverage of the road dataset.
        11. Combine findings from multiple retrieval methods logically,
            but keep unrelated evidence separate.
        12. Be clear and concise. Include only supported dates, places,
            road names, and values.
        Never claim that no active advisories exist across the entire
        city unless the system has verified a sufficiently complete,
        current advisory dataset.

        Instead, say:
        "No active advisory was found in the retrieved evidence.
        The available advisory record is historical and expired."

        WEATHER DATA HANDLING:
        - Distinguish historical snapshots from synthetic data.
        - Never describe a saved weather snapshot as simulated
        unless its source explicitly identifies it as synthetic.
        - Always report the recorded timestamp.
        - Never present historical weather information as current.
        - State explicitly when live weather information is unavailable.

        DIRECT QUESTION ANSWERING:

        1. Always answer the specific question asked by the user first.

        2. When the user asks which junctions should be avoided,
        explicitly list ALL junctions available in:

        advisory_metadata["junctions_to_avoid"]

        3. Do not list only a few examples when the complete
        junction list is available.

        4. If the advisory is expired, clearly label the list
        as historical restrictions, not current restrictions.

        5. Explain the advisory validity period.

        6. Never replace the requested junction list with
        a general statement about the advisory expiration.

        7. If the user asks for current junction restrictions,
        distinguish historical advisory information from
        verified current traffic conditions.

        8. Keep the answer focused on the user's actual question.
        """

        user_prompt = f"""
        CURRENT DATE/TIME (Asia/Kolkata):
        {current_time}
        CURRENT DATE:
        {current_date}

        This is temporal context only. Do not use it to invent current
        traffic, weather, or advisory information.

        USER QUESTION:
        {query}

        RETRIEVED EVIDENCE:
        {evidence_text}

        Generate an answer grounded strictly in the retrieved evidence.
        """

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            temperature=0.2,
            max_completion_tokens=1500
        )

        answer = response.choices[0].message.content

        if not answer:
            raise RuntimeError(
                "LLM returned an empty response."
            )

        return answer.strip()



#Temporary Test code

if __name__ == "__main__":

    generator = AnswerGenerator()

    # Sample evidence similar to the output
    # produced by our semantic retriever.
    sample_evidence = {
        "query": "What diversion applies near Liberty?",
        "retrieval_methods": ["semantic"],
        "advisory_metadata": {
                "event": "Durga Matha Idols Immersion",
                "issuing_authority": "Hyderabad Traffic Police",
                "valid_from": "21 October 2026, 15:00",
                "valid_until": "24 October 2026, 06:00",
                "status": "historical_expired"
            },
        "semantic_evidence": [
            {
                "evidence_type": "traffic_diversion",
                "content": (
                    "Traffic coming from Upper Tank Bund "
                    "towards Secretariat Junction will be "
                    "diverted at Old Ambedkar Statue "
                    "towards Liberty."
                )
            },
            
            {
                "evidence_type": "traffic_diversion",
                "content": (
                    "Traffic coming from Liberty will not "
                    "be allowed towards Secretariat Junction "
                    "and will be diverted at Old Ambedkar "
                    "Statue towards Upper Tank Bund."
                )
            }
        ],
        "structured_evidence": [],
        "geospatial_evidence": []
    }

    answer = generator.generate_answer(sample_evidence)

    print("\n--- UrbanPulse AI Answer ---")
    print(answer)
