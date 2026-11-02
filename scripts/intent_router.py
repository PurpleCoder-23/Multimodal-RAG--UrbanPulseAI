import re


class IntentRouter:
    """
    Identifies whether a user query is conversational,
    related to UrbanPulse AI, or outside the application scope.
    """

    GREETINGS = {
        "hi",
        "hello",
        "hey",
        "hii",
        "hiii",
        "good morning",
        "good afternoon",
        "good evening",
        "good night",
        "how are you",
        "hey there",
    }

    DOMAIN_KEYWORDS = (
        "traffic",
        "congestion",
        "vehicle",
        "vehicles",
        "road",
        "roads",
        "junction",
        "diversion",
        "advisory",
        "weather",
        "temperature",
        "humidity",
        "rain",
        "precipitation",
        "wind",
        "route",
        "routes",
        "commute",
        "commuter",
        "travel",
        "closure",
        "closed",
        "accident",
        "incident",
        "immersion",
        "traffic police",
        "liberty",
        "basheerbagh",
        "ravindra bharathi",
        "tank bund",
        "saifabad",
        "hitec city",
        "kondapur",
    )

    def classify(self, query: str) -> str:

        cleaned = re.sub(
            r"[!?.,]+$",
            "",
            query.strip().lower()
        )

        if not cleaned:
            return "empty"

        # 1. Greeting
        if cleaned in self.GREETINGS:
            return "greeting"

        # 2. Identity-related questions
        if re.search(
            r"\b(who are you|what are you|introduce yourself|"
            r"are you urbanpulse|what is urbanpulse)\b",
            cleaned,
        ):
            return "identity"

        # 3. Capability-related questions
        if re.search(
            r"\b(what can you do|what are your capabilities|"
            r"how can you help|what do you do|your features)\b",
            cleaned,
        ):
            return "capabilities"

        # 4. Coordinate-based questions
        has_coordinates = bool(
            re.search(
                r"\blat(?:itude)?\s*[:=]\s*-?\d",
                cleaned
            )
        )

        if has_coordinates:
            return "domain"

        # 5. UrbanPulse domain questions
        if any(
            keyword in cleaned
            for keyword in self.DOMAIN_KEYWORDS
        ):
            return "domain"

        # 6. Other questions
        return "out_of_domain"

    @staticmethod
    def response(intent: str) -> str:

        responses = {

            "greeting": (
                "Hello! I'm UrbanPulse AI. I can help you explore "
                "Hyderabad traffic information, road advisories, "
                "diversions, weather snapshots and nearby road data. "
                "What would you like to know?"
            ),

            "identity": (
                "I'm UrbanPulse AI, a traffic-information assistant "
                "designed to answer questions using retrieved traffic, "
                "advisory, weather and road-network data."
            ),

            "capabilities": (
                "I can help with traffic summaries and congestion records, "
                "weather snapshots, traffic-police advisories and diversions, "
                "junction information, and nearby road-network searches "
                "when you provide latitude and longitude. I distinguish "
                "synthetic traffic records from real observations and "
                "report when supporting evidence is unavailable."
            ),

            "out_of_domain": (
                "I'm designed for UrbanPulse AI topics: Hyderabad traffic, "
                "road networks, traffic advisories and diversions, and the "
                "available weather and traffic datasets. Please ask me "
                "something related to those areas."
            ),
        }

        return responses.get(intent, "")