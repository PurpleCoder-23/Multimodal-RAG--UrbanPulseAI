
import re
from typing import Annotated, TypedDict

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
)

from langgraph.graph import (
    StateGraph,
    START,
    END,
    add_messages,
)

from langgraph.checkpoint.memory import InMemorySaver

from hybrid_retriever import HybridRetriever
from answer_generator import AnswerGenerator
from intent_router import IntentRouter


# ==========================================================
# 1. DEFINE GRAPH STATE
# ==========================================================

class UrbanPulseState(TypedDict, total=False):

    # Stores conversation messages
    messages: Annotated[list, add_messages]

    # Current user query
    query: str

    # Classified intent
    intent: str

    # Retrieved information
    evidence: dict

    # Final generated response
    answer: str


# ==========================================================
# 2. EXTRACT MESSAGE TEXT
# ==========================================================

def get_message_text(message):

    content = getattr(message, "content", "")

    if isinstance(content, str):
        return content

    if isinstance(content, list):

        text_parts = []

        for part in content:

            if isinstance(part, dict):
                text_parts.append(part.get("text", ""))

        return " ".join(text_parts)

    return str(content)


# ==========================================================
# 3. BUILD URBANPULSE AGENT
# ==========================================================

def build_urbanpulse_agent():

    print("Initializing UrbanPulse AI Agent...")

    # Initialize existing components only once
    retriever = HybridRetriever(top_k=5)

    answer_generator = AnswerGenerator()

    intent_router = IntentRouter()

    # ------------------------------------------------------
    # NODE 1: INTENT CLASSIFICATION
    # ------------------------------------------------------

    def classify_intent(state: UrbanPulseState):

        messages = state.get("messages", [])

        # Find the latest user message
        human_messages = [
            message
            for message in messages
            if isinstance(message, HumanMessage)
        ]

        if not human_messages:

            return {
                "query": "",
                "intent": "empty",
            }

        current_query = get_message_text(
            human_messages[-1]
        )

        intent = intent_router.classify(current_query)

        # Detect simple contextual follow-up questions
        follow_up_pattern = (
            r"\b(what about|how about|is it still|"
            r"are they still|does that|tell me more|"
            r"what does that|is that|why is that|"
            r"when does it)\b"
        )

        is_follow_up = bool(
            re.search(
                follow_up_pattern,
                current_query.lower()
            )
        )

        # Find the previous UrbanPulse domain question
        previous_domain_query = None

        if is_follow_up and len(human_messages) > 1:

            previous_messages = human_messages[:-1]

            for message in reversed(previous_messages):

                previous_query = get_message_text(message)

                # Skip earlier follow-up questions
                if re.search(
                    follow_up_pattern,
                    previous_query.lower()
                ):
                    continue

                previous_intent = intent_router.classify(
                    previous_query
                )

                if previous_intent == "domain":

                    previous_domain_query = previous_query
                    break
                    # Resolve the follow-up using previous context
        if previous_domain_query:

            resolved_query = (
                f"{previous_domain_query}\n"
                f"Follow-up question: {current_query}"
            )

            intent = "domain"

        else:

            resolved_query = current_query

        print(f"\nDetected intent: {intent}")

        return {
            "query": resolved_query,
            "intent": intent,
        }

    # ------------------------------------------------------
    # CONDITIONAL ROUTING
    # ------------------------------------------------------

    def route_after_classification(state: UrbanPulseState):

        if state.get("intent") == "domain":

            return "retrieve"

        return "conversation"

    # ------------------------------------------------------
    # NODE 2: CONVERSATIONAL RESPONSE
    # ------------------------------------------------------

    def conversation_response(state: UrbanPulseState):

        intent = state.get("intent", "out_of_domain")

        response = intent_router.response(intent)

        if not response:

            response = (
                "Hello! How can I help you with "
                "UrbanPulse AI today?"
            )

        return {
            "answer": response,
            "messages": [
                AIMessage(content=response)
            ],
        }

    # ------------------------------------------------------
    # NODE 3: HYBRID RETRIEVAL
    # ------------------------------------------------------

    
    def retrieve_evidence(state: UrbanPulseState):

        query = state.get("query", "")

        print("\n========== RETRIEVAL DEBUG ==========")

        print("\nResolved query sent to retriever:")
        print(query)

        try:

            print("\nRetrieving relevant evidence...")

            evidence = retriever.retrieve(query)

            print("\nRetrieval methods:")
            print(evidence.get("retrieval_methods", []))

            print("\nSemantic evidence:")

            for item in evidence.get("semantic_evidence", []):
                print("-", item.get("content"))

            print("\nAdvisory metadata:")
            print(evidence.get("advisory_metadata"))

            print("\n========== END DEBUG ==========\n")

            return {
                "evidence": evidence
            }

        except Exception as error:

            print(f"Retrieval error: {error}")

            return {
                "evidence": {
                    "error": str(error)
                }
            }

    # ------------------------------------------------------
    # NODE 4: ANSWER GENERATION
    # ------------------------------------------------------

    def generate_answer(state: UrbanPulseState):

        evidence = state.get("evidence", {})

        print("Generating grounded response...")

        if evidence.get("error"):

            answer = (
                "I encountered an error while retrieving "
                "the requested information. Please try again."
            )

        else:

            try:

                answer = answer_generator.generate_answer(
                    evidence
                )

            except Exception as error:

                print(f"Answer generation error: {error}")

                answer = (
                    "I encountered an error while generating "
                    "the response. Please try again."
                )

        return {
            "answer": answer,
            "messages": [
                AIMessage(content=answer)
            ],
        }

    # ======================================================
    # 4. CREATE LANGGRAPH WORKFLOW
    # ======================================================

    workflow = StateGraph(UrbanPulseState)

    # Register nodes
    workflow.add_node(
        "classify_intent",
        classify_intent
    )

    workflow.add_node(
        "conversation_response",
        conversation_response
    )

    workflow.add_node(
        "retrieve_evidence",
        retrieve_evidence
    )

    workflow.add_node(
        "generate_answer",
        generate_answer
    )

    # Starting node
    workflow.add_edge(
        START,
        "classify_intent"
    )

    # Conditional routing
    workflow.add_conditional_edges(
        "classify_intent",
        route_after_classification,
        {
            "conversation": "conversation_response",
            "retrieve": "retrieve_evidence",
        }
    )

    # Conversation route
    workflow.add_edge(
        "conversation_response",
        END
    )

    # Retrieval route
    workflow.add_edge(
        "retrieve_evidence",
        "generate_answer"
    )

    workflow.add_edge(
        "generate_answer",
        END
    )

    # ------------------------------------------------------
    # 5. ADD SHORT-TERM MEMORY
    # ------------------------------------------------------

    memory = InMemorySaver()

    agent = workflow.compile(
        checkpointer=memory
    )

    print("UrbanPulse AI Agent is ready.")

    return agent


# ==========================================================
# 6. RUN CONVERSATIONAL APPLICATION
# ==========================================================

def main():

    agent = build_urbanpulse_agent()

    # Same thread_id preserves conversation between turns
    config = {
        "configurable": {
            "thread_id": "urbanpulse-user-session"
        }
    }

    print("\n======================================")
    print("       URBANPULSE AI AGENT")
    print("======================================")

    print("Ask me about Hyderabad traffic.")
    print("Type 'exit' to stop.")

    while True:

        query = input("\nYou: ").strip()

        if query.lower() in {"exit", "quit"}:

            print("\nUrbanPulse AI: Goodbye!")
            break

        if not query:
            continue

        try:

            result = agent.invoke(
                {
                    "messages": [
                        HumanMessage(content=query)
                    ]
                },
                config=config
            )

            print("\nUrbanPulse AI:")

            print(result.get("answer", "No response generated."))

        except Exception as error:

            print(f"\nAgent execution error: {error}")


if __name__ == "__main__":

    main()
