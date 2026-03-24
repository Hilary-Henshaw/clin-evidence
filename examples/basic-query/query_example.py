"""Basic ClinEvidence query example using httpx."""
from __future__ import annotations

import sys
import uuid

import httpx

BASE_URL = "http://localhost:8000"
SESSION_ID = str(uuid.uuid4())


def check_health(client: httpx.Client) -> bool:
    """Verify the server is healthy before querying."""
    response = client.get(f"{BASE_URL}/health")
    if response.status_code != 200:
        return False
    data = response.json()
    status_ok = data.get("status") == "ok"
    kb_ready = data.get("knowledge_base_ready", False)
    print(f"Server status: {data['status']}")
    print(f"Version: {data['version']}")
    print(f"Knowledge base ready: {kb_ready}")
    if not kb_ready:
        print(
            "Note: Knowledge base has no documents. "
            "Responses will use web evidence."
        )
    return status_ok


def submit_query(
    client: httpx.Client,
    message: str,
    session_id: str,
) -> dict[str, object]:
    """Submit a clinical query and return the response."""
    payload = {
        "message": message,
        "session_id": session_id,
    }
    response = client.post(
        f"{BASE_URL}/v1/chat",
        json=payload,
        timeout=60.0,
    )
    response.raise_for_status()
    return response.json()  # type: ignore[no-any-return]


def display_response(
    question: str,
    result: dict[str, object],
) -> None:
    """Pretty-print a ClinEvidence response."""
    print("\n" + "=" * 60)
    print("Clinical Query Response")
    print("=" * 60)
    print(f"\nQuestion: {question}\n")
    print(f"Agent:          {result.get('agent_used')}")
    confidence = result.get("confidence")
    if confidence is not None:
        print(f"Confidence:     {float(confidence):.0%}")
    print(
        f"Processing time: "
        f"{result.get('processing_time_ms')}ms"
    )
    requires_val = result.get("requires_validation", False)
    if requires_val:
        print("Validation:     REQUIRED")

    print("\nResponse:")
    print("-" * 40)
    print(result.get("message", ""))

    sources = result.get("sources", [])
    if sources:
        print("\nSources:")
        for i, src in enumerate(sources, start=1):
            if isinstance(src, dict):
                title = src.get("title", "Unknown")
                src_type = src.get("source_type", "")
                url = src.get("url", "")
                line = f"  [{i}] {title}"
                if src_type:
                    line += f" ({src_type})"
                if url:
                    line += f"\n      {url}"
                print(line)


def main() -> None:
    """Run the basic query examples."""
    with httpx.Client() as client:
        print("ClinEvidence Basic Query Example")
        print("=" * 60)

        if not check_health(client):
            print(
                "ERROR: Server is not healthy. "
                "Is ClinEvidence running?"
            )
            sys.exit(1)

        questions = [
            (
                "What is the recommended fluid resuscitation "
                "strategy in septic shock according to the "
                "Surviving Sepsis Campaign guidelines?"
            ),
            (
                "What are the diagnostic criteria for acute "
                "respiratory distress syndrome (ARDS)?"
            ),
            (
                "Explain the difference between SIRS and sepsis "
                "under the Sepsis-3 definitions."
            ),
        ]

        for question in questions:
            try:
                result = submit_query(
                    client, question, SESSION_ID
                )
                display_response(question, result)
            except httpx.HTTPStatusError as exc:
                print(
                    f"\nHTTP error {exc.response.status_code}: "
                    f"{exc.response.text}"
                )
            except httpx.ConnectError:
                print(
                    "\nConnection failed. "
                    "Ensure the server is running at "
                    f"{BASE_URL}"
                )
                sys.exit(1)

        print("\n" + "=" * 60)
        print("Example complete.")


if __name__ == "__main__":
    main()
