"""Small API client CLI: exactly the same authorization and services as the browser."""

import argparse
import getpass
from pathlib import Path

import httpx


def run():
    parser = argparse.ArgumentParser(description="Document Assistant CLI")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--username", default="local")
    parser.add_argument("--upload", type=Path)
    parser.add_argument("--session")
    args = parser.parse_args()
    with httpx.Client(base_url=args.url, headers={"X-RAG-Request": "1"}, timeout=240) as client:
        key = getpass.getpass("Access key (empty for local development): ")
        response = client.post(
            "/api/v1/auth/login", json={"username": args.username, "access_key": key}
        )
        response.raise_for_status()
        if args.upload:
            with args.upload.open("rb") as source:
                response = client.post(
                    "/api/v1/documents", files={"file": (args.upload.name, source)}
                )
            response.raise_for_status()
            print(response.json())
            return
        session = args.session
        if not session:
            response = client.post("/api/v1/sessions", json={"title": "CLI conversation"})
            response.raise_for_status()
            session = response.json()["id"]
        print(f"Conversation: {session}. Type /exit to leave.")
        while True:
            query = input("You: ").strip()
            if query == "/exit":
                break
            if not query:
                continue
            response = client.post("/api/v1/chat", json={"session_id": session, "query": query})
            data = response.json()
            if response.is_error:
                print(data.get("error", {}).get("message", "Request failed."))
                continue
            print(data["answer"])
            for source in data["sources"]:
                print(
                    f"[{source['citation_id']}] {source['filename']} · page {source['page'] or 'n/a'}"
                )


if __name__ == "__main__":
    run()
