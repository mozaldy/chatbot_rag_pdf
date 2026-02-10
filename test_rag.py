#!/usr/bin/env python3
"""
Quick test script to verify RAG improvements
Run after re-ingesting your PDF
"""

import requests

API_BASE = "http://localhost:8000/api"

def test_query(question: str):
    """Test a query and display results"""
    print(f"\n{'='*80}")
    print(f"QUESTION: {question}")
    print(f"{'='*80}\n")
    
    response = requests.post(
        f"{API_BASE}/chat",
        json={"messages": question}
    )
    
    if response.status_code == 200:
        data = response.json()
        print("ANSWER:")
        print(data["response"])
        print(f"\n{'-'*80}")
        print("SOURCES:")
        for source in data["sources"]:
            print(
                f"  • {source['id']} | {source['filename']} | "
                f"chunk={source['chunk_index']} | score={source['score']}"
            )
    else:
        print(f"ERROR: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    # Test cases
    test_cases = [
        "Can the traffic analysis application read vehicle plate numbers?",
        "What are the main features of the application?",
        "What object detection model is used?",
        "Does the system support tracking vehicles?",
    ]
    
    for question in test_cases:
        test_query(question)
        print("\n")
