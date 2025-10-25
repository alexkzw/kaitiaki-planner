"""
Claude API Client with Cost Tracking and Safety Features
========================================================

Usage:
    from claude_client import ClaudeClient
    
    client = ClaudeClient(api_key="sk-ant-...", max_spend_usd=5.0)
    response = client.generate_answer(query="...", passages=[...])
    print(f"Cost: ${client.total_cost_usd:.4f}")
"""

import os
from anthropic import Anthropic
from typing import List, Dict, Optional
import json

class ClaudeClient:
    """
    Safe Claude API client with automatic cost tracking and spending limits.
    """
    
    # Pricing as of Oct 2025 (USD per million tokens)
    PRICING = {
        "claude-sonnet-4-5": {
            "input": 3.0,   # $3/MTok
            "output": 15.0  # $15/MTok
        }
    }
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-5-20250929",
        max_spend_usd: float = 5.0
    ):
        """
        Initialize Claude client with safety limits.
        
        Args:
            api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
            model: Claude model to use
            max_spend_usd: Maximum spend before refusing requests (default: $10)
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "No API key provided. Set ANTHROPIC_API_KEY env var or pass api_key parameter.\n"
                "Get your key from: https://console.anthropic.com/settings/keys"
            )
        
        self.client = Anthropic(api_key=self.api_key)
        self.model = model
        self.max_spend_usd = max_spend_usd
        self.total_cost_usd = 0.0
        self.request_count = 0
        self.token_stats = {"input": 0, "output": 0}
    
    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost in USD for given token counts."""
        pricing = self.PRICING["claude-sonnet-4-5"]
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost
    
    def _check_budget(self):
        """Raise error if budget exceeded."""
        if self.total_cost_usd >= self.max_spend_usd:
            raise RuntimeError(
                f"Budget exceeded! Spent ${self.total_cost_usd:.4f} / ${self.max_spend_usd:.2f}. "
                f"Increase max_spend_usd if needed."
            )
    
    def generate_answer(
        self,
        query: str,
        passages: List[Dict],
        max_tokens: int = 300,
        temperature: float = 0.0
    ) -> Dict:
        """
        Generate grounded answer from retrieved passages.
        
        Args:
            query: User question
            passages: List of dicts with keys: doc_id, text, char_start, char_end
            max_tokens: Max output tokens
            temperature: 0.0 = deterministic, 1.0 = creative
        
        Returns:
            {
                "answer": str,
                "citations": [{"doc_id": str, "char_start": int, "char_end": int}],
                "refusal": bool,
                "usage": {"input_tokens": int, "output_tokens": int},
                "cost_usd": float
            }
        """
        self.check_budget()
        
        # Format passages for prompt
        passage_text = "\n\n".join([
            f"[Passage {i+1} from {p['doc_id']}]\n{p['text']}"
            for i, p in enumerate(passages)
        ])
        
        # A permissive system prompt that doesn't discourage Māori answers
        system_prompt = """You are a helpful assistant that answers questions using the provided passages.

You work with content in both English and Te Reo Māori (Māori language). Treat both languages equally - answer questions in Māori using Māori passages, and questions in English using English passages.

RULES:
1. Base your answer on information in the provided passages
2. Cite your sources by referencing passage numbers: [Passage 1], [Passage 2], etc.
3. If the passages contain relevant information, provide your best answer even if some details are unclear
4. Only refuse if the passages are completely unrelated to the question
5. Keep answers concise (2-3 sentences)
6. For Māori queries, answer in Māori if the passages are in Māori

Your goal is to be helpful and accurate, using the information provided."""

        user_prompt = f"""Question: {query}

Available passages:
{passage_text}

Please answer the question using the information in these passages. Cite which passage(s) you used."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )
            
            # Extract usage and calculate cost
            usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens
            }
            cost = self._calculate_cost(usage["input_tokens"], usage["output_tokens"])
            
            # Update tracking
            self.total_cost_usd += cost
            self.request_count += 1
            self.token_stats["input"] += usage["input_tokens"]
            self.token_stats["output"] += usage["output_tokens"]
            
            # Extract answer text
            answer_text = response.content[0].text
            
            # Simple citation extraction (map passage references to actual doc_ids)
            citations = self._extract_citations(answer_text, passages)
            
            # Check for refusal patterns (less strict now)
            refusal = self._is_refusal(answer_text)
            
            return {
                "answer": answer_text,
                "citations": citations,
                "refusal": refusal,
                "usage": usage,
                "cost_usd": cost
            }
            
        except Exception as e:
            # Log error but don't crash
            return {
                "answer": "",
                "citations": [],
                "refusal": True,
                "usage": {"input_tokens": 0, "output_tokens": 0},
                "cost_usd": 0.0,
                "error": str(e)
            }
    
    def _extract_citations(self, answer: str, passages: List[Dict]) -> List[Dict]:
        """
        Extract citations from answer text.
        Maps [Passage N] references to actual doc_ids and char offsets.
        """
        citations = []
        for i, passage in enumerate(passages):
            # Look for [Passage N] or "Passage N" references
            if f"[Passage {i+1}]" in answer or f"Passage {i+1}" in answer:
                citations.append({
                    "doc_id": passage["doc_id"],
                    "char_start": passage.get("char_start", 0),
                    "char_end": passage.get("char_end", len(passage.get("text", "")))
                })
        return citations
    
    def _is_refusal(self, answer: str) -> bool:
        """
        Check if answer is a refusal/insufficient info response.
        
        FIXED: Less strict - only catches explicit refusals, not cautious statements.
        """
        # Only catch very clear refusals
        refusal_phrases = [
            "cannot answer this question",
            "unable to answer",
            "passages are completely unrelated",
            "passages do not contain",
            "passages don't contain any information"
        ]
        answer_lower = answer.lower()
        return any(phrase in answer_lower for phrase in refusal_phrases)
    
    def get_stats(self) -> Dict:
        """Get cumulative usage statistics."""
        return {
            "requests": self.request_count,
            "total_cost_usd": self.total_cost_usd,
            "total_tokens": sum(self.token_stats.values()),
            "input_tokens": self.token_stats["input"],
            "output_tokens": self.token_stats["output"],
            "avg_cost_per_request": self.total_cost_usd / max(self.request_count, 1),
            "budget_remaining_usd": self.max_spend_usd - self.total_cost_usd
        }
    
    def print_stats(self):
        """Pretty-print usage statistics."""
        stats = self.get_stats()
        print("\n" + "="*50)
        print("CLAUDE API USAGE STATISTICS")
        print("="*50)
        print(f"Requests:           {stats['requests']}")
        print(f"Total Cost:         ${stats['total_cost_usd']:.4f} USD")
        print(f"Avg Cost/Request:   ${stats['avg_cost_per_request']:.4f} USD")
        print(f"Budget Remaining:   ${stats['budget_remaining_usd']:.2f} USD")
        print(f"Input Tokens:       {stats['input_tokens']:,}")
        print(f"Output Tokens:      {stats['output_tokens']:,}")
        print(f"Total Tokens:       {stats['total_tokens']:,}")
        print("="*50 + "\n")


# Example usage / test
if __name__ == "__main__":
    # Test with dummy data
    client = ClaudeClient(max_spend_usd=1.0)  # $1 limit for testing
    
    test_passages = [
        {
            "doc_id": "en_kea",
            "text": "The kea is a species of large parrot endemic to New Zealand's South Island.",
            "char_start": 0,
            "char_end": 100
        }
    ]
    
    response = client.generate_answer(
        query="What is a kea?",
        passages=test_passages
    )
    
    print("Answer:", response["answer"])
    print("Citations:", response["citations"])
    print("Cost:", f"${response['cost_usd']:.6f}")
    
    client.print_stats()
