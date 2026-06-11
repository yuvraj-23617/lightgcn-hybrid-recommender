"""
llm_agent.py
============
LLM-powered recommendation agent backed by Groq (Llama-3.1-70B).

Capabilities
------------
  - Parse natural language user queries
  - Call the LightGCN recommender
  - Return friendly, conversational responses
  - Answer follow-up questions about specific movies
"""

import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

GROQ_MODEL  = "llama-3.3-70b-versatile"
MAX_TOKENS  = 1024
TEMPERATURE = 0.7


SYSTEM_PROMPT = """You are a movie recommendation assistant for "The 38th Suggestion Project", powered by LightGCN — a Graph Neural Network trained on the MovieLens-1M dataset.

CRITICAL CONSTRAINT: You must ONLY recommend movies that exist in the MovieLens-1M dataset.
This dataset contains approximately 3,900 movies released before 2001 (Hollywood films, popular titles).
Do NOT recommend anime, recent films (post-2001), or any title not plausibly in MovieLens-1M.

When a user asks for recommendations:
1. If they mention a user ID, present the LightGCN recommendations provided in the context.
2. If they describe tastes (e.g. "I love sci-fi action"), suggest genre-matched MovieLens titles only.
3. Present titles with genre and a brief reason — keep it conversational and concise.
4. If the user asks about a movie not in MovieLens-1M, politely note that your dataset covers films up to 2000.
5. Never invent ratings or claim data you do not have.

Examples of valid titles: Toy Story (1995), The Matrix (1999), Schindler's List (1993), Pulp Fiction (1994).
Examples of INVALID suggestions: One Punch Man, Demon Slayer, any 2002+ release.

When you receive JSON recommendation results, present them in natural language without mentioning technical terms like user_idx, score, or item_idx.
"""


class LLMAgent:
    def __init__(self, recommender=None):
        """
        Parameters
        ----------
        recommender : RecommenderAPI instance (optional)
                      If None, operates in pure LLM mode.
        """
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found. Set it in the .env file.")
        self.client      = Groq(api_key=api_key)
        self.recommender = recommender
        self.history: list[dict] = []

    def reset(self):
        """Clear conversation history."""
        self.history = []

    def _build_messages(self, user_message: str) -> list[dict]:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": user_message})
        return messages

    def _extract_user_id(self, text: str) -> int | None:
        """Try to extract a user ID from the message."""
        import re
        patterns = [
            r"user\s*(?:id)?\s*[:#]?\s*(\d+)",
            r"i'?m\s+(?:user\s+)?(\d+)",
            r"for\s+user\s+(\d+)",
            r"#(\d+)",
        ]
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                return int(m.group(1))
        return None

    def _enrich_message(self, user_message: str) -> str:
        """
        If the user is asking for recommendations, fetch them and
        inject the results into the message for the LLM to format.
        """
        if self.recommender is None:
            return user_message

        rec_keywords = ["recommend", "suggest", "watch", "movies for me", "what should i", "top movies"]
        is_rec_request = any(kw in user_message.lower() for kw in rec_keywords)

        if not is_rec_request:
            return user_message

        user_id = self._extract_user_id(user_message)
        if user_id is not None:
            try:
                recs = self.recommender.get_recommendations(user_id, k=10)
                recs_json = json.dumps(recs, indent=2)
                enriched = (
                    f"{user_message}\n\n"
                    f"[SYSTEM CONTEXT — not visible to user — Recommendations from LightGCN for user {user_id}:]\n"
                    f"{recs_json}"
                )
                return enriched
            except Exception as e:
                return f"{user_message}\n\n[SYSTEM: Could not fetch recommendations: {e}]"

        # Genre/taste-based request without user ID
        genres = self._extract_genres(user_message)
        if genres and self.recommender:
            try:
                recs = self.recommender.get_content_based(genres, k=10)
                recs_json = json.dumps(recs, indent=2)
                enriched = (
                    f"{user_message}\n\n"
                    f"[SYSTEM CONTEXT — Content-based recommendations for genres {genres}:]\n"
                    f"{recs_json}"
                )
                return enriched
            except Exception:
                pass

        return user_message

    def _extract_genres(self, text: str) -> list[str]:
        known_genres = [
            "action", "adventure", "animation", "children", "comedy", "crime",
            "documentary", "drama", "fantasy", "film-noir", "horror", "musical",
            "mystery", "romance", "sci-fi", "thriller", "war", "western",
        ]
        found = [g for g in known_genres if g in text.lower()]
        return found

    def chat(self, user_message: str) -> str:
        """
        Send a message and get a response.
        Maintains conversation history.
        """
        enriched = self._enrich_message(user_message)
        messages = self._build_messages(enriched)

        response = self.client.chat.completions.create(
            model       = GROQ_MODEL,
            messages    = messages,
            max_tokens  = MAX_TOKENS,
            temperature = TEMPERATURE,
        )

        assistant_reply = response.choices[0].message.content.strip()

        # Store in history (store original user message, not enriched)
        self.history.append({"role": "user",      "content": user_message})
        self.history.append({"role": "assistant", "content": assistant_reply})

        # Keep history bounded (last 10 exchanges = 20 messages)
        if len(self.history) > 20:
            self.history = self.history[-20:]

        return assistant_reply
