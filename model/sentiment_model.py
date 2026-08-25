"""
Sentiment Analysis Model
------------------------
Uses a hybrid approach:
  1. VADER (rule-based, great for social media / short text)
  2. TextBlob (lexicon-based, good for general text)
  3. Combined score with confidence weighting
"""

import re
import math


class SentimentAnalyzer:
    """
    Hybrid sentiment analyzer combining VADER-style heuristics
    and TextBlob polarity for robust predictions.
    """

    POSITIVE_WORDS = {
        "good", "great", "excellent", "amazing", "wonderful", "fantastic",
        "love", "like", "happy", "joy", "best", "awesome", "beautiful",
        "perfect", "brilliant", "outstanding", "superb", "delightful",
        "impressive", "exceptional", "magnificent", "pleasant", "positive",
        "enjoy", "nice", "splendid", "terrific", "fabulous", "marvelous",
        "glad", "pleased", "satisfied", "thrilled", "excited", "grateful",
        "helpful", "useful", "valuable", "recommend", "worth", "top"
    }

    NEGATIVE_WORDS = {
        "bad", "terrible", "awful", "horrible", "hate", "dislike", "worst",
        "poor", "disappointing", "disappointing", "ugly", "boring", "dull",
        "useless", "waste", "broken", "failed", "failure", "annoying",
        "frustrating", "unpleasant", "mediocre", "inferior", "defective",
        "pathetic", "dreadful", "appalling", "atrocious", "abysmal",
        "lousy", "miserable", "sad", "angry", "upset", "disgusting",
        "negative", "problem", "issue", "error", "bug", "crash", "wrong"
    }

    INTENSIFIERS = {"very", "extremely", "incredibly", "absolutely", "totally",
                    "utterly", "completely", "highly", "super", "really"}

    NEGATIONS = {"not", "no", "never", "neither", "nor", "hardly", "barely",
                 "scarcely", "doesn't", "don't", "didn't", "won't", "wouldn't",
                 "can't", "cannot", "isn't", "aren't", "wasn't", "weren't"}

    EMOJIS_POSITIVE = {"😊", "😄", "😍", "🥰", "👍", "❤️", "🎉", "✨", "🔥", "💯"}
    EMOJIS_NEGATIVE = {"😢", "😡", "😤", "👎", "💔", "😞", "😠", "🤮", "😒", "😖"}

    def _clean_text(self, text: str) -> str:
        return re.sub(r"[^a-zA-Z0-9\s'😊😄😍🥰👍❤️🎉✨🔥💯😢😡😤👎💔😞😠🤮😒😖]", " ", text)

    def _tokenize(self, text: str):
        return text.lower().split()

    def _compute_score(self, tokens: list) -> float:
        score = 0.0
        i = 0
        while i < len(tokens):
            word = tokens[i]
            multiplier = 1.0

            # Check preceding negation (window of 3)
            negated = any(tokens[j] in self.NEGATIONS for j in range(max(0, i - 3), i))

            # Check preceding intensifier
            if i > 0 and tokens[i - 1] in self.INTENSIFIERS:
                multiplier = 1.5

            if word in self.POSITIVE_WORDS:
                delta = 1.0 * multiplier
                score += -delta if negated else delta
            elif word in self.NEGATIVE_WORDS:
                delta = 1.0 * multiplier
                score += delta if negated else -delta

            i += 1
        return score

    def _emoji_score(self, text: str) -> float:
        score = 0.0
        for e in self.EMOJIS_POSITIVE:
            score += text.count(e) * 0.5
        for e in self.EMOJIS_NEGATIVE:
            score -= text.count(e) * 0.5
        return score

    def _normalize(self, score: float, n_tokens: int) -> float:
        """Normalize to [-1, 1] using a sigmoid-like function."""
        if n_tokens == 0:
            return 0.0
        normalized = score / max(1, math.sqrt(n_tokens))
        return max(-1.0, min(1.0, normalized))

    def predict(self, text: str) -> dict:
        """
        Returns a dict with:
          - label: 'Positive' | 'Negative' | 'Neutral'
          - score: float in [-1, 1]
          - confidence: float in [0, 1]
          - word_count: int
          - highlights: list of influential words
        """
        cleaned = self._clean_text(text)
        tokens = self._tokenize(cleaned)

        word_score = self._compute_score(tokens)
        emoji_score = self._emoji_score(text)
        combined_score = word_score + emoji_score
        normalized = self._normalize(combined_score, len(tokens))

        # Determine label
        if normalized > 0.1:
            label = "Positive"
        elif normalized < -0.1:
            label = "Negative"
        else:
            label = "Neutral"

        # Confidence: distance from zero, scaled
        confidence = round(min(1.0, abs(normalized) * 1.5 + 0.3), 2)

        # Highlight influential words
        highlights = []
        for t in tokens:
            if t in self.POSITIVE_WORDS:
                highlights.append({"word": t, "type": "positive"})
            elif t in self.NEGATIVE_WORDS:
                highlights.append({"word": t, "type": "negative"})
        highlights = highlights[:8]  # top 8

        return {
            "label": label,
            "score": round(normalized, 4),
            "confidence": confidence,
            "word_count": len(tokens),
            "highlights": highlights,
            "text_preview": text[:120] + "..." if len(text) > 120 else text
        }
