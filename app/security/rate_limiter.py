import time
from collections import defaultdict
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class RateLimiter:
    """
    Sliding window rate limiter.
    Enforces per-minute and per-day request limits per user.
    """

    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_day: int = 1000
    ):
        self.requests_per_minute = requests_per_minute
        self.requests_per_day = requests_per_day

        self.minute_window: Dict[str, list] = defaultdict(list)
        self.day_window: Dict[str, list] = defaultdict(list)

    def is_allowed(self, user_id: str) -> Tuple[bool, str]:
        """
        Check if user is within rate limits.

        Returns:
            (allowed, reason)
        """
        now = time.time()

        # Per minute check - remove old timestamps
        self.minute_window[user_id] = [
            t for t in self.minute_window[user_id]
            if now - t < 60
        ]

        if len(self.minute_window[user_id]) >= self.requests_per_minute:
            logger.warning(f"Rate limit exceeded (minute) for: {user_id}")
            return False, f"Rate limit exceeded: {self.requests_per_minute} requests/minute"

        # Per day check - remove old timestamps
        self.day_window[user_id] = [
            t for t in self.day_window[user_id]
            if now - t < 86400
        ]

        if len(self.day_window[user_id]) >= self.requests_per_day:
            logger.warning(f"Rate limit exceeded (day) for: {user_id}")
            return False, f"Rate limit exceeded: {self.requests_per_day} requests/day"

        # Allow request
        self.minute_window[user_id].append(now)
        self.day_window[user_id].append(now)

        return True, "OK"

    def get_stats(self, user_id: str) -> dict:
        """Return current usage stats for a user"""
        now = time.time()

        minute_count = len([
            t for t in self.minute_window[user_id]
            if now - t < 60
        ])

        day_count = len([
            t for t in self.day_window[user_id]
            if now - t < 86400
        ])

        return {
            "user_id": user_id,
            "requests_last_minute": minute_count,
            "requests_today": day_count,
            "minute_limit": self.requests_per_minute,
            "day_limit": self.requests_per_day,
            "minute_remaining": self.requests_per_minute - minute_count,
            "day_remaining": self.requests_per_day - day_count,
        }


# Global rate limiter object
rate_limiter = RateLimiter(
    requests_per_minute=60,
    requests_per_day=1000
)