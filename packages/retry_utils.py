import time
import random
from typing import Callable, Type, Tuple

def retry_with_backoff(
    func: Callable,
    exceptions: Tuple[Type[BaseException], ...],
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
):
    """
    Prosty mechanizm retry z wykładniczym backoffem + jitter.
    - func: funkcja bezargumentowa lub lambda
    - exceptions: krotka wyjątków do obsługi (np. (RateLimitError, TimeoutError))
    """
    for attempt in range(max_retries):
        try:
            return func()
        except exceptions as e:
            if attempt == max_retries - 1:
                raise  # po ostatniej próbie przepuszczamy wyjątek
            # oblicz opóźnienie: 2^attempt * base_delay (z jitterem)
            sleep_time = min(max_delay, base_delay * (2 ** attempt))
            sleep_time *= random.uniform(0.8, 1.2)  # jitter
            print(f"[retry] Próba {attempt+1}/{max_retries} nieudana ({e}). "
                  f"Retry za {sleep_time:.1f}s...")
            time.sleep(sleep_time)