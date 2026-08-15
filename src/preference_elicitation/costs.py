from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from threading import Lock


Money = Decimal | int | float | str
PER_MILLION = Decimal("1000000")
DEFAULT_PROTOCOL_OVERHEAD_TOKENS = 512


def _decimal(value: Money, name: str) -> Decimal:
    try:
        amount = value if isinstance(value, Decimal) else Decimal(str(value))
    except Exception as exc:
        raise ValueError(f"{name} must be a decimal number") from exc
    if not amount.is_finite() or amount < 0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return amount


def _token_count(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    cached_input_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0

    def __post_init__(self) -> None:
        for name in (
            "input_tokens",
            "cached_input_tokens",
            "cache_write_tokens",
            "output_tokens",
            "reasoning_tokens",
        ):
            _token_count(getattr(self, name), name)
        if self.cached_input_tokens + self.cache_write_tokens > self.input_tokens:
            raise ValueError("cached reads and cache writes exceed input tokens")
        if self.reasoning_tokens > self.output_tokens:
            raise ValueError("reasoning tokens exceed output tokens")


@dataclass(frozen=True)
class ModelRates:
    input_per_million: Decimal
    cached_input_per_million: Decimal
    output_per_million: Decimal
    cache_write_per_million: Decimal | None = None

    def __post_init__(self) -> None:
        for name in (
            "input_per_million",
            "cached_input_per_million",
            "output_per_million",
        ):
            object.__setattr__(self, name, _decimal(getattr(self, name), name))
        if self.cache_write_per_million is not None:
            object.__setattr__(
                self,
                "cache_write_per_million",
                _decimal(self.cache_write_per_million, "cache_write_per_million"),
            )

    @classmethod
    def from_mapping(cls, values: Mapping[str, Money]) -> "ModelRates":
        try:
            return cls(
                input_per_million=values["input"],
                cached_input_per_million=values["cached_input"],
                output_per_million=values["output"],
                cache_write_per_million=values.get("cache_write"),
            )
        except KeyError as exc:
            raise ValueError(f"missing model rate: {exc.args[0]}") from exc


def calculate_cost(usage: Usage, rates: ModelRates) -> Decimal:
    # The API includes reasoning tokens in output_tokens, so do not add them again.
    if rates.cache_write_per_million is None:
        ordinary_input = usage.input_tokens - usage.cached_input_tokens
        cache_write_cost = Decimal(0)
    else:
        ordinary_input = (
            usage.input_tokens - usage.cached_input_tokens - usage.cache_write_tokens
        )
        cache_write_cost = (
            Decimal(usage.cache_write_tokens) * rates.cache_write_per_million
        )

    total = (
        Decimal(ordinary_input) * rates.input_per_million
        + Decimal(usage.cached_input_tokens) * rates.cached_input_per_million
        + cache_write_cost
        + Decimal(usage.output_tokens) * rates.output_per_million
    )
    return total / PER_MILLION


def reservation_cost(
    prompt: str,
    max_output_tokens: int,
    rates: ModelRates,
    *,
    protocol_overhead_tokens: int = DEFAULT_PROTOCOL_OVERHEAD_TOKENS,
) -> Decimal:
    # UTF-8 bytes plus fixed protocol overhead is deliberately pessimistic.
    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")
    output_tokens = _token_count(max_output_tokens, "max_output_tokens")
    overhead = _token_count(protocol_overhead_tokens, "protocol_overhead_tokens")
    input_tokens = len(prompt.encode("utf-8")) + overhead
    possible_input_rates = [
        rates.input_per_million,
        rates.cached_input_per_million,
    ]
    if rates.cache_write_per_million is not None:
        possible_input_rates.append(rates.cache_write_per_million)
    total = (
        Decimal(input_tokens) * max(possible_input_rates)
        + Decimal(output_tokens) * rates.output_per_million
    )
    return total / PER_MILLION


class BudgetExceeded(RuntimeError):
    pass


class BudgetLedger:
    def __init__(self, cap: Money) -> None:
        self._cap = _decimal(cap, "cap")
        self._spent = Decimal(0)
        self._reserved = Decimal(0)
        self._reservations: dict[str, Decimal] = {}
        self._next_id = 1
        self._lock = Lock()

    @property
    def spent(self) -> Decimal:
        with self._lock:
            return self._spent

    @property
    def reserved(self) -> Decimal:
        with self._lock:
            return self._reserved

    @property
    def available(self) -> Decimal:
        with self._lock:
            return self._cap - self._spent - self._reserved

    def reserve(self, amount: Money, reservation_id: str | None = None) -> str:
        requested = _decimal(amount, "reservation")
        if requested == 0:
            raise ValueError("reservation must be greater than zero")
        with self._lock:
            if reservation_id is not None and (
                not isinstance(reservation_id, str) or not reservation_id
            ):
                raise ValueError("reservation_id must be a non-empty string")
            if reservation_id is not None and reservation_id in self._reservations:
                raise ValueError(f"duplicate reservation_id: {reservation_id}")
            committed = self._spent + self._reserved + requested
            if committed > self._cap:
                raise BudgetExceeded(f"reservation would exceed budget cap {self._cap}")
            if reservation_id is None:
                reservation_id = f"reservation-{self._next_id}"
                self._next_id += 1
            self._reservations[reservation_id] = requested
            self._reserved += requested
            return reservation_id

    def reconcile(self, reservation_id: str, actual: Money) -> None:
        actual_cost = _decimal(actual, "actual cost")
        with self._lock:
            reserved = self._reservations[reservation_id]
            committed = self._spent + self._reserved - reserved + actual_cost
            if committed > self._cap:
                raise BudgetExceeded(f"actual cost would exceed budget cap {self._cap}")
            del self._reservations[reservation_id]
            self._reserved -= reserved
            self._spent += actual_cost
