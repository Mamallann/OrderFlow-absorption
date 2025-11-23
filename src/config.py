"""Configuration management for the backtest system."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any
from decimal import Decimal
import yaml


@dataclass
class DataConfig:
    """Data fetching configuration."""
    symbols: List[str] = field(default_factory=lambda: ["BTCUSDT"])
    timeframe: str = "10m"
    tick_size: float = 100.0
    data_dir: str = "data"


@dataclass
class DateRangeConfig:
    """Date range configuration."""
    start: str = "2024-01-01"
    end: str = "2024-12-31"
    lookback_days: Optional[int] = None


@dataclass
class BinanceConfig:
    """Binance API configuration."""
    api_key: str = ""
    api_secret: str = ""
    testnet: bool = False
    rate_limit_delay: float = 0.1


@dataclass
class ImbalanceConfig:
    """Imbalance detection configuration."""
    ratio_threshold: float = 3.0
    min_volume: float = 1.0
    consecutive_rows: int = 3
    diagonal_offset: int = 1


@dataclass
class AbsorptionConfig:
    """Absorption detection configuration."""
    delta_lookback: int = 2
    delta_comparison: str = "both"  # "both", "any", "average"
    delta_multiplier: float = 1.0
    price_hold_mode: str = "no_break"  # "no_break", "wick_rejection", "both"
    wick_rejection_ratio: float = 0.3
    allow_equal_lows: bool = True
    min_imbalance_rows: int = 3
    min_imbalance_ratio: float = 3.0


@dataclass
class ShiftConfig:
    """Shift detection configuration."""
    require_opposite_imbalance: bool = True
    min_opposite_imbalances: int = 1
    require_delta_flip: bool = False
    require_buy_aggression: bool = True
    max_candles_after_absorption: int = 3


@dataclass
class EntryConfig:
    """Entry rules configuration."""
    mode: str = "shift_close"  # "shift_close", "breakout", "next_candle_open"
    breakout_buffer: float = 0.0001
    require_volume_confirmation: bool = False


@dataclass
class StopLossConfig:
    """Stop loss configuration."""
    mode: str = "absorption_low"  # "absorption_low", "shift_low", "atr", "fixed"
    buffer: float = 0.001
    atr_multiplier: float = 1.5
    fixed_percent: float = 0.02
    max_sl_percent: float = 0.05


@dataclass
class TrailingConfig:
    """Trailing stop configuration."""
    activation_rr: float = 1.0
    trail_percent: float = 0.005
    trail_atr_multiplier: float = 1.0
    use_atr: bool = False


@dataclass
class RRTarget:
    """Single R:R target."""
    ratio: float
    size: float  # Percentage of position to close


@dataclass
class TakeProfitConfig:
    """Take profit configuration."""
    mode: str = "rr_then_trail"  # "rr", "rr_then_trail", "trail_only", "targets"
    rr_ratio: float = 2.0
    rr_targets: List[RRTarget] = field(default_factory=lambda: [
        RRTarget(ratio=1.0, size=0.5),
        RRTarget(ratio=2.0, size=0.3),
        RRTarget(ratio=3.0, size=0.2),
    ])
    trailing: TrailingConfig = field(default_factory=TrailingConfig)


@dataclass
class BacktestConfig:
    """Backtest execution configuration."""
    initial_capital: float = 10000.0
    position_size_mode: str = "risk"  # "risk", "fixed", "percent"
    risk_per_trade: float = 0.02
    fixed_size: float = 1000.0
    position_percent: float = 0.1
    commission: float = 0.0004
    slippage: float = 0.0001
    max_trades_per_day: int = 10
    max_concurrent_trades: int = 3
    min_trade_interval: int = 3
    min_volatility: float = 0.0
    max_volatility: float = 1.0


@dataclass
class ReportingConfig:
    """Reporting configuration."""
    output_dir: str = "data/results"
    save_trades: bool = True
    save_equity_curve: bool = True
    generate_html: bool = True
    generate_charts: bool = True


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    file: str = "logs/backtest.log"
    console: bool = True


@dataclass
class Config:
    """Main configuration container."""
    data: DataConfig = field(default_factory=DataConfig)
    date_range: DateRangeConfig = field(default_factory=DateRangeConfig)
    binance: BinanceConfig = field(default_factory=BinanceConfig)
    imbalance: ImbalanceConfig = field(default_factory=ImbalanceConfig)
    absorption: AbsorptionConfig = field(default_factory=AbsorptionConfig)
    shift: ShiftConfig = field(default_factory=ShiftConfig)
    entry: EntryConfig = field(default_factory=EntryConfig)
    stop_loss: StopLossConfig = field(default_factory=StopLossConfig)
    take_profit: TakeProfitConfig = field(default_factory=TakeProfitConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load configuration from YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Create Config from dictionary."""
        config = cls()

        if "data" in data:
            config.data = DataConfig(**data["data"])

        if "date_range" in data:
            config.date_range = DateRangeConfig(**data["date_range"])

        if "binance" in data:
            config.binance = BinanceConfig(**data["binance"])

        if "imbalance" in data:
            config.imbalance = ImbalanceConfig(**data["imbalance"])

        if "absorption" in data:
            config.absorption = AbsorptionConfig(**data["absorption"])

        if "shift" in data:
            config.shift = ShiftConfig(**data["shift"])

        if "entry" in data:
            config.entry = EntryConfig(**data["entry"])

        if "stop_loss" in data:
            config.stop_loss = StopLossConfig(**data["stop_loss"])

        if "take_profit" in data:
            tp_data = data["take_profit"]
            trailing_data = tp_data.pop("trailing", {})
            rr_targets_data = tp_data.pop("rr_targets", [])

            config.take_profit = TakeProfitConfig(**tp_data)
            config.take_profit.trailing = TrailingConfig(**trailing_data)
            config.take_profit.rr_targets = [
                RRTarget(**t) for t in rr_targets_data
            ]

        if "backtest" in data:
            config.backtest = BacktestConfig(**data["backtest"])

        if "reporting" in data:
            config.reporting = ReportingConfig(**data["reporting"])

        if "logging" in data:
            config.logging = LoggingConfig(**data["logging"])

        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        from dataclasses import asdict
        return asdict(self)


def load_config(path: str | Path = "config.yaml") -> Config:
    """Load configuration from file with defaults."""
    path = Path(path)
    if path.exists():
        return Config.from_yaml(path)
    return Config()
