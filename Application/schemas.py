"""
schemas.py - Data models for SparkysIDS

Defines the data structures that flow through the hybrid IDS pipeline:
Rule Engine → Preprocessing → ML Model → Decision Fusion → Alert
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from enum import Enum
from datetime import datetime
import json


class AlertLevel(Enum):
    """Alert severity levels."""
    NORMAL = 0          # No threat detected
    LOW = 1            # Minor suspicious indicators
    MEDIUM = 2         # Multiple indicators, needs investigation
    HIGH = 3           # Strong indicators of malicious activity
    CRITICAL = 4       # Confirmed botnet/malware behavior


@dataclass
class FlowMetadata:
    """Basic flow identification and timing."""
    source_ip: str
    dest_ip: str
    source_port: int
    dest_port: int
    protocol: str  # "TCP", "UDP", "DNS", "ICMP", "OTHER"
    first_seen: float
    last_seen: float
    capture_duration: float = 0.0

    @property
    def duration(self) -> float:
        """Flow duration in seconds."""
        return max(self.last_seen - self.first_seen, 1e-6)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class RuleDecision:
    """Output from the Rule Engine (Rule.py)."""
    is_botnet: int  # 1 or 0
    rule_score: float  # 0-15 (higher = more suspicious)
    threshold: float  # Decision threshold used (typically 6 or 8)
    
    # Raw flow statistics
    packet_count: float
    byte_total: float
    avg_bytes: float
    var_bytes: float
    std_bytes: float
    push_count: float
    push_pct: float
    iat_mean: float  # Inter-arrival time mean
    iat_std: float   # Inter-arrival time std dev
    pps: float       # Packets per second
    bps: float       # Bytes per second
    
    # Protocol info
    sport: float
    dport: float
    proto: float  # 1.0=TCP, 2.0=UDP, 3.0=DNS, 0.0=OTHER
    
    reason_count: float  # Number of rules triggered
    triggered_reasons: List[str] = field(default_factory=list)

    @property
    def confidence(self) -> float:
        """Rule confidence: ratio of score to threshold."""
        if self.threshold == 0:
            return 0.0
        return min(self.rule_score / self.threshold, 1.0)

    def to_dict(self) -> Dict:
        d = asdict(self)
        d['confidence'] = self.confidence
        return d


@dataclass
class MLDecision:
    """Output from the XGBoost Model."""
    probability: float  # 0.0-1.0 (higher = more likely botnet)
    prediction: int     # 1 (botnet) or 0 (normal)
    confidence: float   # Confidence in prediction
    feature_importance: Optional[Dict[str, float]] = None
    raw_score: Optional[float] = None  # Raw model output before sigmoid
    features_used: int = 0  # Number of features fed to model

    def to_dict(self) -> Dict:
        d = asdict(self)
        return d


@dataclass
class FusionWeights:
    """Weighting strategy for combining rule and ML decisions."""
    rule_weight: float = 0.40      # 40% from rules
    ml_weight: float = 0.60         # 60% from ML model
    
    # Thresholds for different fusion strategies
    rule_only_threshold: float = 0.75  # If rule score alone suggests botnet
    ml_only_threshold: float = 0.80    # If ML alone suggests botnet
    consensus_threshold: float = 0.70  # Final fused score threshold
    
    # Boosting: if both agree, increase confidence
    consensus_boost: float = 1.2

    def __post_init__(self):
        """Ensure weights sum to 1.0."""
        total = self.rule_weight + self.ml_weight
        if abs(total - 1.0) > 0.01:
            self.rule_weight /= total
            self.ml_weight /= total


@dataclass
class FinalDecision:
    """Final detection decision combining rules + ML."""
    alert_level: AlertLevel
    is_botnet: bool
    fused_score: float  # 0.0-1.0 (final confidence)
    
    # Component scores
    rule_score_normalized: float
    ml_score_normalized: float
    
    # Details
    rule_decision: RuleDecision
    ml_decision: MLDecision
    
    # Metadata
    flow_metadata: FlowMetadata
    decision_timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    
    # Reasoning
    primary_indicator: str = ""  # Which component triggered alert
    secondary_indicators: List[str] = field(default_factory=list)
    
    # For incident response
    should_block: bool = False  # Actionable recommendation
    investigation_notes: str = ""

    @property
    def summary(self) -> str:
        """Human-readable summary of decision."""
        if not self.is_botnet:
            return f"NORMAL: {self.flow_metadata.source_ip} (score: {self.fused_score:.2f})"
        
        return (
            f"ALERT [{self.alert_level.name}]: "
            f"{self.flow_metadata.source_ip}→"
            f"{self.flow_metadata.dest_ip}:{self.flow_metadata.dest_port} "
            f"(score: {self.fused_score:.2f}, rule: {self.rule_score_normalized:.2f}, "
            f"ml: {self.ml_score_normalized:.2f})"
        )

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON logging."""
        return {
            'alert_level': self.alert_level.name,
            'is_botnet': self.is_botnet,
            'fused_score': round(self.fused_score, 4),
            'rule_score': round(self.rule_score_normalized, 4),
            'ml_score': round(self.ml_score_normalized, 4),
            'primary_indicator': self.primary_indicator,
            'secondary_indicators': self.secondary_indicators,
            'should_block': self.should_block,
            'flow': self.flow_metadata.to_dict(),
            'timestamp': datetime.fromtimestamp(self.decision_timestamp).isoformat(),
            'rule_details': self.rule_decision.to_dict(),
            'ml_details': self.ml_decision.to_dict(),
        }

    def to_json(self) -> str:
        """JSON serialization for logging."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class MLFeatures:
    """Preprocessed features for XGBoost model.
    
    NOTE: Feature order and names MUST match your training data (CTU-13).
    This is a template - customize based on your actual training features.
    """
    # Flow duration and packet timing
    duration: float
    first_packet_time: float
    last_packet_time: float
    
    # Packet count and sizes
    packet_count: int
    byte_total: int
    avg_bytes: float
    min_bytes: float
    max_bytes: float
    var_bytes: float
    std_bytes: float
    
    # Rate-based features
    packets_per_second: float
    bytes_per_second: float
    
    # Inter-arrival time (IAT) statistics
    iat_mean: float
    iat_min: float
    iat_max: float
    iat_std: float
    
    # TCP-specific
    push_count: int
    push_pct: float
    syn_count: int = 0
    ack_count: int = 0
    fin_count: int = 0
    rst_count: int = 0
    
    # Port information
    source_port: int
    dest_port: int
    is_ephemeral: bool = False
    
    # Protocol
    protocol: str  # "TCP", "UDP", "DNS", etc.
    
    # DNS specific
    dns_count: int = 0
    
    # Additional statistical features
    skewness_pkt_size: float = 0.0
    kurtosis_pkt_size: float = 0.0
    entropy_pkt_size: float = 0.0

    def to_array(self) -> List[float]:
        """Convert to feature array for model input.
        
        IMPORTANT: Order MUST match model.pkl training order.
        Customize this method to match your exact feature order.
        """
        # Template order - modify based on your CTU-13 training
        features = [
            self.duration,
            self.packet_count,
            self.byte_total,
            self.avg_bytes,
            self.var_bytes,
            self.std_bytes,
            self.packets_per_second,
            self.bytes_per_second,
            self.iat_mean,
            self.iat_std,
            float(self.push_count),
            self.push_pct,
            float(self.source_port),
            float(self.dest_port),
            float(self.is_ephemeral),
            float(self.protocol == "TCP"),
            float(self.protocol == "UDP"),
            float(self.protocol == "DNS"),
            float(self.dns_count),
            self.skewness_pkt_size,
            self.kurtosis_pkt_size,
            self.entropy_pkt_size,
        ]
        return features

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class IDSConfig:
    """Configuration for the entire IDS system."""
    # Capture settings
    interface: str = "eth0"
    packet_timeout: int = 30  # seconds
    capture_snaplen: int = 65535  # max packet size
    
    # Flow settings
    flow_window_seconds: int = 30
    min_packets_per_flow: int = 3
    
    # Rule engine
    enable_rules: bool = True
    rule_suspicious_ports: set = field(default_factory=lambda: {
        2077, 3389, 6667, 5765,  # Common botnet ports
        11161, 11251, 11256, 11301, 11374, 11735, 11776,
        11821, 11866, 11952, 12074, 12164, 12301, 12389,
        12724, 12767, 12834, 12879, 12970, 13089,
    })
    
    # ML model
    enable_ml: bool = True
    model_path: str = "Application/model.pkl"
    ml_confidence_threshold: float = 0.70
    
    # Fusion
    fusion_weights: FusionWeights = field(default_factory=FusionWeights)
    
    # Output
    alert_log_path: str = "alerts.jsonl"
    flow_log_path: str = "flows.jsonl"
    verbose: bool = True
    
    # Alert thresholds
    alert_thresholds: Dict[AlertLevel, float] = field(
        default_factory=lambda: {
            AlertLevel.LOW: 0.50,
            AlertLevel.MEDIUM: 0.65,
            AlertLevel.HIGH: 0.80,
            AlertLevel.CRITICAL: 0.90,
        }
    )

    