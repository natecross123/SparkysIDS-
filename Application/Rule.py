from __future__ import annotations
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from scapy.all import DNS, ICMP, IP, TCP, UDP, Packet, sniff


FlowKey = Tuple[str, str, int, int, str]


@dataclass
class FlowStats:
    """State tracked per 5-tuple flow for near-real-time decisions."""

    key: FlowKey
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    packet_count: int = 0
    byte_total: int = 0
    packet_sizes: List[int] = field(default_factory=list)
    push_count: int = 0
    timestamps: List[float] = field(default_factory=list)
    dns_count: int = 0

    def update(self, pkt: Packet, ts: Optional[float] = None) -> None:
        now = ts if ts is not None else float(getattr(pkt, "time", time.time()))
        self.last_seen = now
        if self.packet_count == 0:
            self.first_seen = now

        size = len(pkt)
        self.packet_count += 1
        self.byte_total += size
        self.packet_sizes.append(size)
        self.timestamps.append(now)

        if TCP in pkt and pkt[TCP].flags & 0x08:
            self.push_count += 1

        if DNS in pkt:
            self.dns_count += 1

    @property
    def duration(self) -> float:
        return max(self.last_seen - self.first_seen, 1e-6)

    @property
    def avg_bytes(self) -> float:
        if self.packet_count == 0:
            return 0.0
        return self.byte_total / self.packet_count

    @property
    def var_bytes(self) -> float:
        if self.packet_count == 0:
            return 0.0
        mu = self.avg_bytes
        return sum((s - mu) ** 2 for s in self.packet_sizes) / self.packet_count

    @property
    def std_bytes(self) -> float:
        return math.sqrt(self.var_bytes)

    @property
    def push_pct(self) -> float:
        if self.packet_count == 0:
            return 0.0
        return (self.push_count / self.packet_count) * 100.0

    @property
    def packets_per_second(self) -> float:
        return self.packet_count / self.duration

    @property
    def bytes_per_second(self) -> float:
        return self.byte_total / self.duration

    @property
    def iat_mean(self) -> float:
        if len(self.timestamps) < 2:
            return 0.0
        iats = [self.timestamps[i] - self.timestamps[i - 1] for i in range(1, len(self.timestamps))]
        return sum(iats) / len(iats)

    @property
    def iat_std(self) -> float:
        if len(self.timestamps) < 3:
            return 0.0
        iats = [self.timestamps[i] - self.timestamps[i - 1] for i in range(1, len(self.timestamps))]
        mu = sum(iats) / len(iats)
        return math.sqrt(sum((x - mu) ** 2 for x in iats) / len(iats))


class BotnetRuleEngine:
    """Hybrid-friendly rules engine for flow-level botnet detection."""

    def __init__(self, window_seconds: int = 30, min_packets: int = 3):
        self.window_seconds = window_seconds
        self.min_packets = min_packets
        self.flows: Dict[FlowKey, FlowStats] = {}

        # Keep these strongly suspicious only if corroborated by behavior.
        self.suspicious_ports = {
            2077,
            3389,
            6667,
            5765,
            11161,
            11251,
            11256,
            11301,
            11374,
            11735,
            11776,
            11821,
            11866,
            11952,
            12074,
            12164,
            12301,
            12389,
            12724,
            12767,
            12834,
            12879,
            12970,
            13089,
        }

        self.ambiguous_ports = {53, 80, 443}

    @staticmethod
    def packet_protocol(pkt: Packet) -> str:
        if DNS in pkt:
            return "DNS"
        if TCP in pkt:
            return "TCP"
        if UDP in pkt:
            return "UDP"
        if ICMP in pkt:
            return "ICMP"
        return "OTHER"

    @staticmethod
    def flow_key(pkt: Packet) -> Optional[FlowKey]:
        if IP not in pkt:
            return None

        ip_layer = pkt[IP]
        src = ip_layer.src
        dst = ip_layer.dst
        proto = BotnetRuleEngine.packet_protocol(pkt)

        sport = -1
        dport = -1
        if TCP in pkt:
            sport = int(pkt[TCP].sport)
            dport = int(pkt[TCP].dport)
        elif UDP in pkt:
            sport = int(pkt[UDP].sport)
            dport = int(pkt[UDP].dport)

        return (src, dst, sport, dport, proto)

    def update_packet(self, pkt: Packet) -> Optional[Tuple[FlowKey, Dict[str, float]]]:
        key = self.flow_key(pkt)
        if key is None:
            return None

        stats = self.flows.get(key)
        if stats is None:
            stats = FlowStats(key=key)
            self.flows[key] = stats

        stats.update(pkt)

        # Emit only when enough packets observed to reduce noise.
        if stats.packet_count >= self.min_packets:
            return key, self.flow_check(stats)
        return None

    def flush_expired(self, now: Optional[float] = None) -> List[Tuple[FlowKey, Dict[str, float]]]:
        now = time.time() if now is None else now
        results: List[Tuple[FlowKey, Dict[str, float]]] = []
        expired = [k for k, s in self.flows.items() if now - s.last_seen >= self.window_seconds]

        for k in expired:
            stats = self.flows.pop(k)
            if stats.packet_count >= self.min_packets:
                results.append((k, self.flow_check(stats)))

        return results

    def flow_check(self, flow: FlowStats) -> Dict[str, float]:
        """Return rule decision plus transparent score/reasons for fusion with ML output."""
        src, dst, sport, dport, proto = flow.key

        score = 0
        reasons: List[str] = []

        has_suspicious_port = sport in self.suspicious_ports or dport in self.suspicious_ports
        has_ambiguous_port = sport in self.ambiguous_ports or dport in self.ambiguous_ports

        if has_suspicious_port:
            score += 4
            reasons.append("suspicious_port")

        if proto in {"TCP", "UDP", "DNS"} and flow.avg_bytes < 100:
            score += 2
            reasons.append("small_avg_pkt")

        if flow.var_bytes < 300:
            score += 2
            reasons.append("low_pkt_size_variance")

        if flow.push_count == 0 and proto == "TCP":
            score += 1
            reasons.append("no_tcp_push")

        if flow.std_bytes < 20:
            score += 1
            reasons.append("low_std_pkt_size")

        if 10000 <= sport <= 20000 or 10000 <= dport <= 20000:
            score += 1
            reasons.append("high_ephemeral_port_band")

        if flow.packet_count < 10 and flow.var_bytes < 200:
            score += 1
            reasons.append("short_low_variance_flow")

        # Beacon-like periodicity: low inter-arrival jitter with repeated packets.
        if flow.packet_count >= 5 and flow.iat_mean > 0 and flow.iat_std < max(flow.iat_mean * 0.2, 0.02):
            score += 3
            reasons.append("beacon_like_periodicity")

        if proto == "DNS" and flow.packet_count >= 8 and flow.packets_per_second > 5:
            score += 2
            reasons.append("high_rate_dns_flow")

        # Ambiguous ports (53/80/443) require stronger corroboration.
        threshold = 8 if has_ambiguous_port and not has_suspicious_port else 6
        is_botnet = 1 if score >= threshold else 0

        return {
            "is_botnet": is_botnet,
            "rule_score": float(score),
            "threshold": float(threshold),
            "packet_count": float(flow.packet_count),
            "byte_total": float(flow.byte_total),
            "avg_bytes": round(flow.avg_bytes, 3),
            "var_bytes": round(flow.var_bytes, 3),
            "std_bytes": round(flow.std_bytes, 3),
            "push_count": float(flow.push_count),
            "push_pct": round(flow.push_pct, 3),
            "iat_mean": round(flow.iat_mean, 5),
            "iat_std": round(flow.iat_std, 5),
            "pps": round(flow.packets_per_second, 3),
            "bps": round(flow.bytes_per_second, 3),
            "sport": float(sport),
            "dport": float(dport),
            "proto": 1.0 if proto == "TCP" else 2.0 if proto == "UDP" else 3.0 if proto == "DNS" else 0.0,
            "reason_count": float(len(reasons)),
        }


def run_live_capture(interface: str = "eth0", timeout: int = 30) -> List[Tuple[FlowKey, Dict[str, float]]]:
    """Capture live packets and return scored flow decisions.

    Suitable for virtual-lab testing (Kali/Windows victim traffic) and local experiments.
    """
    engine = BotnetRuleEngine(window_seconds=30, min_packets=3)
    decisions: List[Tuple[FlowKey, Dict[str, float]]] = []

    def on_packet(pkt: Packet) -> None:
        result = engine.update_packet(pkt)
        if result is not None:
            decisions.append(result)

    sniff(prn=on_packet, store=False, iface=interface, timeout=timeout)
    decisions.extend(engine.flush_expired())
    return decisions

