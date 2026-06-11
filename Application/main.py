

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import xgboost as xgb
import numpy as np
from scapy.all import sniff, Packet

from Rule import BotnetRuleEngine, FlowKey
from preprocess import FeatureExtractor, CTU13FeatureMapper
from decision import DecisionFusionEngine
from schemas import (
    IDSConfig, FinalDecision, RuleDecision, MLDecision,
    AlertLevel, FlowMetadata
)


class SparkysIDS:
    """Main IDS application - real-time hybrid detection."""

    def __init__(self, config: IDSConfig):
        """Initialize SparkysIDS."""
        self.config = config
        self.rule_engine: Optional[BotnetRuleEngine] = None
        self.ml_model: Optional[xgb.Booster] = None
      ##  self.fusion_engine: Optional[DecisionFusionEngine] = None
        
        # Results storage
        self.decisions: List[FinalDecision] = []
        self.alerts: List[FinalDecision] = []
        self.statistics = {
            "total_packets": 0,
            "total_flows": 0,
            "alerts": 0,
            "botnet_detected": 0,
            "start_time": None,
            "end_time": None,
        }

        self.logger = self._setup_logging()
        self.logger.info("SparkysIDS initialized (9-feature model)")

    def _setup_logging(self) -> logging.Logger:
        """Configure logging."""
        logger = logging.getLogger("SparkysIDS")
        logger.setLevel(logging.DEBUG if self.config.verbose else logging.INFO)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S"
        )
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)

        file_handler = logging.FileHandler("ids_runtime.log")
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

        return logger

    def initialize_engines(self) -> bool:
        """Initialize rule engine, ML model, and fusion engine."""
        try:
            # Rule engine
            if self.config.enable_rules:
                self.logger.info("Initializing rule engine (EXTENDED with TCP state tracking)")
                self.rule_engine = BotnetRuleEngine(
                    window_seconds=self.config.flow_window_seconds,
                    min_packets=self.config.min_packets_per_flow,
                )

            # ML model
            if self.config.enable_ml:
                self.logger.info(f"Loading XGBoost model from {self.config.model_path}")
                with open(self.config.model_path, "rb") as f:
                    self.ml_model = pickle.load(f)
                
                # Validate feature count
                num_features = self.ml_model.num_features()
                self.logger.info(f"✓ Model loaded: {num_features} features")
                
                if num_features != 9:
                    self.logger.warning(f"⚠️  Model expects {num_features} features, but we extract 9!")
                    self.logger.warning("   This may cause errors. Verify your training features match:")
                    self.logger.warning("   [dur, proto, dir, state, stos, dtos, tot_pkts, tot_bytes, src_bytes]")

            # Fusion engine
           ## self.fusion_engine = DecisionFusionEngine(weights=self.config.fusion_weights)
            self.logger.info(f"Fusion engine ready (rule: {self.config.fusion_weights.rule_weight:.0%}, "
                           f"ml: {self.config.fusion_weights.ml_weight:.0%})")

            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize engines: {e}")
            import traceback
            traceback.print_exc()
            return False

    def process_packet(self, pkt: Packet) -> None:
        """Process a single packet."""
        self.statistics["total_packets"] += 1

        if self.rule_engine is not None:
            rule_result = self.rule_engine.update_packet(pkt)
            if rule_result is not None:
                flow_key, rule_stats_dict = rule_result
                self._process_flow_decision(flow_key, rule_stats_dict)

    def _process_flow_decision(self, flow_key: FlowKey, rule_stats_dict: Dict) -> None:
        """Process a single flow decision from the rule engine."""
        self.statistics["total_flows"] += 1

        # Create rule decision object
        rule_decision = RuleDecision(
            is_botnet=int(rule_stats_dict["is_botnet"]),
            rule_score=rule_stats_dict["rule_score"],
            threshold=rule_stats_dict["threshold"],
            packet_count=rule_stats_dict["packet_count"],
            byte_total=rule_stats_dict["byte_total"],
            avg_bytes=rule_stats_dict["avg_bytes"],
            var_bytes=rule_stats_dict["var_bytes"],
            std_bytes=rule_stats_dict["std_bytes"],
            push_count=rule_stats_dict["push_count"],
            push_pct=rule_stats_dict["push_pct"],
            iat_mean=rule_stats_dict["iat_mean"],
            iat_std=rule_stats_dict["iat_std"],
            pps=rule_stats_dict["pps"],
            bps=rule_stats_dict["bps"],
            sport=rule_stats_dict["sport"],
            dport=rule_stats_dict["dport"],
            proto=rule_stats_dict["proto"],
            reason_count=rule_stats_dict["reason_count"],
        )

        # ML prediction
        ml_decision = self._get_ml_prediction(flow_key, rule_stats_dict)

        # Flow metadata
        src_ip, dst_ip, sport, dport, proto = flow_key
        flow_metadata = FlowMetadata(
            source_ip=src_ip,
            dest_ip=dst_ip,
            source_port=int(sport),
            dest_port=int(dport),
            protocol=proto,
            first_seen=self.rule_engine.flows[flow_key].first_seen,
            last_seen=self.rule_engine.flows[flow_key].last_seen,
        )

        # Fusion
        final_decision = self.fusion_engine.fuse(rule_decision, ml_decision, flow_metadata)
        self.decisions.append(final_decision)

        # Track alerts
        if final_decision.is_botnet:
            self.statistics["alerts"] += 1
            if final_decision.alert_level == AlertLevel.CRITICAL:
                self.statistics["botnet_detected"] += 1
            self.alerts.append(final_decision)
            self._log_alert(final_decision)
        else:
            self._log_normal_flow(final_decision)

    def _get_ml_prediction(self, flow_key: FlowKey, rule_stats_dict: Dict) -> MLDecision:
        """Get ML prediction for a flow."""
        if self.ml_model is None:
            return MLDecision(
                probability=0.5,
                prediction=0,
                confidence=0.0,
                features_used=0,
            )

        try:
            # Get flow stats
            flow_stats = self.rule_engine.flows[flow_key]
            
            src_ip, dst_ip, sport, dport, proto = flow_key
            flow_metadata = FlowMetadata(
                source_ip=src_ip,
                dest_ip=dst_ip,
                source_port=int(sport),
                dest_port=int(dport),
                protocol=proto,
                first_seen=flow_stats.first_seen,
                last_seen=flow_stats.last_seen,
            )

            # Extract features (CORRECTED: exactly 9 features)
            features, warnings = FeatureExtractor.extract_from_flow_stats(
                flow_stats, flow_metadata
            )

            if features is None:
                self.logger.debug(f"Feature extraction failed for {flow_key}: {warnings}")
                return MLDecision(
                    probability=0.5,
                    prediction=0,
                    confidence=0.0,
                    features_used=0,
                )

            # Validate features
            valid, errors = FeatureExtractor.validate_features(features)
            if not valid:
                self.logger.debug(f"Feature validation failed for {flow_key}: {errors}")
                return MLDecision(
                    probability=0.5,
                    prediction=0,
                    confidence=0.0,
                    features_used=9,
                )

            # XGBoost prediction
            dmatrix = xgb.DMatrix([features])
            raw_score = self.ml_model.predict(dmatrix)[0]
            
            # Convert to probability (assuming logistic)
            probability = 1.0 / (1.0 + np.exp(-raw_score))
            probability = float(np.clip(probability, 0.0, 1.0))
            
            prediction = 1 if probability >= self.config.ml_confidence_threshold else 0

            if self.config.verbose:
                self.logger.debug(
                    f"ML: {flow_key[0]}→{flow_key[1]}:{flow_key[3]} "
                    f"prob={probability:.3f} features={CTU13FeatureMapper.features_to_dict(features)}"
                )

            return MLDecision(
                probability=probability,
                prediction=prediction,
                confidence=abs(probability - 0.5) * 2,
                features_used=9,
                raw_score=float(raw_score),
            )

        except Exception as e:
            self.logger.warning(f"ML prediction error for {flow_key}: {e}")
            import traceback
            traceback.print_exc()
            return MLDecision(
                probability=0.5,
                prediction=0,
                confidence=0.0,
                features_used=0,
            )

    def _log_alert(self, decision: FinalDecision) -> None:
        """Log an alert."""
        self.logger.warning(f"🚨 {decision.summary}")
        
        with open(self.config.alert_log_path, "a") as f:
            f.write(decision.to_json() + "\n")

    def _log_normal_flow(self, decision: FinalDecision) -> None:
        """Log normal flow."""
        with open(self.config.flow_log_path, "a") as f:
            f.write(json.dumps({
                'timestamp': time.time(),
                'source_ip': decision.flow_metadata.source_ip,
                'dest_ip': decision.flow_metadata.dest_ip,
                'score': round(decision.fused_score, 4),
                'alert_level': decision.alert_level.name,
            }) + "\n")

    def run_capture(self) -> None:
        """Run live packet capture."""
        if not self.initialize_engines():
            self.logger.error("Failed to initialize engines")
            return

        self.statistics["start_time"] = time.time()

        try:
            self.logger.info(f"Starting capture on {self.config.interface} for {self.config.packet_timeout}s")

            sniff(
                prn=self.process_packet,
                store=False,
                iface=self.config.interface,
                timeout=self.config.packet_timeout,
            )

            # Flush expired flows
            if self.rule_engine:
                expired = self.rule_engine.flush_expired()
                for flow_key, stats_dict in expired:
                    self._process_flow_decision(flow_key, stats_dict)

        except PermissionError:
            self.logger.error("Packet capture requires elevated privileges (sudo)")
        except Exception as e:
            self.logger.error(f"Capture error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.statistics["end_time"] = time.time()
            self._print_summary()

    def _print_summary(self) -> None:
        """Print capture summary."""
        duration = (self.statistics["end_time"] - self.statistics["start_time"])
        
        self.logger.info("=" * 70)
        self.logger.info("CAPTURE SUMMARY")
        self.logger.info("=" * 70)
        self.logger.info(f"Duration: {duration:.1f}s")
        self.logger.info(f"Packets: {self.statistics['total_packets']}")
        self.logger.info(f"Flows: {self.statistics['total_flows']}")
        self.logger.info(f"Alerts: {self.statistics['alerts']}")
        self.logger.info(f"Botnet detections: {self.statistics['botnet_detected']}")
        if self.statistics['total_flows'] > 0:
            alert_rate = 100.0 * self.statistics['alerts'] / self.statistics['total_flows']
            self.logger.info(f"Alert rate: {alert_rate:.1f}%")
        self.logger.info("=" * 70)


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="SparkysIDS - Hybrid ML+Rules Intrusion Detection (9-feature model)"
    )
    parser.add_argument("--interface", default="eth0", help="Network interface")
    parser.add_argument("--duration", type=int, default=60, help="Capture duration (seconds)")
    parser.add_argument("--model", default="XGB_model/xgb_tuned_model.pkl", help="Model path")
    parser.add_argument("--alert-log", default="alerts.jsonl", help="Alert output")
    parser.add_argument("--flow-log", default="flows.jsonl", help="Flow output")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--no-ml", action="store_true", help="Disable ML")
    parser.add_argument("--no-rules", action="store_true", help="Disable rules")

    args = parser.parse_args()

    config = IDSConfig(
        interface=args.interface,
        packet_timeout=args.duration,
        model_path=args.model,
        alert_log_path=args.alert_log,
        flow_log_path=args.flow_log,
        verbose=args.verbose,
        enable_ml=not args.no_ml,
        enable_rules=not args.no_rules,
    )

    ids = SparkysIDS(config)
    ids.run_capture()


if __name__ == "__main__":
    main()

    