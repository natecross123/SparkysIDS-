"""
decision.py - Decision Fusion Engine

Combines rule-based detection with ML predictions to create a final
consensus decision with confidence scores.

Strategies:
1. Weighted Average: (rules_score * w_rules) + (ml_score * w_ml)
2. Consensus Voting: Both agree = higher confidence
3. Threshold-based: Different thresholds for rules-only, ML-only, both
"""

import math
from typing import Tuple, Optional
from schemas import (
    RuleDecision, MLDecision, FinalDecision, FusionWeights,
    FlowMetadata, AlertLevel
)


class DecisionFusionEngine:
    """Fuse rules and ML predictions into a single decision."""

    def __init__(self, weights: FusionWeights = None):
        """
        Initialize fusion engine.
        
        Args:
            weights: FusionWeights for combining rules and ML
        """
        self.weights = weights or FusionWeights()

    def fuse(
        self,
        rule_decision: RuleDecision,
        ml_decision: MLDecision,
        flow_metadata: FlowMetadata,
    ) -> FinalDecision:
        """
        Fuse rule and ML decisions into a single decision.
        
        Args:
            rule_decision: Output from BotnetRuleEngine
            ml_decision: Output from XGBoost model
            flow_metadata: Flow identification and timing info
            
        Returns:
            FinalDecision with alert level and confidence
        """

        # Normalize scores to 0-1 range
        rule_score_norm = self._normalize_rule_score(rule_decision)
        ml_score_norm = ml_decision.probability  # Already 0-1 from model

        # Apply weighting strategy
        fused_score = (
            self.weights.rule_weight * rule_score_norm +
            self.weights.ml_weight * ml_score_norm
        )

        # Check for strong signals from either component alone
        is_botnet_rule_only = rule_decision.is_botnet == 1
        is_botnet_ml_only = ml_decision.prediction == 1

        # Consensus boost: if both agree, increase confidence
        if is_botnet_rule_only and is_botnet_ml_only:
            fused_score = min(fused_score * self.weights.consensus_boost, 1.0)

        # Determine alert level
        alert_level = self._get_alert_level(fused_score)

        # Identify primary and secondary indicators
        primary_indicator, secondary_indicators = self._analyze_indicators(
            rule_decision, ml_decision, rule_score_norm, ml_score_norm
        )

        # Final decision
        is_botnet = fused_score >= self.weights.consensus_threshold

        # Recommendation
        should_block = alert_level in [AlertLevel.HIGH, AlertLevel.CRITICAL]

        decision = FinalDecision(
            alert_level=alert_level,
            is_botnet=is_botnet,
            fused_score=fused_score,
            rule_score_normalized=rule_score_norm,
            ml_score_normalized=ml_score_norm,
            rule_decision=rule_decision,
            ml_decision=ml_decision,
            flow_metadata=flow_metadata,
            primary_indicator=primary_indicator,
            secondary_indicators=secondary_indicators,
            should_block=should_block,
            investigation_notes=self._generate_investigation_notes(
                rule_decision, ml_decision, fused_score
            ),
        )

        return decision

    def _normalize_rule_score(self, rule_decision: RuleDecision) -> float:
        """
        Normalize rule score (0-15) to 0-1 range.
        
        Uses confidence property which is already rule_score/threshold.
        Clamp to 0-1 for consistency with ML probabilities.
        """
        confidence = rule_decision.confidence
        return min(confidence, 1.0)

    def _get_alert_level(self, fused_score: float) -> AlertLevel:
        """
        Map fused score to alert level.
        """
        if fused_score >= 0.90:
            return AlertLevel.CRITICAL
        elif fused_score >= 0.80:
            return AlertLevel.HIGH
        elif fused_score >= 0.65:
            return AlertLevel.MEDIUM
        elif fused_score >= 0.50:
            return AlertLevel.LOW
        else:
            return AlertLevel.NORMAL

    def _analyze_indicators(
        self,
        rule_decision: RuleDecision,
        ml_decision: MLDecision,
        rule_score_norm: float,
        ml_score_norm: float,
    ) -> Tuple[str, list]:
        """
        Analyze which components triggered alerts and why.
        
        Returns:
            (primary_indicator, [secondary_indicators])
        """
        indicators = []

        # Rule-based indicators
        if rule_decision.triggered_reasons:
            indicators.extend(
                [f"RULE:{r}" for r in rule_decision.triggered_reasons]
            )

        # ML-based indicators
        if ml_decision.prediction == 1:
            indicators.append(f"ML:probability={ml_decision.probability:.2f}")

        # Determine primary
        if rule_score_norm > ml_score_norm:
            primary = "Rules-driven"
        elif ml_score_norm > rule_score_norm:
            primary = "ML-driven"
        elif rule_score_norm > 0.5 and ml_score_norm > 0.5:
            primary = "Consensus"
        else:
            primary = "Low confidence"

        secondary = indicators[1:] if len(indicators) > 1 else []

        return primary, secondary

    def _generate_investigation_notes(
        self,
        rule_decision: RuleDecision,
        ml_decision: MLDecision,
        fused_score: float,
    ) -> str:
        """
        Generate human-readable investigation notes.
        """
        notes = []

        # Rule insights
        if rule_decision.rule_score > 0:
            notes.append(
                f"Rule engine detected {int(rule_decision.reason_count)} suspicious indicators "
                f"(score {rule_decision.rule_score:.1f}/{rule_decision.threshold:.0f})"
            )
            if "suspicious_port" in rule_decision.triggered_reasons:
                notes.append(f"  → Suspicious port usage: "
                            f"{int(rule_decision.sport)}→{int(rule_decision.dport)}")
            if "beacon_like_periodicity" in rule_decision.triggered_reasons:
                notes.append(f"  → Beacon-like pattern detected "
                            f"(IAT mean: {rule_decision.iat_mean:.3f}s, "
                            f"std: {rule_decision.iat_std:.5f}s)")
            if "high_rate_dns_flow" in rule_decision.triggered_reasons:
                notes.append(f"  → High-rate DNS query: "
                            f"{rule_decision.pps:.1f} packets/sec")

        # ML insights
        if ml_decision.probability > 0.5:
            notes.append(
                f"ML model assigns {ml_decision.probability:.2%} probability to botnet behavior"
            )
            if ml_decision.feature_importance:
                top_features = sorted(
                    ml_decision.feature_importance.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:3]
                if top_features:
                    notes.append(f"  → Most important features: "
                               f"{', '.join(f[0] for f in top_features)}")

        # Overall assessment
        if fused_score >= 0.80:
            notes.append("⚠️ RECOMMENDATION: Investigate immediately, consider blocking")
        elif fused_score >= 0.65:
            notes.append("⚠️ RECOMMENDATION: Monitor flow, review logs for related activity")
        elif fused_score >= 0.50:
            notes.append("ℹ️ RECOMMENDATION: Log for reference, low immediate threat")

        return "\n".join(notes) if notes else "No specific indicators detected"

    @staticmethod
    def agreement_score(rule_decision: RuleDecision, ml_decision: MLDecision) -> float:
        """
        Measure how much rule and ML predictions agree (0.0-1.0).
        
        High agreement means both are confident, increasing reliability.
        """
        rule_pred = rule_decision.is_botnet  # 0 or 1
        ml_pred = ml_decision.prediction  # 0 or 1

        # If predictions match, measure confidence agreement
        if rule_pred == ml_pred:
            rule_conf = rule_decision.confidence
            ml_conf = ml_decision.confidence

            # How similar are confidences?
            conf_distance = abs(rule_conf - ml_conf)
            agreement = 1.0 - conf_distance
            return max(agreement, 0.0)
        else:
            # Predictions disagree - measure partial agreement
            # If one is very confident and other is not, allow partial credit
            rule_conf = rule_decision.confidence
            ml_conf = ml_decision.confidence

            # Example: rule=0.9 (confident normal), ml=0.6 (moderate botnet)
            # Partial agreement = how close the lower one is to threshold
            min_conf = min(rule_conf, ml_conf)
            max_conf = max(rule_conf, ml_conf)

            # If one is weak (< 0.6) and other is moderate, partial agreement
            if min_conf < 0.6 and max_conf < 0.8:
                return min_conf
            else:
                return 0.0  # Strong disagreement


class ThresholdTuner:
    """Helper to optimize fusion thresholds based on test data."""

    @staticmethod
    def calculate_f1(tp: int, fp: int, fn: int) -> float:
        """Calculate F1 score from confusion matrix."""
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        return f1

    @staticmethod
    def suggest_threshold(
        fused_scores: list,
        true_labels: list,
        target_metric: str = "f1",
    ) -> Tuple[float, float]:
        """
        Suggest optimal threshold using grid search.
        
        Args:
            fused_scores: List of fusion engine scores (0-1)
            true_labels: List of ground truth labels (0 or 1)
            target_metric: "f1", "precision", or "recall"
            
        Returns:
            (optimal_threshold, score_value)
        """
        best_threshold = 0.5
        best_score = 0.0

        for threshold in [i * 0.01 for i in range(1, 100)]:
            predictions = [1 if s >= threshold else 0 for s in fused_scores]

            tp = sum(1 for p, t in zip(predictions, true_labels) if p == 1 and t == 1)
            fp = sum(1 for p, t in zip(predictions, true_labels) if p == 1 and t == 0)
            fn = sum(1 for p, t in zip(predictions, true_labels) if p == 0 and t == 1)
            tn = sum(1 for p, t in zip(predictions, true_labels) if p == 0 and t == 0)

            if target_metric == "f1":
                score = ThresholdTuner.calculate_f1(tp, fp, fn)
            elif target_metric == "precision":
                score = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            elif target_metric == "recall":
                score = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            if score > best_score:
                best_score = score
                best_threshold = threshold

        return best_threshold, best_score