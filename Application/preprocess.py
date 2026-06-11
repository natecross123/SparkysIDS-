import numpy as np
from typing import Optional, Tuple, Dict
from Rule import FlowStats
from schemas import FlowMetadata


class FeatureExtractor:
    """Extract features matching your XGBoost training data exactly."""

    # Protocol encoding (must match training)
    PROTO_ENCODING = {
        "TCP": 6,
        "UDP": 17,
        "ICMP": 1,
        "DNS": 17,  # DNS is UDP, so use UDP code
        "OTHER": 0,
    }

    DIR_ENCODING = {
        "forward": 0,
        "backward": 1,
    }
    
    # TCP state encoding (simplified)
    STATE_ENCODING = {
        "-": 0,
        "SYN_SENT": 1,
        "SYN_RECV": 2,
        "ESTABLISHED": 3,
        "FIN_WAIT": 4,
        "CLOSED": 5,
    }

    @staticmethod
    def extract_from_flow_stats(
        flow_stats: FlowStats,
        flow_metadata: FlowMetadata,
    ) -> Tuple[Optional[np.ndarray], list]:
        """
        Extract features in EXACT order for XGBoost model.
        
        Returns:
            (feature_array, warnings) where feature_array is shape (9,)
            matching your training features exactly
        """
        warnings = []

        try:
            # Get basic values
            dur = flow_stats.duration
            proto = flow_metadata.protocol
            stos = float(flow_stats.src_tos)
            dtos = float(flow_stats.dst_tos)
            tot_pkts = float(flow_stats.packet_count)
            tot_bytes = float(flow_stats.byte_total)
            src_bytes = float(flow_stats.src_bytes)
            tcp_state = flow_stats.tcp_state
            
            # Encode categorical variables
            proto_encoded = FeatureExtractor.PROTO_ENCODING.get(proto, 0)
            state_encoded = FeatureExtractor.STATE_ENCODING.get(tcp_state, 0)
            dir_encoded = FeatureExtractor.DIR_ENCODING["forward"]  # We always capture forward direction
            
            # Build feature array in EXACT order from training
            features = np.array([
                dur,             # 0. dur
                proto_encoded,   # 1. proto (encoded)
                dir_encoded,     # 2. dir (encoded)
                state_encoded,   # 3. state (encoded)
                stos,            # 4. stos
                dtos,            # 5. dtos
                tot_pkts,        # 6. tot_pkts
                tot_bytes,       # 7. tot_bytes
                src_bytes,       # 8. src_bytes
            ], dtype=np.float32)
            
            return features, warnings

        except Exception as e:
            return None, [f"Feature extraction error: {str(e)}"]

    @staticmethod
    def validate_features(features: np.ndarray) -> Tuple[bool, list]:
        """
        Validate extracted features.
        """
        errors = []
        
        if features.shape != (9,):
            errors.append(f"Wrong shape: {features.shape}, expected (9,)")
            return False, errors
        
        # Check for NaN/Inf
        for i, val in enumerate(features):
            if np.isnan(val) or np.isinf(val):
                errors.append(f"Feature {i} is NaN or Inf: {val}")
        
        # Check reasonable ranges
        checks = [
            (0, "dur", 1e-6, 3600),           # Duration: 1µs to 1 hour
            (1, "proto", 0, 17),              # Protocol code
            (2, "dir", 0, 1),                 # Direction: 0 or 1
            (3, "state", 0, 5),               # State: 0-5
            (4, "stos", 0, 255),              # ToS: 0-255
            (5, "dtos", 0, 255),              # ToS: 0-255
            (6, "tot_pkts", 1, 100000),       # Packets
            (7, "tot_bytes", 1, 10e9),        # Bytes
            (8, "src_bytes", 0, 10e9),        # Source bytes
        ]
        
        for idx, name, min_val, max_val in checks:
            val = features[idx]
            if not (min_val <= val <= max_val):
                errors.append(f"{name} out of range: {val} (expected {min_val}-{max_val})")
        
        return len(errors) == 0, errors


class CTU13FeatureMapper:
    """Map your flow data to CTU-13 feature format."""
    
    # Feature names in order (for reference)
    FEATURE_NAMES = [
        'dur',      # 0
        'proto',    # 1
        'dir',      # 2
        'state',    # 3
        'stos',     # 4
        'dtos',     # 5
        'tot_pkts', # 6
        'tot_bytes',# 7
        'src_bytes',# 8
    ]
    
    @staticmethod
    def features_to_model_input(features: np.ndarray) -> np.ndarray:
        """
        Convert feature array to model input format.
        Your model was trained with these exact features, so no transformation needed.
        """
        return features.astype(np.float32)
    
    @staticmethod
    def features_to_dict(features: np.ndarray) -> Dict[str, float]:
        """Convert feature array to named dict for debugging."""
        return {
            name: float(val) 
            for name, val in zip(CTU13FeatureMapper.FEATURE_NAMES, features)
        }
    
    @staticmethod
    def validate_feature_count(model_num_features: int) -> bool:
        """Verify your model expects exactly 9 features."""
        if model_num_features != 9:
            print(f"⚠️  WARNING: Model expects {model_num_features} features, but we extract 9")
            print(f"   Your training features: {CTU13FeatureMapper.FEATURE_NAMES}")
            return False
        print(f"✓ Model expects 9 features - GOOD!")
        return True