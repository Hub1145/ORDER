# detection/system.py
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import logging
from scipy import stats
from sklearn.preprocessing import StandardScaler
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
import ruptures as rpt
from hmmlearn import hmm
import stumpy
from adtk.detector import LevelShiftAD
from adtk.data import validate_series
import os
import json
from datetime import datetime
from collections import defaultdict, deque # Added import


logger = logging.getLogger(__name__)


class DetectionSystem:
    def __init__(self, pipeline_config: Dict[str, Any]): # Accepts the full pipeline config
        self.pipeline_config = pipeline_config
        self.config = pipeline_config.get('detection', {}) # Extract detection-specific config
        self.detectors = {}
        # self.scaler = StandardScaler() # Not directly used in detect_anomalies, can be removed if not used elsewhere
        self._initialize_detectors()
        
    def _initialize_detectors(self):
        self.detectors['pyod_iforest'] = PyODIForest(self.config.get('iforest', {}))
        self.detectors['order_skew'] = OrderSkewDetector(self.config.get('order_skew', {}))
        self.detectors['changepoint'] = ChangePointDetector(self.config.get('changepoint', {}))
        self.detectors['hmm_detect'] = HMMDetect(self.config.get('hmm', {}))
        self.detectors['matrix_profile'] = MatrixProfileDetector(self.config.get('matrix_profile', {}))
        self.detectors['level_shift'] = LevelShiftDetector(self.config.get('level_shift', {}))
        
    async def detect_anomalies(self, features: Dict[str, Any], feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run all detectors and return detailed outputs."""
        results = {}
        
        feature_array = self._prepare_features(features)
        
        # --- run each detector
        for name, detector in self.detectors.items():
            try:
                # Assuming each detector's detect method returns a dict with 'score', 'is_anomaly', etc.
                # Or adapt the detector classes to return this structure.
                res = detector.detect(feature_array, features) # Pass raw features for some detectors
                
                # Standardize structure for output
                results[name] = {
                    "score_raw": float(res.get("score", np.nan)),
                    "is_anomaly": bool(res.get("is_anomaly", False)),
                    "threshold": res.get("threshold", getattr(detector, 'threshold', np.nan)),
                    "p_value": res.get("p_value", None), # Use None instead of np.nan for JSON serializability
                    "method_details": res.get("details", {}),
                    "feature_subset": res.get("features_used", []),
                    "model_params": res.get("model_params", getattr(detector, "get_params", lambda: {})()) # Prefer from res, fallback to get_params
                }
            except Exception as e:
                logger.error(f"Detector {name} failed: {e}", exc_info=True)
                results[name] = {"error": str(e), "timestamp": datetime.utcnow().isoformat()} # Add timestamp to error for debugging

        # --- composite
        valid_scores = [v["score_raw"] for v in results.values() if isinstance(v.get("score_raw"), (int, float)) and not np.isnan(v["score_raw"])]
        composite_score = np.mean(valid_scores) if valid_scores else np.nan
        severity = self._classify(composite_score)

        # --- assemble output
        output = {
            "timestamp": datetime.utcnow().isoformat(),
            "composite": {"weighted_score": composite_score, "severity": severity, "num_detectors_flagged": sum(1 for v in results.values() if v.get("is_anomaly"))},
            "detectors": results
        }

        # --- if full mode, enrich with stats
        if self.pipeline_config.get("output", {}).get("mode") == "full":
            output["meta_statistics"] = self._compute_meta_statistics(feature_array, feature_names)

        # --- optional: save to disk
        if self.pipeline_config.get("output", {}).get("save_detector_details"):
            self._write_output(output)

        return output
        
    def _prepare_features(self, features: Dict[str, Any]) -> np.ndarray:
        numeric_features = []
        self.current_feature_names = [] # Store feature names for meta_statistics
        
        for key, value in features.items():
            if isinstance(value, (int, float)) and not np.isnan(value) and not np.isinf(value):
                numeric_features.append(value)
                self.current_feature_names.append(key)
                
        return np.array(numeric_features).reshape(1, -1)
        
    def _compute_meta_statistics(self, X: np.ndarray, feature_names: Optional[List[str]]) -> Dict[str, Any]:
        """Compute simple summaries for mathematician analysis."""
        if X.shape[1] == 0:
            return {}
            
        df = pd.DataFrame(X, columns=feature_names or [f"f{i}" for i in range(X.shape[1])])
        
        meta_stats = {}
        if not df.empty:
            meta_stats["mean"] = df.mean(axis=0).to_dict()
            meta_stats["variance"] = df.var(axis=0).to_dict()
            if len(df.columns) > 1: # Correlation requires more than one column
                meta_stats["correlation"] = df.corr().to_dict()
                
        return meta_stats
        

    def _classify(self, score: float) -> str:
        if np.isnan(score):
            return "unknown"
        if score >= 0.8:
            return 'critical'
        elif score >= 0.6:
            return 'high'
        elif score >= 0.4:
            return 'medium'
        elif score >= 0.2:
            return 'low'
        else:
            return 'normal'
            


class PyODIForest:
    def __init__(self, config: Dict[str, Any]):
        self.threshold = config.get('threshold', 0.5)
        self.contamination = config.get('contamination', 0.05)
        self.model = IForest(contamination=self.contamination, random_state=42)
        self.is_fitted = False
        
    def detect(self, features: np.ndarray, raw_features: Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_fitted:
            # Fit on a larger dataset if available, or just the current one if not
            self.model.fit(features)
            self.is_fitted = True
            
        decision_score = self.model.decision_function(features)[0]
        
        # Check for NaN or inf before np.exp
        if not np.isfinite(decision_score):
            normalized_score = 0.0 # Default to normal if score is not finite
            is_anomaly = False
            logger.warning(f"PyODIForest: decision_score is not finite ({decision_score}). Setting normalized_score to {normalized_score}.")
        else:
            normalized_score = 1 / (1 + np.exp(-decision_score)) # Normalize to 0-1
            is_anomaly = bool(normalized_score > self.threshold)
        
        return {
            "score": normalized_score,
            "is_anomaly": is_anomaly,
            "threshold": self.threshold,
            "details": {"decision_function_output": float(decision_score)},
            "features_used": list(raw_features.keys()), # All features for now
            "model_params": self.get_params()
        }
        
    def get_params(self) -> Dict[str, Any]:
        return {
            "contamination": self.contamination,
            "n_estimators": self.model.n_estimators,
            "max_features": self.model.max_features,
            "max_samples": self.model.max_samples
        }


class OrderSkewDetector:
    def __init__(self, config: Dict[str, Any]):
        self.threshold = config.get('threshold', 0.7)
        self.skew_threshold = config.get('skew_threshold', 0.6)
        self.volume_threshold = config.get('volume_threshold', 2.0)
        
    def detect(self, features: np.ndarray, raw_features: Dict[str, Any]) -> Dict[str, Any]:
        score = 0
        details = {}
        features_used = []
        
        imbalance_features_list = [(k, v) for k, v in raw_features.items() if 'imbalance' in k and isinstance(v, (int, float))]
        if imbalance_features_list:
            max_imbalance = max(abs(i[1]) for i in imbalance_features_list)
            score = max_imbalance
            details['max_imbalance'] = float(max_imbalance)
            features_used.extend([k for k, v in imbalance_features_list])
            
        volume_ratio_features_list = [(k, v) for k, v in raw_features.items() if 'bid_ask_ratio' in k and isinstance(v, (int, float))]
        if volume_ratio_features_list:
            # Avoid division by zero if all ratios are 0 or negative
            valid_ratios = [i[1] for i in volume_ratio_features_list if i[1] > 0]
            extreme_ratio = 1.0
            if valid_ratios:
                extreme_ratio = max(max(valid_ratios), 1/min(valid_ratios) if min(valid_ratios) > 0 else 1)
            
            if extreme_ratio > self.volume_threshold:
                score = max(score, (extreme_ratio - 1) / (self.volume_threshold * 2))
                details['extreme_volume_ratio'] = float(extreme_ratio)
                features_used.extend([k for k, v in volume_ratio_features_list])
                
        final_score = min(score, 1.0)
        return {
            "score": final_score,
            "is_anomaly": bool(final_score > self.threshold),
            "threshold": self.threshold,
            "details": details,
            "features_used": list(set(features_used)),
            "model_params": self.get_params()
        }
        
    def get_params(self) -> Dict[str, Any]:
        return {
            "skew_threshold": self.skew_threshold,
            "volume_threshold": self.volume_threshold
        }


class ChangePointDetector:
    def __init__(self, config: Dict[str, Any]):
        self.threshold = config.get('threshold', 0.6)
        self.model_type = config.get('model', 'rbf')
        self.penalty = config.get('penalty', 3)
        self.min_size = config.get('min_size', 5)
        
    def detect(self, features: np.ndarray, raw_features: Dict[str, Any]) -> Dict[str, Any]:
        if features.shape[1] < self.min_size:
            return {
                "score": 0.0,
                "is_anomaly": False,
                "threshold": self.threshold,
                "details": {"reason": "Not enough features for changepoint detection"},
                "features_used": list(raw_features.keys()),
                "model_params": self.get_params()
            }
            
        try:
            algo = rpt.Pelt(model=self.model_type, min_size=self.min_size).fit(features.flatten())
            change_points = algo.predict(pen=self.penalty)
            
            n_changepoints = len(change_points) - 1
            score = min(n_changepoints / 5, 1.0) # Scale score based on number of changepoints
            
            return {
                "score": float(score),
                "is_anomaly": bool(score > self.threshold),
                "threshold": self.threshold,
                "details": {
                    "n_changepoints": n_changepoints,
                    "change_point_locations": [int(cp) for cp in change_points]
                },
                "features_used": list(raw_features.keys()),
                "model_params": self.get_params()
            }
        except Exception as e:
            logger.warning(f"ChangePointDetector failed: {e}")
            return {
                "score": 0.0,
                "is_anomaly": False,
                "threshold": self.threshold,
                "details": {"error": str(e)},
                "features_used": list(raw_features.keys()),
                "model_params": self.get_params()
            }
            
    def get_params(self) -> Dict[str, Any]:
        return {
            "model_type": self.model_type,
            "penalty": self.penalty,
            "min_size": self.min_size
        }


class HMMDetect:
    def __init__(self, config: Dict[str, Any]):
        self.threshold = config.get('threshold', 0.65)
        self.n_components = config.get('n_components', 3)
        self.model = hmm.GaussianHMM(n_components=self.n_components, covariance_type="diag", n_iter=100, random_state=42)
        self.baseline_log_likelihood = None # Changed to store log likelihood
        
    def detect(self, features: np.ndarray, raw_features: Dict[str, Any]) -> Dict[str, Any]:
        if features.shape[1] < self.n_components: # Need at least n_components samples
            return {
                "score": 0.0,
                "is_anomaly": False,
                "threshold": self.threshold,
                "details": {"reason": "Not enough features for HMM detection"},
                "features_used": list(raw_features.keys()),
                "model_params": self.get_params()
            }
            
        try:
            # Reshape features for HMM (n_samples, n_features)
            # Assuming features are a single time step (1, n_features) for now
            # For HMM, we usually need a sequence. This might need adjustment based on how features are passed.
            # For simplicity, we'll use the single feature array as a sequence of 1 observation.
            
            # Fit/predict on a sequence of the most recent features if available
            # For a single point detection, we'd typically compare to a fitted model's likelihood
            
            # This is a placeholder for a proper HMM anomaly detection strategy
            # A common approach is to fit the HMM on "normal" data and then check the likelihood of new data.
            # If the log-likelihood is significantly lower, it's an anomaly.
            
            # For now, let's just train and get the likelihood for the current features
            self.model.fit(features.T) # Transpose to (n_samples, n_features)
            log_likelihood = self.model.score(features.T)
            
            if self.baseline_log_likelihood is None:
                self.baseline_log_likelihood = log_likelihood
                
            # Score: deviation from baseline likelihood. Higher deviation = higher anomaly score.
            # This scoring mechanism needs careful tuning based on expected log-likelihood distribution.
            score = 1 - (log_likelihood / self.baseline_log_likelihood) if self.baseline_log_likelihood != 0 else 0.0
            score = np.clip(score, 0.0, 1.0) # Clip score to be between 0 and 1
            
            # Get state probabilities for additional details
            state_probs = self.model.predict_proba(features.T)[-1].tolist()
            predicted_state = int(self.model.predict(features.T)[-1])
            
            return {
                "score": float(score),
                "is_anomaly": bool(score > self.threshold),
                "threshold": self.threshold,
                "details": {
                    "log_likelihood": float(log_likelihood),
                    "baseline_log_likelihood": float(self.baseline_log_likelihood),
                    "state_probabilities": state_probs,
                    "predicted_state": predicted_state
                },
                "features_used": list(raw_features.keys()),
                "model_params": self.get_params()
            }
        except Exception as e:
            logger.warning(f"HMMDetect failed: {e}")
            return {
                "score": 0.0,
                "is_anomaly": False,
                "threshold": self.threshold,
                "details": {"error": str(e)},
                "features_used": list(raw_features.keys()),
                "model_params": self.get_params()
            }
            
    def get_params(self) -> Dict[str, Any]:
        return {
            "n_components": self.n_components,
            "covariance_type": self.model.covariance_type,
            "n_iter": self.model.n_iter
        }

class MatrixProfileDetector:
    def __init__(self, config: Dict[str, Any]):
        self.threshold = config.get('threshold', 0.7)
        self.window_size = config.get('window_size', 10)
        # Store historical time series data for Matrix Profile
        self.time_series_data = defaultdict(lambda: deque(maxlen=1000)) # Store up to 1000 data points
        
    def detect(self, features: np.ndarray, raw_features: Dict[str, Any]) -> Dict[str, Any]:
        # Identify relevant time series features from raw_features
        # This needs to be consistent across calls to build a proper time series
        relevant_features = {k: v for k, v in raw_features.items() if any(x in k for x in ['price', 'volume', 'vwap']) and isinstance(v, (int, float))}
        
        # For simplicity, let's assume we are tracking a single composite time series, e.g., 'mid_price' or 'composite_price'
        # In a real scenario, you might have multiple matrix profiles or a way to select the primary one.
        
        # For this example, let's just pick the first numeric feature in the raw_features for the time series
        # A more robust solution would require explicit configuration of which feature to use for MP.
        time_series_value = next(iter(relevant_features.values()), None)
        time_series_name = next(iter(relevant_features.keys()), None)
        
        if time_series_value is None:
            return {
                "score": 0.0,
                "is_anomaly": False,
                "threshold": self.threshold,
                "details": {"reason": "No relevant time series features found for Matrix Profile"},
                "features_used": [],
                "model_params": self.get_params()
            }
            
        self.time_series_data[time_series_name].append(time_series_value)
        
        ts_array = np.array(list(self.time_series_data[time_series_name])).astype(np.float64) # Ensure float64 dtype
        
        if len(ts_array) < self.window_size * 2: # Matrix profile needs at least 2 windows of data
            return {
                "score": 0.0,
                "is_anomaly": False,
                "threshold": self.threshold,
                "details": {
                    "reason": "Not enough historical data for Matrix Profile",
                    "current_history_length": len(ts_array),
                    "required_history_length": self.window_size * 2,
                    "history_buffer_preview": ts_array.tolist() # Include the buffer data
                },
                "features_used": [time_series_name],
                "model_params": self.get_params()
            }
            
        try:
            # Compute Matrix Profile
            mp = stumpy.stump(ts_array, m=self.window_size)
            
            # Discord detection: highest value in the matrix profile indicates the most anomalous subsequence
            discord_idx = np.argmax(mp[:, 0])
            discord_value = mp[discord_idx, 0]
            
            # Normalize discord value to a 0-1 score
            # A simple normalization, could be more sophisticated
            median_mp_value = np.median(mp[:, 0])
            score = 1 - np.exp(-discord_value / (median_mp_value + 1e-10)) # Add epsilon to prevent division by zero
            score = np.clip(score, 0.0, 1.0)
            
            return {
                "score": float(score),
                "is_anomaly": bool(score > self.threshold),
                "threshold": self.threshold,
                "details": {
                    "discord_value": float(discord_value),
                    "discord_index": int(discord_idx),
                    "window_size": self.window_size
                },
                "features_used": [time_series_name],
                "model_params": self.get_params()
            }
        except Exception as e:
            logger.warning(f"MatrixProfileDetector failed: {e}")
            return {
                "score": 0.0,
                "is_anomaly": False,
                "threshold": self.threshold,
                "details": {"error": str(e)},
                "features_used": [time_series_name],
                "model_params": self.get_params()
            }
            
    def get_params(self) -> Dict[str, Any]:
        return {
            "window_size": self.window_size,
            "max_history_length": self.time_series_data[next(iter(self.time_series_data))].maxlen if self.time_series_data else 1000
        }
            


class LevelShiftDetector:
    def __init__(self, config: Dict[str, Any]):
        self.threshold = config.get('threshold', 0.6)
        self.c = config.get('c', 3.0)
        self.window = config.get('window', 5)
        self.price_history = deque(maxlen=1000) # Store historical price data
        
    def detect(self, features: np.ndarray, raw_features: Dict[str, Any]) -> Dict[str, Any]:
        # Extract a single price-like feature for detection
        price_feature_name = next((k for k in raw_features.keys() if 'price' in k.lower() and isinstance(raw_features[k], (int, float))), None)
        
        if price_feature_name is None:
            return {
                "score": 0.0,
                "is_anomaly": False,
                "threshold": self.threshold,
                "details": {"reason": "No price-like features found for LevelShiftDetector"},
                "features_used": [],
                "model_params": self.get_params()
            }
            
        self.price_history.append(raw_features[price_feature_name])
        
        if len(self.price_history) < self.window * 2:
            return {
                "score": 0.0,
                "is_anomaly": False,
                "threshold": self.threshold,
                "details": {
                    "reason": "Not enough historical data for LevelShiftDetector",
                    "current_history_length": len(self.price_history),
                    "required_history_length": self.window * 2,
                    "history_buffer_preview": list(self.price_history) # Include the buffer data
                },
                "features_used": [price_feature_name],
                "model_params": self.get_params()
            }
            
        try:
            # ADTK requires a pandas Series with a DatetimeIndex
            ts = pd.Series(list(self.price_history), index=pd.date_range(end=datetime.utcnow(), periods=len(self.price_history), freq='min'))
            validated = validate_series(ts)
            
            detector = LevelShiftAD(c=self.c, side='both', window=self.window)
            anomalies = detector.fit_detect(validated) # fit_detect for online detection
            
            # Score can be based on the magnitude of the shift or simply if an anomaly is detected
            is_anomaly_detected = anomalies.iloc[-1] # Check the latest point
            
            # For a score, we can use the anomaly magnitude if ADTK provides it, or a binary score
            score = 1.0 if is_anomaly_detected else 0.0
            
            return {
                "score": float(score),
                "is_anomaly": bool(is_anomaly_detected),
                "threshold": self.threshold,
                "details": {
                    "shift_detected": bool(is_anomaly_detected),
                    # "shift_magnitude": detector.model.shift_magnitude if hasattr(detector.model, 'shift_magnitude') else None # ADTK might not expose this easily
                },
                "features_used": [price_feature_name],
                "model_params": self.get_params()
            }
        except Exception as e:
            logger.error(f"LevelShiftDetector failed: {e}", exc_info=True)
            return {
                "score": 0.0,
                "is_anomaly": False,
                "threshold": self.threshold,
                "details": {"error": str(e)},
                "features_used": [price_feature_name],
                "model_params": self.get_params()
            }
            
    def get_params(self) -> Dict[str, Any]:
        return {
            "c": self.c,
            "window": self.window
        }