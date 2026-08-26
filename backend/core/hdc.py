import numpy as np
from typing import Dict, List, Tuple

class HDCEngine:
    def __init__(self, dimensions: int = 10000):
        self.D = dimensions
        # Seed for reproducible random vectors
        self.rng = np.random.default_rng(42)
        # A FIXED tie-break pattern for bundling. Ties (a dimension summing to exactly
        # zero) have to resolve to something, and drawing fresh randomness each time
        # made `encode` non-deterministic: the same clinical state produced a different
        # hypervector on every call, so the associative memory returned different
        # precedents for an unchanged patient. It bit hardest where it matters least
        # to be noisy — ties only occur when an EVEN number of features contribute, i.e.
        # on beds with MISSING data, which on a 30-bed cohort was 16 of 30 beds and
        # ~12% of each of their vectors.
        #
        # Drawn from its own generator so the main `rng` stream is untouched and every
        # previously generated id/level vector is bit-identical to before.
        self._tie_break = np.random.default_rng(4242).choice(
            [-1, 1], size=self.D).astype(np.int8)
        
        # Item memory for feature IDs
        self.feature_ids = {
            "heart_rate": self.random_vector(),
            "hrv": self.random_vector(),
            "spo2": self.random_vector(),
            "skin_temp": self.random_vector(),
            "motion": self.random_vector(),
            "sweat_lactate": self.random_vector(),
            "systolic_bp": self.random_vector(),
            "diastolic_bp": self.random_vector(),
            "map": self.random_vector(),
            "respiratory_rate": self.random_vector()
        }
        self.position_ids = [self.random_vector() for _ in range(16)]
        
        # Create continuous level vectors for value encoding
        self.num_levels = 100
        self.level_vectors = self._generate_level_vectors(self.num_levels)
        
        # Define ranges for normalization
        self.ranges = {
            "heart_rate": (40.0, 200.0),
            "hrv": (10.0, 150.0),
            "spo2": (70.0, 100.0),
            "skin_temp": (34.0, 42.0),
            "motion": (0.0, 100.0),
            "sweat_lactate": (0.0, 10.0),
            "systolic_bp": (70.0, 180.0),
            "diastolic_bp": (40.0, 120.0),
            "map": (45.0, 130.0),
            "respiratory_rate": (8.0, 36.0)
        }

        self.healthy_reference_states = [
            {"heart_rate": 70.0, "hrv": 80.0, "spo2": 99.0, "skin_temp": 36.6, "motion": 15.0, "sweat_lactate": 1.0, "systolic_bp": 120.0, "diastolic_bp": 80.0, "map": 93.3, "respiratory_rate": 14.0},
            {"heart_rate": 65.0, "hrv": 90.0, "spo2": 98.0, "skin_temp": 36.5, "motion": 10.0, "sweat_lactate": 0.8, "systolic_bp": 118.0, "diastolic_bp": 76.0, "map": 90.7, "respiratory_rate": 13.0},
            {"heart_rate": 75.0, "hrv": 70.0, "spo2": 99.5, "skin_temp": 36.8, "motion": 20.0, "sweat_lactate": 1.2, "systolic_bp": 124.0, "diastolic_bp": 82.0, "map": 94.7, "respiratory_rate": 15.0}
        ]
        self.sepsis_reference_states = [
            {"heart_rate": 115.0, "hrv": 15.0, "spo2": 91.0, "skin_temp": 39.1, "motion": 5.0, "sweat_lactate": 4.5, "systolic_bp": 88.0, "diastolic_bp": 54.0, "map": 66.7, "respiratory_rate": 24.0},
            {"heart_rate": 125.0, "hrv": 12.0, "spo2": 88.0, "skin_temp": 35.8, "motion": 2.0, "sweat_lactate": 5.2, "systolic_bp": 78.0, "diastolic_bp": 46.0, "map": 55.3, "respiratory_rate": 28.0},
            {"heart_rate": 105.0, "hrv": 20.0, "spo2": 93.0, "skin_temp": 38.9, "motion": 8.0, "sweat_lactate": 3.8, "systolic_bp": 94.0, "diastolic_bp": 58.0, "map": 68.7, "respiratory_rate": 23.0}
        ]

        # Archetype state vectors (representing clean clusters of Healthy vs Sepsis states)
        self.healthy_archetype = self._create_healthy_archetype()
        self.sepsis_archetype = self._create_sepsis_archetype()
        self.healthy_temporal_archetype = self._create_temporal_archetype(self.healthy_reference_states)
        self.sepsis_temporal_archetype = self._create_temporal_archetype(self.sepsis_reference_states)

    def random_vector(self) -> np.ndarray:
        """Generates a random bipolar vector (+1 or -1) of dimension D."""
        return self.rng.choice([-1, 1], size=self.D).astype(np.int8)

    def bind(self, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        """Performs element-wise multiplication (XOR equivalent for bipolar)."""
        return (v1 * v2).astype(np.int8)

    def bundle(self, vectors: List[np.ndarray]) -> np.ndarray:
        """Performs bundling (summation and thresholding/sign function)."""
        if not vectors:
            return self.random_vector()
        stacked = np.stack(vectors)
        summed = np.sum(stacked, axis=0)
        
        # Resolve ties (0 values) to prevent losing information. Deterministically:
        # the same state must always bundle to the same hypervector, or retrieval from
        # the associative memory is not reproducible. See `_tie_break` in __init__.
        zero_indices = summed == 0
        summed[zero_indices] = self._tie_break[zero_indices]
        
        return np.sign(summed).astype(np.int8)

    def similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Computes normalized cosine similarity (dot product divided by D)."""
        return float(np.dot(v1.astype(np.int32), v2.astype(np.int32)) / self.D)

    def _generate_level_vectors(self, num_levels: int) -> np.ndarray:
        """
        Generates level vectors where level 0 and level num_levels-1 are orthogonal,
        and intermediate levels have a linear transition in similarity.
        """
        levels = np.zeros((num_levels, self.D), dtype=np.int8)
        # Create base endpoints
        v_min = self.random_vector()
        v_max = self.random_vector()
        
        # To make them orthogonal, force half of v_max to be independent of v_min
        # and morph linearly. A standard way:
        # Start with v_min, and at each level, flip D / 2 / (num_levels - 1) indices
        # to match v_max.
        levels[0] = v_min
        indices_to_flip = np.where(v_min != v_max)[0]
        self.rng.shuffle(indices_to_flip)
        
        flips_per_level = len(indices_to_flip) / (num_levels - 1)
        
        for i in range(1, num_levels):
            # Copy previous level
            levels[i] = levels[i-1].copy()
            # Determine which indices to transition
            start_idx = int((i - 1) * flips_per_level)
            end_idx = int(i * flips_per_level)
            flip_subset = indices_to_flip[start_idx:end_idx]
            levels[i, flip_subset] = v_max[flip_subset]
            
        return levels

    def encode_value(self, feature_name: str, value: float) -> np.ndarray:
        """Maps a scalar value to a continuous level hypervector."""
        if feature_name not in self.ranges:
            return self.random_vector()
            
        val_min, val_max = self.ranges[feature_name]
        # Clip value within range
        clipped_val = max(val_min, min(val_max, value))
        
        # Map to level index [0, num_levels - 1]
        fraction = (clipped_val - val_min) / (val_max - val_min)
        level_idx = int(fraction * (self.num_levels - 1))
        
        return self.level_vectors[level_idx]

    def encode_patient_state(self, telemetry: Dict[str, float]) -> np.ndarray:
        """
        Binds each physiological metric with its value vector, then
        bundles all active feature vectors into a single Patient State Hypervector.
        """
        bound_vectors = []
        for feature, val in telemetry.items():
            if feature in self.feature_ids and val is not None:
                id_vec = self.feature_ids[feature]
                val_vec = self.encode_value(feature, val)
                bound_vec = self.bind(id_vec, val_vec)
                bound_vectors.append(bound_vec)
                
        return self.bundle(bound_vectors)

    def encode_temporal_window(self, telemetry_window: List[Dict[str, float]]) -> np.ndarray:
        """
        Encodes a short ordered window of patient states. Recent states receive stable
        position bindings so trajectories can be compared in the same HDC space.
        """
        if not telemetry_window:
            return self.random_vector()

        window = telemetry_window[-len(self.position_ids):]
        encoded_steps = []
        start = len(self.position_ids) - len(window)
        for offset, telemetry in enumerate(window):
            state_vec = self.encode_patient_state(telemetry)
            position_vec = self.position_ids[start + offset]
            encoded_steps.append(self.bind(position_vec, state_vec))

        return self.bundle(encoded_steps)

    def _create_healthy_archetype(self) -> np.ndarray:
        """Generates a reference archetype vector representing healthy physiology."""
        state_vectors = [self.encode_patient_state(state) for state in self.healthy_reference_states]
        return self.bundle(state_vectors)

    def _create_sepsis_archetype(self) -> np.ndarray:
        """Generates a reference archetype vector representing septic physiology."""
        state_vectors = [self.encode_patient_state(state) for state in self.sepsis_reference_states]
        return self.bundle(state_vectors)

    def _create_temporal_archetype(self, states: List[Dict[str, float]]) -> np.ndarray:
        trajectories = []
        for state in states:
            trajectories.append(self.encode_temporal_window([state] * 8))
        return self.bundle(trajectories)

    def classify_state(self, patient_state_vec: np.ndarray) -> Tuple[float, float]:
        """
        Compares the patient state vector against healthy/sepsis reference vectors.
        Returns:
            (healthy_similarity, sepsis_similarity)
        """
        sim_healthy = self.similarity(patient_state_vec, self.healthy_archetype)
        sim_sepsis = self.similarity(patient_state_vec, self.sepsis_archetype)
        return sim_healthy, sim_sepsis

    def classify_temporal_state(self, temporal_state_vec: np.ndarray) -> Tuple[float, float]:
        sim_healthy = self.similarity(temporal_state_vec, self.healthy_temporal_archetype)
        sim_sepsis = self.similarity(temporal_state_vec, self.sepsis_temporal_archetype)
        return sim_healthy, sim_sepsis
