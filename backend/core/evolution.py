import numpy as np
from typing import Dict, List, Any, Tuple

class EvolutionaryOptimizer:
    def __init__(
        self,
        pop_size: int = 40,
        generations: int = 25,
        mutation_rate: float = 0.15,
        crossover_rate: float = 0.8
    ):
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.rng = np.random.default_rng(42)

    def _project_patient_outcome(
        self,
        protocol: Tuple[float, float, int], # (fluid_rate, vasopressor_dose, antibiotic_code)
        current_vitals: Dict[str, float],
        clinical_history: Dict[str, Any]
    ) -> float:
        """
        Runs a mathematical physiological projection model over a 6-hour window.
        Returns a penalty/fitness score (higher is better, max 0.0, negative represents degradation).
        """
        fluid_rate, vaso_dose, ab_code = protocol
        
        # Patient clinical indicators
        has_chf = clinical_history.get("chf", False)
        has_renal_failure = clinical_history.get("renal_failure", False)
        
        # Initial states
        map_val = current_vitals.get("map", current_vitals.get("systolic_bp", 120.0) * 0.6 + current_vitals.get("diastolic_bp", 80.0) * 0.4)
        hr = current_vitals.get("heart_rate", 80.0)
        lactate = current_vitals.get("sweat_lactate", current_vitals.get("lactate", 1.5))
        infection_level = 5.0 if current_vitals.get("sepsis_similarity", 0.0) > 0.4 else 1.0
        fluid_overload = 0.0

        fitness_score = 0.0
        
        # Simulate hourly steps (6 hours)
        for t in range(6):
            # 1. Update infection level based on antibiotic effectiveness
            # ab_code maps: 0=None, 1=PipTazo, 2=Vanco, 3=Cefepime, 4=Mero, 5=Levo
            ab_potency = 0.0
            if ab_code > 0:
                ab_potency = 1.2 if ab_code in [1, 4] else 0.8 # Broad spectrum covers more
            
            infection_level = max(0.0, infection_level - ab_potency)
            
            # 2. Fluid effect on MAP and fluid overload
            # Fluids boost venous return and raise MAP, but decay over time
            fluid_absorption = fluid_rate * 0.15
            map_val += fluid_absorption
            
            # Fluids cause overload, magnified by cardiac (CHF) or renal failure
            overload_multiplier = 2.0 if has_chf else 1.0
            if has_renal_failure:
                overload_multiplier *= 2.5
            fluid_overload += fluid_rate * 0.1 * overload_multiplier

            # 3. Vasopressor effect on MAP (strong vasoconstriction)
            # Norepinephrine increases MAP rapidly
            vaso_effect = vaso_dose * 40.0
            map_val += vaso_effect
            
            # 4. Sepsis vasodilation decay (infection lowers blood pressure)
            sepsis_drop = infection_level * 1.5
            map_val -= sepsis_drop
            
            # Cap MAP in physiological limits
            map_val = min(150.0, max(30.0, map_val))
            
            # 5. Heart Rate compensation
            # Heart rate rises if MAP is low to maintain cardiac output
            if map_val < 65.0:
                hr += (65.0 - map_val) * 1.5
            else:
                hr = max(60.0, hr - (map_val - 65.0) * 0.5)
                
            # Infection also increases HR (fever/tachycardia)
            hr += infection_level * 3.0
            hr = min(200.0, max(40.0, hr))

            # 6. Lactate accumulation due to tissue hypoperfusion (MAP < 65)
            # If MAP is adequate, lactate clears slowly
            if map_val < 65.0:
                lactate += (65.0 - map_val) * 0.08
            else:
                lactate = max(0.5, lactate - 0.15)
            
            # Evaluate penalties for this hour
            # Penalty for hypoperfusion (MAP < 65 is bad, < 60 is critical)
            if map_val < 65.0:
                fitness_score -= (65.0 - map_val) ** 2 * 2.0
            if map_val > 100.0:
                fitness_score -= (map_val - 100.0) * 0.5 # hypertension penalty
                
            # Penalty for tachycardia
            if hr > 100.0:
                fitness_score -= (hr - 100.0) * 0.3
                
            # Penalty for elevated lactate (organ dysfunction marker)
            if lactate > 2.0:
                fitness_score -= (lactate - 2.0) ** 2 * 5.0
                
            # Penalty for fluid overload
            if fluid_overload > 10.0:
                fitness_score -= (fluid_overload - 10.0) * 8.0
                
            # Penalty for vasopressor toxicity (arrhythmia risk at high doses)
            if vaso_dose > 0.15:
                fitness_score -= (vaso_dose - 0.15) * 50.0

        return fitness_score

    def optimize(
        self,
        current_vitals: Dict[str, float],
        clinical_history: Dict[str, Any],
        sepsis_risk: str
    ) -> Tuple[Dict[str, Any], float]:
        """
        Runs the genetic algorithm to evolve the treatment protocol.
        Returns:
            (best_protocol_dict, best_fitness)
        """
        # Initialize population
        # Gene: (fluid_rate [0, 20.0], vasopressor_dose [0, 0.4], antibiotic_code [0, 5])
        population = []
        for _ in range(self.pop_size):
            fluid = self.rng.uniform(0.0, 15.0)
            vaso = self.rng.uniform(0.0, 0.25)
            # If high risk sepsis, bias towards prescribing antibiotics
            if sepsis_risk == "HIGH":
                ab = self.rng.choice([1, 2, 3, 4, 5])
            else:
                ab = self.rng.choice([0, 1, 2, 3, 4, 5], p=[0.7, 0.06, 0.06, 0.06, 0.06, 0.06])
            population.append((fluid, vaso, int(ab)))

        best_individual = population[0]
        best_fitness = -999999.0

        # Evolutionary loop
        for gen in range(self.generations):
            # 1. Evaluate fitness
            fitness_scores = [
                self._project_patient_outcome(ind, current_vitals, clinical_history)
                for ind in population
            ]

            # Track global best
            for idx, fit in enumerate(fitness_scores):
                if fit > best_fitness:
                    best_fitness = fit
                    best_individual = population[idx]

            # 2. Selection (Tournament Selection)
            next_generation = []
            while len(next_generation) < self.pop_size:
                # Select parents
                parents = []
                for _ in range(2):
                    candidates = self.rng.choice(len(population), size=3, replace=False)
                    best_cand_idx = candidates[np.argmax([fitness_scores[c] for c in candidates])]
                    parents.append(population[best_cand_idx])
                
                p1, p2 = parents
                
                # 3. Crossover
                if self.rng.random() < self.crossover_rate:
                    # Blend crossover for floats, random select for antibiotics
                    alpha = self.rng.random()
                    c1_fluid = alpha * p1[0] + (1 - alpha) * p2[0]
                    c2_fluid = (1 - alpha) * p1[0] + alpha * p2[0]
                    
                    c1_vaso = alpha * p1[1] + (1 - alpha) * p2[1]
                    c2_vaso = (1 - alpha) * p1[1] + alpha * p2[1]
                    
                    c1_ab = p1[2] if self.rng.random() < 0.5 else p2[2]
                    c2_ab = p2[2] if self.rng.random() < 0.5 else p1[2]
                    
                    child1 = (c1_fluid, c1_vaso, int(c1_ab))
                    child2 = (c2_fluid, c2_vaso, int(c2_ab))
                else:
                    child1, child2 = p1, p2

                # 4. Mutation
                child1 = self._mutate(child1, sepsis_risk)
                child2 = self._mutate(child2, sepsis_risk)
                
                next_generation.extend([child1, child2])

            population = next_generation[:self.pop_size]

        # Return best protocol mapped to readable dictionary
        fluid_rate_opt, vaso_dose_opt, ab_code_opt = best_individual
        
        return {
            "fluid_rate": round(fluid_rate_opt, 2),
            "vasopressor_dose": round(vaso_dose_opt, 3),
            "antibiotic_code": int(ab_code_opt)
        }, best_fitness

    def _mutate(self, ind: Tuple[float, float, int], sepsis_risk: str) -> Tuple[float, float, int]:
        fluid, vaso, ab = ind
        
        # Mutate fluid rate
        if self.rng.random() < self.mutation_rate:
            fluid = max(0.0, min(25.0, fluid + self.rng.normal(0.0, 2.0)))
            
        # Mutate vasopressor
        if self.rng.random() < self.mutation_rate:
            vaso = max(0.0, min(0.4, vaso + self.rng.normal(0.0, 0.05)))
            
        # Mutate antibiotic code
        if self.rng.random() < self.mutation_rate:
            if sepsis_risk == "HIGH":
                ab = self.rng.choice([1, 2, 3, 4, 5])
            else:
                ab = self.rng.choice([0, 1, 2, 3, 4, 5])
                
        return (fluid, vaso, int(ab))
