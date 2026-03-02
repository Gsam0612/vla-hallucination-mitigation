"""
Multi-view consistency checking for hallucination mitigation.

Dissertation Section 4.4.2 — Multi-View Consistency:
- Captures 4 views: front (0°), left (-45°), right (+45°), overhead (-90°)
- Aggregation methods: intersection, majority, union
- Objects must be detected in ≥threshold fraction of views to be validated
- Reduces hallucination by cross-checking object presence across camera angles

Drawing inspiration from MIHBench [3]:
  "hallucinations may occur or change depending on the camera view
   when you do not impose cross-view consistency"
"""

import re
from typing import List, Dict, Any, Optional

from .objects import AI2THOR_OBJECTS
from .config import MultiViewConfig


class MultiViewConsistency:
    """Cross-view object verification to filter hallucinations.

    Usage:
        checker = MultiViewConsistency()
        # Simulated detections from 4 camera angles
        views = {
            'front':    ['CoffeeMachine', 'Mug', 'Apple'],
            'left':     ['CoffeeMachine', 'Mug', 'Plate'],
            'right':    ['CoffeeMachine', 'Apple'],
            'overhead': ['CoffeeMachine', 'Mug', 'Apple', 'Plate'],
        }
        result = checker.check_consistency(views)
        # result['consistent']   -> objects in ≥50% of views
        # result['inconsistent'] -> objects in <50% of views
    """

    def __init__(self, config: Optional[MultiViewConfig] = None):
        self.config = config or MultiViewConfig()

    # ── Core consistency check ──────────────────────────────────────

    def check_consistency(
        self, view_detections: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        """Score each object by the fraction of views in which it appears.

        Args:
            view_detections: {view_name: [object_names]} from detector.

        Returns:
            Dict with 'consistent', 'inconsistent', 'scores', meta.
        """
        if not view_detections:
            return {'consistent': [], 'inconsistent': [], 'scores': {},
                    'num_views': 0, 'threshold': self.config.consistency_threshold}

        # Collect all unique objects (lowercased)
        all_objects: set = set()
        for objs in view_detections.values():
            all_objects.update(o.lower() for o in objs)

        num_views = len(view_detections)
        scores: Dict[str, float] = {}

        for obj in all_objects:
            count = sum(
                1 for view_objs in view_detections.values()
                if obj in {o.lower() for o in view_objs}
            )
            scores[obj] = count / num_views

        consistent = [o for o, s in scores.items()
                      if s >= self.config.consistency_threshold]
        inconsistent = [o for o, s in scores.items()
                        if s < self.config.consistency_threshold]

        return {
            'consistent': sorted(consistent),
            'inconsistent': sorted(inconsistent),
            'scores': scores,
            'num_views': num_views,
            'threshold': self.config.consistency_threshold,
        }

    # ── Aggregation ─────────────────────────────────────────────────

    def aggregate_detections(
        self, view_detections: Dict[str, List[str]]
    ) -> List[str]:
        """Aggregate detections using the configured method.

        Methods (Section 4.4.2):
        - 'intersection': Only objects in ALL views (strictest)
        - 'majority':     Objects in >threshold of views (default)
        - 'union':        Objects in ANY view (most permissive)
        """
        if not view_detections:
            return []

        sets = [set(o.lower() for o in objs)
                for objs in view_detections.values()]

        if self.config.aggregation == 'intersection':
            result = sets[0]
            for s in sets[1:]:
                result &= s
            return sorted(result)

        elif self.config.aggregation == 'majority':
            return self.check_consistency(view_detections)['consistent']

        elif self.config.aggregation == 'union':
            result: set = set()
            for s in sets:
                result |= s
            return sorted(result)

        return []

    # ── VLM response filtering ──────────────────────────────────────

    def filter_hallucinations(
        self,
        vlm_response: str,
        view_detections: Dict[str, List[str]],
    ) -> Dict[str, Any]:
        """Compare VLM response to multi-view aggregated detections.

        Returns:
            Dict with grounded objects, potential hallucinations,
            and grounding rate.
        """
        # Extract objects from VLM text
        mentioned: set = set()
        response_lower = vlm_response.lower()
        for obj_name in AI2THOR_OBJECTS:
            obj_lower = obj_name.lower()
            spaced = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', obj_name).lower()
            if obj_lower in response_lower or spaced in response_lower:
                mentioned.add(obj_lower)

        # Aggregated consistent objects
        consistent = set(self.aggregate_detections(view_detections))

        grounded = mentioned & consistent
        hallucinated = mentioned - consistent

        return {
            'mentioned': sorted(mentioned),
            'grounded': sorted(grounded),
            'potential_hallucinations': sorted(hallucinated),
            'grounding_rate': len(grounded) / max(len(mentioned), 1),
        }

    # ── View simulation (for training data) ─────────────────────────

    @staticmethod
    def simulate_views(
        gt_objects: List[str],
        noise_rate: float = 0.15,
        num_views: int = 4,
    ) -> Dict[str, List[str]]:
        """Simulate multi-view detections from ground-truth objects.

        Each view may randomly drop some objects (simulating occlusion)
        and occasionally add a false detection (simulating detector noise).

        Args:
            gt_objects:  Ground-truth object list.
            noise_rate:  Probability of missing/adding per object per view.
            num_views:   Number of views to simulate.

        Returns:
            {view_name: [detected_objects]} dictionary.
        """
        import random

        view_names = ['front', 'left', 'right', 'overhead'][:num_views]
        views: Dict[str, List[str]] = {}

        for view_name in view_names:
            detected = []
            for obj in gt_objects:
                # Object may be occluded in some views
                if random.random() > noise_rate:
                    detected.append(obj)
            # Small chance of a false positive detection
            if random.random() < noise_rate * 0.5:
                all_names = list(AI2THOR_OBJECTS.keys())
                false_pos = random.choice(all_names)
                if false_pos not in gt_objects:
                    detected.append(false_pos)
            views[view_name] = detected

        return views
