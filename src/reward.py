"""
Hallucination-aware reward function for GRPO training.

Dissertation Section 4.5 — Reward Structure:
- Negative penalties for hallucinated objects, incorrect attributes, wrong spatial relations
- Positive rewards for correctly mentioning objects, accurate counting, safe recommendations
- Minor penalties for excessive verbosity
- Bonus for multi-view verified objects and proper CoT reasoning
"""

import re
from typing import List, Dict, Any, Optional, Set

from .objects import AI2THOR_OBJECTS
from .config import RewardConfig


class HallucinationReward:
    """Computes hallucination-aware rewards for GRPO training.

    Implements the four-type penalty system from dissertation Section 4.5:
    1. Object existence hallucination  (-1.0)
    2. Object misidentification        (-0.8)
    3. Attribute hallucination          (-0.5)
    4. Spatial relation hallucination   (-0.6)
    """

    # ── Number words for count extraction ───────────────────────────
    NUMBER_WORDS = {
        'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
        'a': 1, 'an': 1, 'single': 1, 'several': 3, 'many': 5,
    }

    COLOR_WORDS = [
        'red', 'blue', 'green', 'yellow', 'black', 'white',
        'silver', 'brown', 'gray', 'grey', 'pink', 'orange',
        'gold', 'chrome', 'clear', 'glass',
    ]

    def __init__(self, config: Optional[RewardConfig] = None):
        self.config = config or RewardConfig()

        # Pre-build lookup: lowercase name -> original name
        self._obj_lookup = {}
        for obj_name in AI2THOR_OBJECTS:
            self._obj_lookup[obj_name.lower()] = obj_name
            # CamelCase -> spaced: "CoffeeMachine" -> "coffee machine"
            spaced = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', obj_name).lower()
            self._obj_lookup[spaced] = obj_name

    # ── Extraction helpers ──────────────────────────────────────────

    def extract_objects(self, text: str) -> Set[str]:
        """Extract AI2-THOR object names mentioned in text."""
        mentioned = set()
        text_lower = text.lower()
        for pattern, canonical in self._obj_lookup.items():
            if pattern in text_lower:
                mentioned.add(canonical)
        return mentioned

    def extract_counts(self, text: str) -> Dict[str, int]:
        """Extract object counts from text (e.g. 'two mugs', '3 cups')."""
        counts: Dict[str, int] = {}
        text_lower = text.lower()

        # Digit patterns: "3 cups"
        for m in re.finditer(r'(\d+)\s+(\w+)', text_lower):
            obj_word = m.group(2).rstrip('s')
            counts.setdefault(obj_word, int(m.group(1)))

        # Word patterns: "two mugs"
        num_words_pat = '|'.join(self.NUMBER_WORDS.keys())
        for m in re.finditer(rf'({num_words_pat})\s+(\w+)', text_lower):
            obj_word = m.group(2).rstrip('s')
            counts.setdefault(obj_word, self.NUMBER_WORDS[m.group(1)])

        return counts

    def extract_colors(self, text: str) -> Dict[str, str]:
        """Extract color attributions from text (e.g. 'red apple')."""
        colors: Dict[str, str] = {}
        text_lower = text.lower()
        for color in self.COLOR_WORDS:
            for m in re.finditer(rf'{color}\s+(\w+)', text_lower):
                obj_word = m.group(1).rstrip('s')
                colors.setdefault(obj_word, color)
        return colors

    def extract_spatial_relations(self, text: str) -> List[Dict[str, str]]:
        """Extract spatial relations from text."""
        relations = []
        text_lower = text.lower()

        patterns = [
            r'(\w+)\s+is\s+(left of|right of|above|below|in front of|behind|on top of|next to|inside)\s+(?:the\s+)?(\w+)',
            r'(\w+)\s+(?:is\s+)?(?:to the\s+)?(left|right)\s+of\s+(?:the\s+)?(\w+)',
        ]
        for pat in patterns:
            for m in re.finditer(pat, text_lower):
                relations.append({
                    'subject': m.group(1),
                    'relation': m.group(2).replace(' ', '_'),
                    'object': m.group(3),
                })
        return relations

    def check_cot_quality(self, response: str) -> float:
        """Score chain-of-thought quality (0.0 – 1.0).

        Checks for presence of structured CoT sections:
        [Observation], [Reasoning], [Multi-View Check], [Verification], [Answer]
        """
        sections = ['[Observation]', '[Reasoning]', '[Verification]', '[Answer]']
        found = sum(1 for s in sections if s.lower() in response.lower())
        return found / len(sections)

    # ── Main reward computation ─────────────────────────────────────

    def compute_reward(
        self,
        response: str,
        gt_objects: List[str],
        gt_counts: Optional[Dict[str, int]] = None,
        gt_colors: Optional[Dict[str, str]] = None,
        gt_relations: Optional[List[Dict[str, str]]] = None,
        gt_hazards: Optional[List[str]] = None,
        multi_view_verified: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Compute hallucination-aware reward for a VLM response.

        Args:
            response:             Generated text from the model.
            gt_objects:           Ground-truth list of object names.
            gt_counts:            Ground-truth object counts.
            gt_colors:            Ground-truth color map {object: color}.
            gt_relations:         Ground-truth spatial relations.
            gt_hazards:           List of hazard objects in scene.
            multi_view_verified:  Objects confirmed across multiple views.

        Returns:
            Dict with total_reward, bonuses, penalties, metrics.
        """
        gt_counts = gt_counts or {}
        gt_colors = gt_colors or {}
        gt_relations = gt_relations or []
        gt_hazards = gt_hazards or []
        multi_view_verified = multi_view_verified or []

        gt_set = {o.lower() for o in gt_objects}
        mv_set = {o.lower() for o in multi_view_verified}

        mentioned = self.extract_objects(response)
        mentioned_lower = {o.lower() for o in mentioned}
        mentioned_counts = self.extract_counts(response)
        mentioned_colors = self.extract_colors(response)

        reward = self.config.base_reward
        penalties: Dict[str, float] = {}
        bonuses: Dict[str, float] = {}
        details: Dict[str, Any] = {
            'mentioned': list(mentioned),
            'gt_objects': gt_objects,
            'hallucinated': [],
            'missing': [],
            'correct': [],
        }

        # 1. Object existence hallucination (worst penalty)
        hallucinated = mentioned_lower - gt_set
        if hallucinated:
            p = len(hallucinated) * self.config.object_existence_penalty
            penalties['object_existence'] = p
            reward += p
            details['hallucinated'] = list(hallucinated)

        # 2. Correct objects (bonus)
        correct = mentioned_lower & gt_set
        if correct:
            b = len(correct) * self.config.correct_object_bonus
            bonuses['correct_objects'] = b
            reward += b
            details['correct'] = list(correct)

        # 3. Multi-view consistency bonus
        mv_confirmed = mentioned_lower & mv_set
        if mv_confirmed:
            b = len(mv_confirmed) * self.config.multi_view_consistency_bonus
            bonuses['multi_view'] = b
            reward += b

        # 4. Missing objects (mild penalty)
        missing = gt_set - mentioned_lower
        if missing:
            p = len(missing) * self.config.missing_object_penalty
            penalties['missing_objects'] = p
            reward += p
            details['missing'] = list(missing)

        # 5. Attribute hallucination (colors)
        color_errors = 0
        for obj_word, mentioned_color in mentioned_colors.items():
            gt_color = gt_colors.get(obj_word) or gt_colors.get(obj_word.rstrip('s'))
            if gt_color and mentioned_color != gt_color.lower():
                color_errors += 1
        if color_errors:
            p = color_errors * self.config.attribute_error_penalty
            penalties['color_errors'] = p
            reward += p

        # 6. Count hallucination
        count_errors = 0
        count_correct = 0
        for obj, gt_count in gt_counts.items():
            mc = mentioned_counts.get(obj.lower(), 0)
            if mc > 0:
                if mc != gt_count:
                    count_errors += 1
                else:
                    count_correct += 1
        if count_errors:
            p = count_errors * self.config.attribute_error_penalty
            penalties['count_errors'] = p
            reward += p
        if count_correct:
            b = count_correct * self.config.correct_count_bonus
            bonuses['correct_counts'] = b
            reward += b

        # 7. Safety / hazard awareness
        if gt_hazards:
            hazards_found = sum(1 for h in gt_hazards if h.lower() in mentioned_lower)
            if hazards_found > 0:
                b = hazards_found * self.config.safety_awareness_bonus
                bonuses['safety'] = b
                reward += b

        # 8. CoT quality bonus
        cot_score = self.check_cot_quality(response)
        if cot_score > 0:
            b = cot_score * self.config.cot_quality_bonus
            bonuses['cot_quality'] = b
            reward += b

        # 9. Grounding bonus (response stays within detected objects)
        if mentioned and len(hallucinated) == 0:
            bonuses['grounded'] = self.config.grounded_response_bonus
            reward += self.config.grounded_response_bonus

        # Clip
        reward = max(self.config.min_reward, min(self.config.max_reward, reward))

        return {
            'total_reward': reward,
            'bonuses': bonuses,
            'penalties': penalties,
            'details': details,
            'hallucination_rate': len(hallucinated) / max(len(mentioned), 1),
            'recall': len(correct) / max(len(gt_set), 1),
            'precision': len(correct) / max(len(mentioned), 1),
            'multi_view_rate': len(mv_confirmed) / max(len(mentioned), 1),
            'cot_quality': cot_score,
        }
