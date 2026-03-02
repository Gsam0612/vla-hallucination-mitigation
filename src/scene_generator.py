"""
Scene generation for AI2-THOR environments.

Dissertation Section 4.2 — Three categories of scenes:
1. Simple rooms   — 3-5 objects with clear visibility
2. Cluttered rooms — 8-15 objects (multi-object + spatial hallucination triggers)
3. Hazard scenarios — 5-10 objects including dangerous items (safety testing)
"""

import random
from dataclasses import dataclass, field
from typing import List, Dict, Any

from .objects import (
    AI2THOR_OBJECTS, SPATIAL_RELATIONS, POSITIONS, ROOM_TYPES,
    get_objects_for_room, get_hazard_objects_for_room,
)


# ── Scene data class ────────────────────────────────────────────────

@dataclass
class SceneData:
    """Ground-truth representation of a generated scene."""
    scene_id: str
    room_type: str
    complexity: str            # 'simple' | 'cluttered' | 'hazard'
    objects: List[Dict[str, Any]]
    spatial_relations: List[Dict[str, str]]
    hazards: List[str]

    # ── Accessors ───────────────────────────────────────────────────

    def get_object_list(self) -> List[str]:
        return [obj['name'] for obj in self.objects]

    def get_object_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for obj in self.objects:
            counts[obj['name']] = counts.get(obj['name'], 0) + obj['count']
        return counts

    def get_object_colors(self) -> Dict[str, str]:
        return {obj['name']: obj['color'] for obj in self.objects}

    def get_ground_truth(self) -> Dict[str, Any]:
        """Pack all ground truth into a single dict."""
        return {
            'objects': self.get_object_list(),
            'counts': self.get_object_counts(),
            'colors': self.get_object_colors(),
            'relations': self.spatial_relations,
            'hazards': self.hazards,
        }


# ── Scene generator ─────────────────────────────────────────────────

COUNT_RANGES = {
    'simple':   (3, 5),
    'cluttered': (8, 15),
    'hazard':   (5, 10),
}

COMPLEXITY_DISTRIBUTION = {'simple': 0.4, 'cluttered': 0.4, 'hazard': 0.2}


def generate_scene(
    complexity: str = 'simple',
    room_type: str = 'Kitchen',
) -> SceneData:
    """Generate a single AI2-THOR scene with ground truth.

    Args:
        complexity: 'simple', 'cluttered', or 'hazard'.
        room_type:  'Kitchen', 'LivingRoom', 'Bedroom', or 'Bathroom'.

    Returns:
        SceneData instance with all annotations.
    """
    min_c, max_c = COUNT_RANGES.get(complexity, (3, 5))
    num_objects = random.randint(min_c, max_c)

    room_objects = get_objects_for_room(room_type)

    # ── Select objects ──────────────────────────────────────────────
    if complexity == 'hazard':
        hazard_pool = get_hazard_objects_for_room(room_type)
        selected_hazards = random.sample(
            hazard_pool, min(2, len(hazard_pool)))
        remaining = num_objects - len(selected_hazards)
        others = [o for o in room_objects if o not in selected_hazards]
        selected = selected_hazards + random.sample(
            others, min(remaining, len(others)))
    else:
        selected = random.sample(
            room_objects, min(num_objects, len(room_objects)))

    # ── Build object instances with attributes ──────────────────────
    objects: List[Dict[str, Any]] = []
    for name in selected:
        info = AI2THOR_OBJECTS[name]
        count = random.choices([1, 2, 3], weights=[0.7, 0.2, 0.1])[0]
        objects.append({
            'name': name,
            'color': random.choice(info['colors']),
            'position': random.choice(POSITIONS),
            'count': count,
            'is_hazard': info['hazard'],
            'confidence': round(random.uniform(0.75, 0.99), 2),
        })

    # ── Spatial relations ───────────────────────────────────────────
    relations: List[Dict[str, str]] = []
    if len(objects) >= 2:
        n_rels = min(3, len(objects))
        for _ in range(n_rels):
            o1, o2 = random.sample(objects, 2)
            relations.append({
                'subject': o1['name'],
                'relation': random.choice(SPATIAL_RELATIONS),
                'object': o2['name'],
            })

    hazards = [obj['name'] for obj in objects if obj['is_hazard']]
    scene_id = f"{room_type}_{complexity}_{random.randint(1000, 9999)}"

    return SceneData(
        scene_id=scene_id,
        room_type=room_type,
        complexity=complexity,
        objects=objects,
        spatial_relations=relations,
        hazards=hazards,
    )


def generate_scenes(
    n: int,
    complexity_dist: Dict[str, float] = None,
    rooms: List[str] = None,
) -> List[SceneData]:
    """Generate a batch of diverse scenes.

    Args:
        n: Number of scenes.
        complexity_dist: Probability weights for each complexity.
        rooms: Room types to sample from.

    Returns:
        List of SceneData instances.
    """
    complexity_dist = complexity_dist or COMPLEXITY_DISTRIBUTION
    rooms = rooms or ROOM_TYPES

    scenes = []
    for _ in range(n):
        complexity = random.choices(
            list(complexity_dist.keys()),
            weights=list(complexity_dist.values()),
        )[0]
        room = random.choice(rooms)
        scenes.append(generate_scene(complexity, room))
    return scenes
