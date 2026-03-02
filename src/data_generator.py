"""
Training data generator with CoT format, detector grounding, and multi-view.

Dissertation Section 4.4:
- §4.4.1  Object-Detection Grounding   — detector output injected into prompt
- §4.4.2  Multi-View Consistency        — 4 views, consistency threshold
- §4.4.3  Chain-of-Thought + Self-Verify — structured reasoning format

Training sample format:
  SYSTEM: You are a vision-language agent...
  USER: <image>
  Detected objects: {list with confidences}
  Question: {question}
  ASSISTANT:
  [Observation] ...
  [Multi-View Check] ...
  [Reasoning] ...
  [Verification] ...
  [Answer] ...
"""

import random
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from .objects import AI2THOR_OBJECTS, SPATIAL_RELATIONS
from .scene_generator import SceneData, generate_scene, ROOM_TYPES, COMPLEXITY_DISTRIBUTION
from .multi_view import MultiViewConsistency


# ── Question templates ──────────────────────────────────────────────

QUESTION_TEMPLATES = {
    'scene_description': [
        "What do you see in front of you?",
        "Describe all objects visible in this scene.",
        "What objects are in this image?",
        "List everything you can see.",
        "What's in this room?",
        "Describe the scene in detail.",
    ],
    'object_location': [
        "Where is the {object}?",
        "Can you find the {object}?",
        "Where can I find the {object}?",
        "What is the position of the {object}?",
    ],
    'object_color': [
        "What color is the {object}?",
        "What's the color of the {object}?",
        "Describe the color of the {object}.",
    ],
    'object_count': [
        "How many {object}s are there?",
        "Count the {object}s in this scene.",
        "How many {object}s can you see?",
    ],
    'object_existence': [
        "Is there a {object} in the scene?",
        "Do you see a {object}?",
        "Can you spot a {object}?",
        "Is a {object} present in the image?",
    ],
    'spatial_relation': [
        "Where is {object1} relative to {object2}?",
        "Is {object1} on the left or right of {object2}?",
        "What is near the {object1}?",
    ],
    'hazard_awareness': [
        "Are there any dangerous objects in the scene?",
        "What hazards can you identify?",
        "Is it safe to reach for objects in this scene?",
        "What should I be careful of?",
    ],
}


# ── System prompt ───────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a precise vision-language agent operating in an indoor environment. "
    "You MUST only describe objects confirmed by the detector and verified across "
    "multiple camera views. Use the structured reasoning format: "
    "[Observation], [Multi-View Check], [Reasoning], [Verification], [Answer]. "
    "Never mention objects not in the detector output. "
    "If uncertain, say so explicitly."
)


# ── Detector prompt builder ─────────────────────────────────────────

def build_detector_prompt(scene: SceneData) -> str:
    """Build a simulated YOLOv8 detector output string (Section 4.4.1).

    Example output:
      Detected objects: coffee machine (conf: 0.95), mug (conf: 0.88), apple (conf: 0.92)
    """
    detections = []
    for obj in scene.objects:
        name_lower = obj['name'].lower()
        # Insert space before uppercase in CamelCase
        import re
        name_spaced = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', obj['name']).lower()
        conf = obj.get('confidence', round(random.uniform(0.75, 0.99), 2))
        detections.append(f"{name_spaced} (conf: {conf:.2f})")
    return "Detected objects: " + ", ".join(detections)


# ── Multi-view summary builder ──────────────────────────────────────

def build_multiview_summary(
    scene: SceneData,
    mv_checker: MultiViewConsistency,
) -> tuple:
    """Simulate multi-view detections and produce summary text.

    Returns:
        (summary_text, consistent_objects, view_detections)
    """
    gt_objects = scene.get_object_list()
    views = MultiViewConsistency.simulate_views(gt_objects, noise_rate=0.15)
    result = mv_checker.check_consistency(views)

    parts = []
    for obj in gt_objects:
        obj_lower = obj.lower()
        score = result['scores'].get(obj_lower, 0)
        views_seen = int(score * result['num_views'])
        status = "confirmed" if score >= result['threshold'] else "uncertain"
        import re
        name_spaced = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', obj).lower()
        parts.append(f"{name_spaced}: {views_seen}/{result['num_views']} views ({status})")

    summary = " | ".join(parts)
    return summary, result['consistent'], views


# ── CoT answer builder ──────────────────────────────────────────────

def build_cot_answer(
    scene: SceneData,
    question_type: str,
    question: str,
    mv_summary: str,
    consistent_objects: List[str],
    target_obj: Optional[Dict] = None,
    target_relation: Optional[Dict] = None,
    absent_name: Optional[str] = None,
) -> str:
    """Build a structured Chain-of-Thought answer (Section 4.4.3).

    Format:
      [Observation] ...
      [Multi-View Check] ...
      [Reasoning] ...
      [Verification] ...
      [Answer] ...
    """
    import re

    # Helper to format object name
    def fmt(name):
        return re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', name).lower()

    # ── Observation ─────────────────────────────────────────────────
    det_list = ", ".join(
        f"{fmt(o['name'])} ({o['color']}, conf: {o.get('confidence', 0.90):.2f})"
        for o in scene.objects
    )
    observation = f"[Observation] The detector identified: {det_list}."

    # ── Multi-View Check ────────────────────────────────────────────
    mv_check = f"[Multi-View Check] {mv_summary}"

    # ── Reasoning + Verification + Answer per question type ─────────
    if question_type == 'scene_description':
        # Only mention consistent objects
        descs = []
        for obj in scene.objects:
            if obj['name'].lower() in consistent_objects:
                cnt = obj['count']
                if cnt > 1:
                    descs.append(f"{cnt} {obj['color']} {fmt(obj['name'])}s")
                else:
                    descs.append(f"a {obj['color']} {fmt(obj['name'])}")
        if len(descs) > 1:
            answer_text = f"I can see {', '.join(descs[:-1])}, and {descs[-1]}."
        elif descs:
            answer_text = f"I can see {descs[0]}."
        else:
            answer_text = "I cannot confidently identify any objects in this scene."

        reasoning = "[Reasoning] I will only list objects confirmed by the detector and verified in multiple views."
        verification = "[Verification] All mentioned objects appear in the detector output and pass the multi-view consistency check. No undetected objects are mentioned."
        answer = f"[Answer] {answer_text}"

    elif question_type == 'object_location' and target_obj:
        obj = target_obj
        reasoning = f"[Reasoning] The {fmt(obj['name'])} was detected with confidence {obj.get('confidence', 0.90):.2f} and appears to be on the {obj['position']} side."
        verification = f"[Verification] The {fmt(obj['name'])} is {'confirmed' if obj['name'].lower() in consistent_objects else 'uncertain'} across multiple views."
        answer = f"[Answer] The {fmt(obj['name'])} is located on the {obj['position']} side of the scene."

    elif question_type == 'object_color' and target_obj:
        obj = target_obj
        reasoning = f"[Reasoning] The detector identifies the object as a {fmt(obj['name'])}. Checking its visual attributes for color."
        verification = f"[Verification] The {fmt(obj['name'])} color appears consistent across views where it is visible."
        answer = f"[Answer] The {fmt(obj['name'])} is {obj['color']}."

    elif question_type == 'object_count' and target_obj:
        obj = target_obj
        count = obj['count']
        reasoning = f"[Reasoning] The detector reports {count} instance(s) of {fmt(obj['name'])}."
        verification = f"[Verification] Count is consistent with multi-view observations."
        if count == 1:
            answer = f"[Answer] There is 1 {fmt(obj['name'])} in the scene."
        else:
            answer = f"[Answer] There are {count} {fmt(obj['name'])}s in the scene."

    elif question_type == 'object_existence':
        if target_obj:
            obj = target_obj
            status = "confirmed" if obj['name'].lower() in consistent_objects else "detected but unverified"
            reasoning = f"[Reasoning] The {fmt(obj['name'])} is {status} in the current scene."
            verification = f"[Verification] Checking detector output and multi-view agreement for {fmt(obj['name'])}."
            answer = f"[Answer] Yes, there is a {obj['color']} {fmt(obj['name'])} in the scene."
        elif absent_name:
            reasoning = f"[Reasoning] I checked the detector output for {fmt(absent_name)} but it was not found."
            verification = f"[Verification] The {fmt(absent_name)} does not appear in any camera view."
            answer = f"[Answer] No, I don't see a {fmt(absent_name)} in this scene."
        else:
            reasoning = "[Reasoning] Checking detector output."
            verification = "[Verification] No matching detections."
            answer = "[Answer] I cannot confirm the presence of that object."

    elif question_type == 'spatial_relation' and target_relation:
        rel = target_relation
        reasoning = f"[Reasoning] Based on detected positions, the {fmt(rel['subject'])} appears {rel['relation'].replace('_', ' ')} the {fmt(rel['object'])}."
        verification = f"[Verification] Spatial relationship checked across available views."
        answer = f"[Answer] The {fmt(rel['subject'])} is {rel['relation'].replace('_', ' ')} the {fmt(rel['object'])}."

    elif question_type == 'hazard_awareness':
        if scene.hazards:
            hazard_list = ', '.join(fmt(h) for h in scene.hazards)
            reasoning = f"[Reasoning] Scanning detector output for known hazard types. Found: {hazard_list}."
            verification = "[Verification] Hazard objects confirmed in detector output and multi-view check."
            answer = f"[Answer] Yes, be careful of the {hazard_list}. These could be dangerous."
        else:
            reasoning = "[Reasoning] Scanning detector output for known hazard types. None found."
            verification = "[Verification] No hazardous objects detected in any view."
            answer = "[Answer] The scene appears safe. I don't see any hazardous objects."
    else:
        reasoning = "[Reasoning] Analyzing the scene based on detector output."
        verification = "[Verification] Cross-checking with multi-view observations."
        answer = "[Answer] I can see various objects in the scene."

    return f"{observation}\n{mv_check}\n{reasoning}\n{verification}\n{answer}"


# ── Full conversation builder ───────────────────────────────────────

def build_conversation(
    question: str,
    answer: str,
    detector_prompt: str,
) -> str:
    """Build full conversation in LLaVA format.

    Format:
      USER: <image>
      {system_prompt}
      {detector_prompt}
      Question: {question}
      ASSISTANT: {answer}
    """
    return (
        f"USER: "
        f"{SYSTEM_PROMPT}\n"
        f"{detector_prompt}\n"
        f"Question: {question}\n"
        f"ASSISTANT: {answer}"
    )


# ── Sample generator ────────────────────────────────────────────────

def generate_training_sample(
    scene: SceneData,
    mv_checker: MultiViewConsistency,
) -> Dict[str, Any]:
    """Generate one training sample with CoT, grounding, and multi-view.

    Returns dict with: conversation, question, answer, question_type,
    scene_id, ground_truth, detector_prompt, mv_summary.
    """
    qtype = random.choice(list(QUESTION_TEMPLATES.keys()))

    target_obj = None
    target_relation = None
    absent_name = None

    # ── Select question ─────────────────────────────────────────────
    if qtype == 'scene_description':
        question = random.choice(QUESTION_TEMPLATES['scene_description'])

    elif qtype == 'object_location':
        target_obj = random.choice(scene.objects)
        question = random.choice(QUESTION_TEMPLATES['object_location']).format(
            object=target_obj['name'].lower())

    elif qtype == 'object_color':
        target_obj = random.choice(scene.objects)
        question = random.choice(QUESTION_TEMPLATES['object_color']).format(
            object=target_obj['name'].lower())

    elif qtype == 'object_count':
        target_obj = random.choice(scene.objects)
        question = random.choice(QUESTION_TEMPLATES['object_count']).format(
            object=target_obj['name'].lower())

    elif qtype == 'object_existence':
        if random.random() > 0.5 and scene.objects:
            target_obj = random.choice(scene.objects)
            question = random.choice(QUESTION_TEMPLATES['object_existence']).format(
                object=target_obj['name'].lower())
        else:
            present = {o['name'] for o in scene.objects}
            absent = [n for n in AI2THOR_OBJECTS if n not in present]
            if absent:
                absent_name = random.choice(absent)
                question = random.choice(QUESTION_TEMPLATES['object_existence']).format(
                    object=absent_name.lower())
            else:
                target_obj = scene.objects[0]
                question = random.choice(QUESTION_TEMPLATES['object_existence']).format(
                    object=target_obj['name'].lower())

    elif qtype == 'spatial_relation':
        if scene.spatial_relations:
            target_relation = random.choice(scene.spatial_relations)
            question = f"Where is the {target_relation['subject'].lower()} relative to the {target_relation['object'].lower()}?"
        else:
            target_obj = random.choice(scene.objects)
            qtype = 'object_location'
            question = f"Where is the {target_obj['name'].lower()}?"

    elif qtype == 'hazard_awareness':
        question = random.choice(QUESTION_TEMPLATES['hazard_awareness'])

    else:
        question = "What do you see?"
        qtype = 'scene_description'

    # ── Build components ────────────────────────────────────────────
    detector_prompt = build_detector_prompt(scene)
    mv_summary, consistent_objects, view_detections = build_multiview_summary(
        scene, mv_checker)

    cot_answer = build_cot_answer(
        scene, qtype, question, mv_summary, consistent_objects,
        target_obj=target_obj,
        target_relation=target_relation,
        absent_name=absent_name,
    )

    conversation = build_conversation(question, cot_answer, detector_prompt)

    return {
        'conversation': conversation,
        'question': question,
        'answer': cot_answer,
        'question_type': qtype,
        'scene_id': scene.scene_id,
        'detector_prompt': detector_prompt,
        'mv_summary': mv_summary,
        'ground_truth': scene.get_ground_truth(),
        'multi_view_verified': consistent_objects,
        'view_detections': view_detections,
        'scene': {
            'room_type': scene.room_type,
            'complexity': scene.complexity,
            'object_list': scene.get_object_list(),
            'objects': scene.objects,
            'hazards': scene.hazards,
            'spatial_relations': scene.spatial_relations,
        },
    }


# ── Dataset generation ──────────────────────────────────────────────

def generate_dataset(
    n: int = 5000,
    complexity_dist: Dict[str, float] = None,
    rooms: List[str] = None,
    mv_checker: MultiViewConsistency = None,
    show_progress: bool = True,
) -> List[Dict[str, Any]]:
    """Generate a complete training / evaluation dataset.

    Args:
        n:               Number of samples.
        complexity_dist: Complexity distribution.
        rooms:           Room types.
        mv_checker:      MultiViewConsistency instance.
        show_progress:   Show tqdm bar.

    Returns:
        List of training sample dicts.
    """
    complexity_dist = complexity_dist or COMPLEXITY_DISTRIBUTION
    rooms = rooms or ROOM_TYPES
    mv_checker = mv_checker or MultiViewConsistency()

    data = []
    iterator = tqdm(range(n), desc="Generating data") if show_progress else range(n)
    for _ in iterator:
        complexity = random.choices(
            list(complexity_dist.keys()),
            weights=list(complexity_dist.values()),
        )[0]
        room = random.choice(rooms)
        scene = generate_scene(complexity, room)
        sample = generate_training_sample(scene, mv_checker)
        data.append(sample)

    return data
