"""
AI2-THOR object definitions for VLA Hallucination Mitigation.

Based on dissertation Section 4.2: Three categories of scenes:
1. Simple rooms - Few objects (3-5) with clear visibility
2. Cluttered rooms - Many overlapping objects (8-15)
3. Hazard scenarios - Dangerous items requiring safety awareness
"""

# ── Object catalogue ────────────────────────────────────────────────
AI2THOR_OBJECTS = {
    # Kitchen Objects
    'CoffeeMachine':  {'colors': ['black', 'silver', 'white'],          'rooms': ['Kitchen'],               'hazard': False},
    'Microwave':      {'colors': ['black', 'silver', 'white'],          'rooms': ['Kitchen'],               'hazard': False},
    'Toaster':        {'colors': ['silver', 'black', 'red'],            'rooms': ['Kitchen'],               'hazard': True},
    'Refrigerator':   {'colors': ['silver', 'white', 'black'],          'rooms': ['Kitchen'],               'hazard': False},
    'StoveBurner':    {'colors': ['black'],                             'rooms': ['Kitchen'],               'hazard': True},
    'Mug':            {'colors': ['white', 'red', 'blue', 'black', 'yellow'], 'rooms': ['Kitchen', 'LivingRoom'], 'hazard': False},
    'Cup':            {'colors': ['white', 'clear', 'blue'],            'rooms': ['Kitchen'],               'hazard': False},
    'Plate':          {'colors': ['white', 'blue'],                     'rooms': ['Kitchen'],               'hazard': False},
    'Bowl':           {'colors': ['white', 'blue', 'red'],              'rooms': ['Kitchen'],               'hazard': False},
    'Knife':          {'colors': ['silver'],                            'rooms': ['Kitchen'],               'hazard': True},
    'Fork':           {'colors': ['silver'],                            'rooms': ['Kitchen'],               'hazard': False},
    'Spoon':          {'colors': ['silver'],                            'rooms': ['Kitchen'],               'hazard': False},
    'Pan':            {'colors': ['black', 'silver'],                   'rooms': ['Kitchen'],               'hazard': True},
    'Pot':            {'colors': ['silver', 'black'],                   'rooms': ['Kitchen'],               'hazard': True},
    'Apple':          {'colors': ['red', 'green'],                      'rooms': ['Kitchen'],               'hazard': False},
    'Bread':          {'colors': ['brown'],                             'rooms': ['Kitchen'],               'hazard': False},
    'Egg':            {'colors': ['white', 'brown'],                    'rooms': ['Kitchen'],               'hazard': False},
    'Tomato':         {'colors': ['red'],                               'rooms': ['Kitchen'],               'hazard': False},
    'Lettuce':        {'colors': ['green'],                             'rooms': ['Kitchen'],               'hazard': False},
    'Potato':         {'colors': ['brown'],                             'rooms': ['Kitchen'],               'hazard': False},
    'SinkBasin':      {'colors': ['silver', 'white'],                   'rooms': ['Kitchen', 'Bathroom'],   'hazard': False},
    'Faucet':         {'colors': ['silver', 'chrome'],                  'rooms': ['Kitchen', 'Bathroom'],   'hazard': False},

    # Living Room Objects
    'Television':     {'colors': ['black'],                             'rooms': ['LivingRoom', 'Bedroom'], 'hazard': False},
    'Sofa':           {'colors': ['brown', 'gray', 'blue', 'black'],    'rooms': ['LivingRoom'],            'hazard': False},
    'Chair':          {'colors': ['brown', 'black', 'white', 'blue'],   'rooms': ['Kitchen', 'LivingRoom'], 'hazard': False},
    'DiningTable':    {'colors': ['brown', 'black', 'white'],           'rooms': ['Kitchen', 'LivingRoom'], 'hazard': False},
    'CoffeeTable':    {'colors': ['brown', 'black', 'glass'],           'rooms': ['LivingRoom'],            'hazard': False},
    'Laptop':         {'colors': ['silver', 'black'],                   'rooms': ['LivingRoom', 'Bedroom'], 'hazard': False},
    'Book':           {'colors': ['red', 'blue', 'green', 'brown', 'black'], 'rooms': ['LivingRoom', 'Bedroom'], 'hazard': False},
    'Lamp':           {'colors': ['white', 'black', 'gold', 'silver'],  'rooms': ['LivingRoom', 'Bedroom'], 'hazard': False},
    'RemoteControl':  {'colors': ['black', 'silver'],                   'rooms': ['LivingRoom'],            'hazard': False},
    'Pillow':         {'colors': ['white', 'blue', 'pink', 'gray'],     'rooms': ['Bedroom', 'LivingRoom'], 'hazard': False},

    # Bedroom Objects
    'Bed':            {'colors': ['white', 'blue', 'brown', 'gray'],    'rooms': ['Bedroom'],               'hazard': False},
    'AlarmClock':     {'colors': ['black', 'white', 'red'],             'rooms': ['Bedroom'],               'hazard': False},
    'CellPhone':      {'colors': ['black', 'white', 'silver'],          'rooms': ['Bedroom', 'LivingRoom'], 'hazard': False},

    # Bathroom Objects
    'Toilet':         {'colors': ['white'],                             'rooms': ['Bathroom'],              'hazard': False},
    'Towel':          {'colors': ['white', 'blue', 'pink', 'green'],    'rooms': ['Bathroom'],              'hazard': False},
    'SoapBottle':     {'colors': ['white', 'blue', 'pink'],             'rooms': ['Bathroom'],              'hazard': False},
    'SprayBottle':    {'colors': ['white', 'blue', 'yellow'],           'rooms': ['Bathroom', 'Kitchen'],   'hazard': True},
}

# ── Derived constants ───────────────────────────────────────────────
SPATIAL_RELATIONS = [
    'left_of', 'right_of', 'above', 'below',
    'in_front_of', 'behind', 'on_top_of', 'next_to', 'inside',
]

HAZARD_OBJECTS = [name for name, info in AI2THOR_OBJECTS.items() if info['hazard']]

ROOM_TYPES = ['Kitchen', 'LivingRoom', 'Bedroom', 'Bathroom']

POSITIONS = ['left', 'right', 'center', 'front', 'back', 'far left', 'far right']


def get_objects_for_room(room: str):
    """Return object names available in a given room."""
    return [name for name, info in AI2THOR_OBJECTS.items()
            if room in info['rooms']]


def get_hazard_objects_for_room(room: str):
    """Return hazard-flagged objects in a given room."""
    return [name for name, info in AI2THOR_OBJECTS.items()
            if info['hazard'] and room in info['rooms']]
