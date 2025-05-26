# agents/agent_types.py

# Suppression range as (dx, dy) tuples
AGENT_SUPPRESS_RANGE = {
    'helicopter': [(0,0), (-1,0), (1,0), (0,-1), (0,1)],  # current + 4-adjacent
    'drone': [(0,0), (1,0)],                             # current + right
    'groundcrew': [(0,0)]                                # current only
}

AGENT_COLORS = {
    'helicopter': 'blue',
    'drone': 'purple',
    'groundcrew': 'black'
}
