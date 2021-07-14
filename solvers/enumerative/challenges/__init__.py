from challenges.challenge import *
from challenges.solutions import *

def contains_node(root, x_node):
    return root is x_node or (hasattr(root, "children") and any(contains_node(k, x_node) for k in root.children))
