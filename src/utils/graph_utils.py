# ontological-playground-designer/src/utils/graph_utils.py

import networkx as nx
from typing import List, Dict, Any, Tuple
import json

# Ensure loguru is set up for structured logging
from loguru import logger
from src.utils.logger import setup_logging

# Setup logging for this module
setup_logging()

def create_conceptual_graph() -> nx.DiGraph:
    """
    Creates and returns an empty directed graph for conceptual relationships.
    This graph will be used to represent axioms, rules, and their interdependencies.
    """
    logger.debug("Created an empty conceptual graph.")
    return nx.DiGraph()

def add_node_to_graph(graph: nx.DiGraph, node_id: str, attributes: Dict[str, Any]):
    """
    Adds a node to the conceptual graph with specified attributes.
    Attributes might include 'type' (e.g., 'axiom', 'rule', 'entity'), 'description', etc.
    """
    if not graph.has_node(node_id):
        graph.add_node(node_id, **attributes)
        logger.debug(f"Added node: {node_id} with attributes: {attributes.keys()}")
    else:
        logger.warning(f"Node '{node_id}' already exists. Updating attributes.")
        graph.nodes[node_id].update(attributes)

def add_edge_to_graph(graph: nx.DiGraph, u: str, v: str, attributes: Dict[str, Any]):
    """
    Adds a directed edge between two nodes with specified attributes.
    Attributes might include 'relation' (e.g., 'influences', 'contradicts', 'depends_on').
    Ensures nodes exist before adding edge.
    """
    if not graph.has_node(u):
        logger.error(f"Cannot add edge: Source node '{u}' does not exist.")
        raise ValueError(f"Source node '{u}' not found.")
    if not graph.has_node(v):
        logger.error(f"Cannot add edge: Target node '{v}' does not exist.")
        raise ValueError(f"Target node '{v}' not found.")

    if not graph.has_edge(u, v):
        graph.add_edge(u, v, **attributes)
        logger.debug(f"Added edge from {u} to {v} with relation: {attributes.get('relation', 'unspecified')}")
    else:
        logger.warning(f"Edge from {u} to {v} already exists. Updating attributes.")
        graph.edges[u, v].update(attributes)

def get_node_attributes(graph: nx.DiGraph, node_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves all attributes for a given node."""
    if graph.has_node(node_id):
        return graph.nodes[node_id]
    logger.warning(f"Node '{node_id}' not found in graph.")
    return None

def get_edge_attributes(graph: nx.DiGraph, u: str, v: str) -> Optional[Dict[str, Any]]:
    """Retrieves all attributes for a given edge."""
    if graph.has_edge(u, v):
        return graph.edges[u, v]
    logger.warning(f"Edge from {u} to {v} not found in graph.")
    return None

def find_paths_between_nodes(graph: nx.DiGraph, source: str, target: str, max_length: int = 5) -> List[List[str]]:
    """
    Finds all simple paths between two nodes up to a maximum length.
    Useful for tracing dependencies or potential conflict pathways.
    """
    if not graph.has_node(source) or not graph.has_node(target):
        logger.warning(f"Source '{source}' or target '{target}' not found for path finding.")
        return []
    
    paths = list(nx.all_simple_paths(graph, source, target, cutoff=max_length))
    logger.debug(f"Found {len(paths)} paths from {source} to {target} (max length {max_length}).")
    return paths

def get_subgraph_around_node(graph: nx.DiGraph, center_node: str, radius: int = 2) -> nx.DiGraph:
    """
    Returns a subgraph centered around a specific node, including nodes
    within a given 'radius' (number of hops).
    """
    if not graph.has_node(center_node):
        logger.warning(f"Center node '{center_node}' not found for subgraph extraction.")
        return nx.DiGraph()

    # Get neighbors within radius
    nodes_in_subgraph = nx.ego_graph(graph, center_node, radius=radius, undirected=False)
    # Convert to directed graph
    subgraph = graph.subgraph(nodes_in_subgraph).copy()

    logger.debug(f"Extracted subgraph around '{center_node}' with {subgraph.number_of_nodes()} nodes and {subgraph.number_of_edges()} edges.")
    return subgraph

def convert_graph_to_json_serializable(graph: nx.DiGraph) -> Dict[str, Any]:
    """
    Converts a networkx.DiGraph object into a JSON-serializable dictionary format.
    Useful for saving graph structures (e.g., in generated world configs or reports).
    """
    nodes_data = [{'id': node_id, **data} for node_id, data in graph.nodes(data=True)]
    edges_data = [{'source': u, 'target': v, **data} for u, v, data in graph.edges(data=True)]
    
    serializable_graph = {
        'nodes': nodes_data,
        'edges': edges_data
    }
    logger.debug("Converted graph to JSON serializable format.")
    return serializable_graph

def reconstruct_graph_from_json_serializable(data: Dict[str, Any]) -> nx.DiGraph:
    """
    Reconstructs a networkx.DiGraph from its JSON-serializable dictionary format.
    """
    graph = nx.DiGraph()
    for node_data in data.get('nodes', []):
        node_id = node_data.pop('id') # 'id' is special for networkx
        graph.add_node(node_id, **node_data)
    for edge_data in data.get('edges', []):
        source = edge_data.pop('source')
        target = edge_data.pop('target')
        graph.add_edge(source, target, **edge_data)
    logger.debug("Reconstructed graph from JSON serializable format.")
    return graph

# --- Example Usage (for testing and demonstration) ---
if __name__ == "__main__":
    # Ensure src/utils directory and logger.py exist for setup_logging
    import os
    if not os.path.exists("src/utils"):
        os.makedirs("src/utils")
        # Assuming logger.py is already there or will be created next

    logger.info("--- Demonstrating Graph Utilities ---")

    # 1. Create a graph
    my_graph = create_conceptual_graph()

    # 2. Add nodes
    add_node_to_graph(my_graph, "Axiom_Flourishing", {"type": "axiom", "description": "Maximize well-being"})
    add_node_to_graph(my_graph, "Rule_Cooperation", {"type": "rule", "description": "Agents cooperate"})
    add_node_to_graph(my_graph, "Rule_Competition", {"type": "rule", "description": "Agents compete"})
    add_node_to_graph(my_graph, "Entity_Agent", {"type": "entity", "description": "Simulated agent"})
    add_node_to_graph(my_graph, "Paradox_Ethical", {"type": "paradox", "description": "Ethical conflict"})

    # 3. Add edges
    add_edge_to_graph(my_graph, "Axiom_Flourishing", "Rule_Cooperation", {"relation": "influences", "strength": 0.9})
    add_edge_to_graph(my_graph, "Axiom_Flourishing", "Rule_Competition", {"relation": "conflicts_with", "strength": 0.7})
    add_edge_to_graph(my_graph, "Rule_Cooperation", "Entity_Agent", {"relation": "applies_to"})
    add_edge_to_graph(my_graph, "Rule_Competition", "Entity_Agent", {"relation": "applies_to"})
    add_edge_to_graph(my_graph, "Rule_Cooperation", "Paradox_Ethical", {"relation": "causes", "severity": 0.5})
    add_edge_to_graph(my_graph, "Rule_Competition", "Paradox_Ethical", {"relation": "causes", "severity": 0.8})

    # 4. Get node/edge attributes
    logger.info(f"\nNode Axiom_Flourishing attributes: {get_node_attributes(my_graph, 'Axiom_Flourishing')}")
    logger.info(f"Edge Axiom_Flourishing -> Rule_Cooperation attributes: {get_edge_attributes(my_graph, 'Axiom_Flourishing', 'Rule_Cooperation')}")

    # 5. Find paths
    paths = find_paths_between_nodes(my_graph, "Axiom_Flourishing", "Paradox_Ethical")
    logger.info(f"\nPaths from Axiom_Flourishing to Paradox_Ethical: {paths}")

    # 6. Get subgraph
    subgraph = get_subgraph_around_node(my_graph, "Paradox_Ethical", radius=1)
    logger.info(f"\nSubgraph around Paradox_Ethical (radius 1): Nodes={subgraph.nodes()}, Edges={subgraph.edges()}")

    # 7. Serialize and reconstruct
    json_graph = convert_graph_to_json_serializable(my_graph)
    reconstructed_graph = reconstruct_graph_from_json_serializable(json_graph)
    logger.info(f"\nOriginal graph nodes: {my_graph.number_of_nodes()}, Reconstructed graph nodes: {reconstructed_graph.number_of_nodes()}")
    logger.info(f"Original graph edges: {my_graph.number_of_edges()}, Reconstructed graph edges: {reconstructed_graph.number_of_edges()}")
    assert nx.is_isomorphic(my_graph, reconstructed_graph), "Reconstructed graph is not isomorphic to original!"
    logger.success("Graph serialization/reconstruction successful and isomorphic.")
