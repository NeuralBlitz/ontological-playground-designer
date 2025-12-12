# ontological-playground-designer/tests/unit/test_graph_utils.py

import pytest
import networkx as nx
import json
import os
from typing import Dict, Any, List

# Ensure src/ is in sys.path for absolute imports
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.graph_utils import (
    create_conceptual_graph,
    add_node_to_graph,
    add_edge_to_graph,
    get_node_attributes,
    get_edge_attributes,
    find_paths_between_nodes,
    get_subgraph_around_node,
    convert_graph_to_json_serializable,
    reconstruct_graph_from_json_serializable
)
from src.utils.logger import setup_logging # Ensure logging is set up for tests

# Setup logging once for tests
setup_logging(log_level="DEBUG", log_file="logs/test_graph_utils.log")

# --- Fixtures for reusable test data ---

@pytest.fixture
def empty_graph():
    """Returns an empty conceptual graph."""
    return create_conceptual_graph()

@pytest.fixture
def populated_graph():
    """Returns a pre-populated conceptual graph for testing."""
    graph = create_conceptual_graph()
    add_node_to_graph(graph, "Axiom_A", {"type": "axiom", "description": "Axiom A"})
    add_node_to_graph(graph, "Rule_1", {"type": "rule", "description": "Rule 1"})
    add_node_to_graph(graph, "Rule_2", {"type": "rule", "description": "Rule 2"})
    add_node_to_graph(graph, "Agent_X", {"type": "entity", "description": "Agent X"})
    add_node_to_graph(graph, "Goal_Z", {"type": "goal", "description": "Goal Z"})

    add_edge_to_graph(graph, "Axiom_A", "Rule_1", {"relation": "influences", "weight": 0.8})
    add_edge_to_graph(graph, "Rule_1", "Rule_2", {"relation": "depends_on"})
    add_edge_to_graph(graph, "Rule_2", "Agent_X", {"relation": "applies_to"})
    add_edge_to_graph(graph, "Agent_X", "Goal_Z", {"relation": "achieves"})
    add_edge_to_graph(graph, "Axiom_A", "Goal_Z", {"relation": "mandates"})
    
    # Add a cyclic path for pathfinding tests
    add_node_to_graph(graph, "Cycle_Node_1", {"type": "cycle_test"})
    add_node_to_graph(graph, "Cycle_Node_2", {"type": "cycle_test"})
    add_edge_to_graph(graph, "Rule_2", "Cycle_Node_1", {"relation": "leads_to"})
    add_edge_to_graph(graph, "Cycle_Node_1", "Cycle_Node_2", {"relation": "leads_to"})
    add_edge_to_graph(graph, "Cycle_Node_2", "Rule_2", {"relation": "feeds_back"}) # Create cycle

    return graph

# --- Test Cases ---

def test_create_conceptual_graph(empty_graph):
    """Tests if an empty graph is created."""
    assert isinstance(empty_graph, nx.DiGraph)
    assert empty_graph.number_of_nodes() == 0
    assert empty_graph.number_of_edges() == 0
    logger.info("Test: create_conceptual_graph works.")

def test_add_node_to_graph(empty_graph):
    """Tests adding nodes and updating attributes."""
    add_node_to_graph(empty_graph, "NewNode", {"type": "test"})
    assert "NewNode" in empty_graph
    assert empty_graph.nodes["NewNode"]["type"] == "test"
    
    # Test updating attributes
    add_node_to_graph(empty_graph, "NewNode", {"description": "Updated"})
    assert empty_graph.nodes["NewNode"]["description"] == "Updated"
    logger.info("Test: add_node_to_graph works correctly.")

def test_add_edge_to_graph(empty_graph):
    """Tests adding edges and error handling for missing nodes."""
    add_node_to_graph(empty_graph, "Node1", {"type": "test"})
    add_node_to_graph(empty_graph, "Node2", {"type": "test"})
    add_edge_to_graph(empty_graph, "Node1", "Node2", {"relation": "connects"})
    assert empty_graph.has_edge("Node1", "Node2")
    assert empty_graph.edges["Node1", "Node2"]["relation"] == "connects"

    # Test error for missing source node
    with pytest.raises(ValueError, match="Source node 'MissingNode' not found."):
        add_edge_to_graph(empty_graph, "MissingNode", "Node1", {"relation": "invalid"})
    # Test error for missing target node
    with pytest.raises(ValueError, match="Target node 'MissingNode' not found."):
        add_edge_to_graph(empty_graph, "Node1", "MissingNode", {"relation": "invalid"})
    
    # Test updating edge attributes
    add_edge_to_graph(empty_graph, "Node1", "Node2", {"weight": 0.5})
    assert empty_graph.edges["Node1", "Node2"]["weight"] == 0.5
    logger.info("Test: add_edge_to_graph works correctly.")

def test_get_node_attributes(populated_graph):
    """Tests retrieving node attributes."""
    attrs = get_node_attributes(populated_graph, "Axiom_A")
    assert attrs is not None
    assert attrs["type"] == "axiom"
    assert get_node_attributes(populated_graph, "NonExistentNode") is None
    logger.info("Test: get_node_attributes works.")

def test_get_edge_attributes(populated_graph):
    """Tests retrieving edge attributes."""
    attrs = get_edge_attributes(populated_graph, "Axiom_A", "Rule_1")
    assert attrs is not None
    assert attrs["relation"] == "influences"
    assert get_edge_attributes(populated_graph, "Rule_1", "Rule_NonExistent") is None
    logger.info("Test: get_edge_attributes works.")

def test_find_paths_between_nodes(populated_graph):
    """Tests finding paths between nodes."""
    paths = find_paths_between_nodes(populated_graph, "Axiom_A", "Goal_Z")
    assert len(paths) == 2 # Axiom_A -> Rule_1 -> Rule_2 -> Agent_X -> Goal_Z, and Axiom_A -> Goal_Z
    assert ["Axiom_A", "Goal_Z"] in paths
    assert ["Axiom_A", "Rule_1", "Rule_2", "Agent_X", "Goal_Z"] in paths
    
    # Test max_length cutoff
    short_paths = find_paths_between_nodes(populated_graph, "Axiom_A", "Goal_Z", max_length=2)
    assert len(short_paths) == 1 # Only Axiom_A -> Goal_Z
    
    # Test non-existent path
    assert find_paths_between_nodes(populated_graph, "Agent_X", "Axiom_A") == [] # No reverse path
    logger.info("Test: find_paths_between_nodes works correctly.")

def test_get_subgraph_around_node(populated_graph):
    """Tests extracting a subgraph."""
    subgraph = get_subgraph_around_node(populated_graph, "Rule_2", radius=1)
    assert "Rule_2" in subgraph
    assert "Rule_1" in subgraph # Incoming edge
    assert "Agent_X" in subgraph # Outgoing edge
    assert "Cycle_Node_1" in subgraph # Outgoing edge from cycle test
    assert "Cycle_Node_2" in subgraph # Outgoing edge from cycle test (2 hops from rule 2 via cycle_node_1)
    assert "Axiom_A" not in subgraph # Too far (radius 1)
    assert subgraph.number_of_nodes() >= 4 # Rule_2, Rule_1, Agent_X, Cycle_Node_1, Cycle_Node_2
    assert subgraph.has_edge("Rule_1", "Rule_2")
    assert subgraph.has_edge("Rule_2", "Agent_X")
    assert subgraph.has_edge("Rule_2", "Cycle_Node_1")
    logger.info("Test: get_subgraph_around_node works.")


def test_convert_and_reconstruct_graph_json_serializable(populated_graph):
    """Tests converting to JSON serializable format and reconstructing."""
    serializable_data = convert_graph_to_json_serializable(populated_graph)
    
    assert isinstance(serializable_data, dict)
    assert 'nodes' in serializable_data
    assert 'edges' in serializable_data
    assert len(serializable_data['nodes']) == populated_graph.number_of_nodes()
    assert len(serializable_data['edges']) == populated_graph.number_of_edges()
    
    # Test reconstruction
    reconstructed_graph = reconstruct_graph_from_json_serializable(serializable_data)
    assert isinstance(reconstructed_graph, nx.DiGraph)
    assert nx.is_isomorphic(populated_graph, reconstructed_graph), "Reconstructed graph is not isomorphic!"
    
    # Check attributes preserved
    assert get_node_attributes(populated_graph, "Axiom_A") == get_node_attributes(reconstructed_graph, "Axiom_A")
    assert get_edge_attributes(populated_graph, "Axiom_A", "Rule_1") == get_edge_attributes(reconstructed_graph, "Axiom_A", "Rule_1")
    logger.info("Test: Graph serialization and reconstruction works and preserves data.")
