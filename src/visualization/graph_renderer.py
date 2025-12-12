# ontological-playground-designer/src/visualization/graph_renderer.py

import networkx as nx
from pyvis.network import Network
from typing import Dict, Any, Optional, List
import os

# Ensure loguru is set up for structured logging
from loguru import logger
from src.utils.logger import setup_logging

# Import data models for clarity
from src.core.axiom_parser import ParsedAxiom
from src.core.rule_generator import GeneratedRule

# Setup logging for this module
setup_logging()

class GraphRenderer:
    """
    Renders networkx graph objects into interactive HTML visualizations.
    This helps users understand the complex conceptual relationships within
    AI-designed worlds, providing critical Causal Explainability.
    """
    def __init__(self, output_dir: str = "data/visualizations"):
        """
        Initializes the GraphRenderer.

        Args:
            output_dir (str): Directory where generated HTML visualizations will be saved.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"GraphRenderer initialized. Output directory: {self.output_dir}")

    def _get_node_color(self, node_type: str) -> str:
        """Assigns a color based on node type for better visualization."""
        colors = {
            "axiom": "#FFD700",         # Gold for foundational principles
            "rule": "#87CEEB",          # Sky Blue for generated rules
            "agent_behavior": "#98FB98", # Pale Green for agent rules
            "environmental_law": "#3CB371", # Medium Sea Green for environment rules
            "resource_mechanic": "#DAA520", # Goldenrod for resource mechanics
            "social_dynamic": "#FF6347", # Tomato for social dynamics
            "system_mechanic": "#BA55D3", # Medium Orchid for system mechanics
            "meta_rule": "#FF4500",     # Orange Red for meta-rules
            "entity": "#D3D3D3",        # Light Gray for generic entities
            "paradox": "#DC143C",       # Crimson for detected paradoxes
            "potential_conflict": "#FF8C00", # Dark Orange for warnings
            "unknown": "#A9A9A9"        # Dark Gray for unclassified nodes
        }
        return colors.get(node_type, colors["unknown"])

    def _get_node_title(self, attributes: Dict[str, Any]) -> str:
        """Generates an HTML tooltip title for a node with its key attributes."""
        title = f"<b>ID:</b> {attributes.get('id', 'N/A')}<br>"
        if 'type' in attributes:
            title += f"<b>Type:</b> {attributes['type'].replace('_', ' ').title()}<br>"
        if 'description' in attributes:
            title += f"<b>Description:</b> {attributes['description']}<br>"
        if 'priority' in attributes:
            title += f"<b>Priority:</b> {attributes['priority']}<br>"
        if 'severity' in attributes:
            title += f"<b>Severity:</b> {attributes['severity']:.2f}<br>"
        if 'axiom_influence' in attributes:
            influence_str = ", ".join([f"{k}:{v:.2f}" for k, v in attributes['axiom_influence'].items()])
            title += f"<b>Axiom Influence:</b> {influence_str}<br>"
        return title

    def _add_nodes_and_edges_to_network(self, net: Network, graph: nx.DiGraph):
        """Helper to add nodes and edges from a networkx graph to a pyvis Network object."""
        for node_id, attributes in graph.nodes(data=True):
            node_type = attributes.get('type', 'unknown')
            net.add_node(
                n_id=node_id,
                label=node_id,
                color=self._get_node_color(node_type),
                title=self._get_node_title(attributes),
                size=attributes.get('size', 15), # Can adjust size based on importance/degree
                font={'color': 'black' if node_type not in ["axiom"] else 'white'}, # Adjust font color
                # Add physics properties for better layout
                physics=True
            )
        
        for u, v, attributes in graph.edges(data=True):
            relation = attributes.get('relation', 'connects')
            net.add_edge(
                source=u,
                to=v,
                title=relation.replace('_', ' ').title(), # Tooltip for edge
                label=relation.replace('_', ' ').title() if len(relation) < 20 else '', # Shorter label for display
                color=attributes.get('color', '#888888'),
                width=attributes.get('weight', 1) * 0.5, # Adjust thickness based on weight
                arrows='to' # Directed graph
            )

    def render_conceptual_graph(self, 
                                graph: nx.DiGraph, 
                                filename: str, 
                                title: str = "Conceptual Graph Visualization",
                                show_buttons: bool = True,
                                notebook: bool = False # For rendering in notebooks vs standalone HTML
                                ) -> str:
        """
        Renders a networkx graph into an interactive HTML visualization using Pyvis.

        Args:
            graph (nx.DiGraph): The networkx graph to render.
            filename (str): The name of the output HTML file (e.g., "world_rules").
            title (str): Title to display at the top of the HTML page.
            show_buttons (bool): If True, shows Pyvis configuration buttons in the HTML.
            notebook (bool): If True, renders to a temporary HTML for Jupyter notebooks.

        Returns:
            str: The full path to the generated HTML file.
        """
        if graph.number_of_nodes() == 0:
            logger.warning("Attempted to render an empty graph. Skipping.")
            return ""

        html_filename = f"{filename}.html"
        output_path = os.path.join(self.output_dir, html_filename)

        # Initialize Pyvis Network
        net = Network(
            height="750px", width="100%", bgcolor="#222222", font_color="white",
            notebook=notebook, cdn_resources='remote' # Use remote CDN for wider compatibility
        )
        net.repulsion(
            node_distance=150, # How far nodes repel each other
            central_gravity=0.05, # Pulls nodes towards the center
            spring_length=100, # Ideal length of springs connecting nodes
            spring_strength=0.05, # How stiff the springs are
            damping=0.9 # Reduces oscillation
        )
        net.set_options("""
        var options = {
          "physics": {
            "barnesHut": {
              "gravitationalConstant": -8000,
              "centralGravity": 0.3,
              "springLength": 90,
              "springConstant": 0.04,
              "damping": 0.09,
              "avoidOverlap": 0.5
            },
            "minVelocity": 0.75,
            "solver": "barnesHut"
          }
        }
        """)

        # Add nodes and edges
        self._add_nodes_and_edges_to_network(net, graph)

        # Set title in HTML
        net.heading = f"<h1>{title}</h1>"

        # Add buttons for physics configuration if requested
        if show_buttons:
            net.show_buttons(filter_=['physics', 'layout']) # Filter to show only relevant options

        # Generate HTML file
        try:
            net.save_graph(output_path)
            logger.success(f"Interactive graph saved to: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to save graph to {output_path}: {e}", exc_info=True)
            raise

# --- Example Usage (for testing and demonstration) ---
if __name__ == "__main__":
    import os
    # Ensure src/utils directory and logger.py exist for setup_logging
    if not os.path.exists("src/utils"):
        os.makedirs("src/utils")
        # Assuming logger.py is already there or will be created next

    # Create dummy output directory if it doesn't exist
    if not os.path.exists("data/visualizations"):
        os.makedirs("data/visualizations")
    
    logger.info("--- Demonstrating GraphRenderer ---")

    # 1. Create a sample graph (similar to what RuleGenerator or ParadoxDetector might output)
    sample_graph = nx.DiGraph()

    # Add Axiom Nodes (Gold)
    sample_graph.add_node("Axiom_Flourishing", id="Axiom_Flourishing", type="axiom", 
                           description="Maximize well-being and adaptive capacity.", priority=1, size=20)
    sample_graph.add_node("Axiom_Sustainability", id="Axiom_Sustainability", type="axiom", 
                           description="Ensure sustainable resource consumption.", priority=2, size=18)

    # Add Rule Nodes (Sky Blue for generic, specific greens/reds for types)
    sample_graph.add_node("Rule_Cooperation", id="Rule_Cooperation", type="agent_behavior", 
                           description="Agents gain well-being from cooperative actions.", parameters={"reward": 0.5})
    sample_graph.add_node("Rule_ResourceRegen", id="Rule_ResourceRegen", type="environmental_law", 
                           description="Resource regeneration tied to ecological health.", parameters={"rate": 0.02})
    sample_graph.add_node("Rule_InequalityPenalty", id="Rule_InequalityPenalty", type="social_dynamic", 
                           description="Agents face penalties if resource inequality is too high.", parameters={"threshold": 0.8})
    sample_graph.add_node("Rule_AdaptiveMutation", id="Rule_AdaptiveMutation", type="system_mechanic", 
                           description="System mutates rules to increase resilience.", parameters={"frequency": 0.1})

    # Add Entity/Conceptual Nodes (Light Gray)
    sample_graph.add_node("Agent_Entity", id="Agent_Entity", type="entity", description="A generic agent in the simulation.")
    sample_graph.add_node("Resource_Entity", id="Resource_Entity", type="entity", description="A renewable resource unit.")

    # Add a Paradox Node (Crimson)
    sample_graph.add_node("Paradox_EthicalConflict", id="Paradox_EthicalConflict", type="paradox", 
                           description="Conflict: agent autonomy vs. collective good.", severity=0.75, involved_rules=["Rule_Cooperation", "Rule_InequalityPenalty"])

    # Add Edges
    sample_graph.add_edge("Axiom_Flourishing", "Rule_Cooperation", relation="influences", weight=1.5, color="#00FF00")
    sample_graph.add_edge("Axiom_Sustainability", "Rule_ResourceRegen", relation="governs", weight=1.2)
    sample_graph.add_edge("Rule_Cooperation", "Agent_Entity", relation="applies_to")
    sample_graph.add_edge("Rule_ResourceRegen", "Resource_Entity", relation="modifies")
    sample_graph.add_edge("Rule_InequalityPenalty", "Agent_Entity", relation="penalizes")
    sample_graph.add_edge("Rule_AdaptiveMutation", "Rule_Cooperation", relation="optimizes")
    sample_graph.add_edge("Rule_AdaptiveMutation", "Rule_ResourceRegen", relation="optimizes")
    
    # Edge showing influence from rules to paradox
    sample_graph.add_edge("Rule_Cooperation", "Paradox_EthicalConflict", relation="contributes_to", color="#FF8C00")
    sample_graph.add_edge("Rule_InequalityPenalty", "Paradox_EthicalConflict", relation="contributes_to", color="#FF8C00")


    # 2. Render the graph
    renderer = GraphRenderer()
    output_html_path = renderer.render_conceptual_graph(
        sample_graph,
        filename="sample_world_design",
        title="Sample World Design - Conceptual Graph",
        show_buttons=True
    )

    logger.info(f"\n[bold green]Interactive graph successfully rendered to: {output_html_path}[/bold green]")
    logger.info(f"Open '{output_html_path}' in your web browser to view the visualization.")
