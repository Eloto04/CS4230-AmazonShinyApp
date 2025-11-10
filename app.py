# app.py
from shiny import App, render, ui
import networkx as nx
import pandas as pd

# --- Load graph ---
G = nx.read_edgelist(
    "AmazonGraph.txt",
    create_using=nx.DiGraph(),
    nodetype=int,
    data=False
)
print("Graph loaded successfully!")

# --- Compute graph stats ---
numberOfNodes = G.number_of_nodes()
numberOfEdges = G.number_of_edges()
density = nx.density(G)

# --- User Interface ---
app_ui = ui.page_fluid(
    ui.h2("Amazon Product Graph Summary"),
    ui.card(
        ui.card_header("Graph Information"),
        ui.output_text("graph_info")
    ),
)

# --- Server Logic ---
def server(input, output, session):
    @output
    @render.text
    def graph_info():
        return (
            f"Number of Nodes: {numberOfNodes:,}\n"
            f"Number of Edges: {numberOfEdges:,}\n"
            f"Graph Density: {density:.8f}"
        )

# --- Create and run app ---
app = App(app_ui, server)

