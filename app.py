# app.py
"""
Shiny dashboard for the Amazon graph project.

This file defines a sidebar + main layout with:
- controls for metric/community selection
- an interactive bar chart showing top nodes
- a dynamic summary panel
- a results table
"""
from shiny import App, ui, render, reactive
import pandas as pd
import plotly.express as px
import networkx as nx
from infomap import Infomap

# User interface (layout)
app_ui = ui.page_fluid(
    ui.h2("Amazon Graph — Interactive Dashboard"),
    # Sidebar for inputs and filters
    ui.layout_sidebar(
        ui.sidebar(
            ui.h4("Filters & Controls"),
            # Metric selection
            ui.input_select(
                "metric",
                "Color / metric:",
                {}
            ),
            # Community selection
            ui.input_select(
                "community",
                "Community:",
                {}
            ),
            # Number of top nodes to show
            ui.input_slider("top_n", "Show top N nodes:", min=5, max=50, value=20),
        ),
        # Main area with graph + summary + table
        ui.row(
            ui.column(8, ui.output_ui("graph_ui")),
            ui.column(
                4,
                ui.panel_well(
                    ui.h4("Summary metrics"),
                    ui.output_ui("summary_ui"),
                ),
            ),
        ),
        ui.h4("Results table"),
        ui.output_data_frame("results_table"),
    ),
)


def server(input, output, session):
    # Use reactive.calc to load data once and cache it
    @reactive.calc
    def load_data():
        print("Loading Amazon graph from AmazonGraph.txt...")
        G = nx.read_edgelist('AmazonGraph.txt', 
                             create_using=nx.DiGraph(), 
                             nodetype=int,
                             data=False)
        print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        # Run Infomap community detection
        print("Running Infomap community detection...")
        im = Infomap(directed=True, silent=True)
        
        # Add all edges to Infomap
        for u, v in G.edges():
            im.add_link(u, v)
        
        # Run the algorithm
        im.run()
        
        # Extract communities
        communities = {}
        node_to_community = {}
        
        for node in im.nodes:
            node_id = node.node_id
            module_id = node.module_id
            if module_id not in communities:
                communities[module_id] = []
            communities[module_id].append(node_id)
            node_to_community[node_id] = module_id
        
        print(f"Found {len(communities)} communities")
        
        # Sort communities by Infomap ID for the dropdown
        community_list = [(comm_id, len(nodes)) for comm_id, nodes in communities.items()]
        community_list.sort(key=lambda x: x[0])  # Sort by community ID
        
        # Build community choices for dropdown
        community_choices = {"all": "All communities"}
        for comm_id, size in community_list:
            community_choices[str(comm_id)] = f"Community {comm_id} ({size} nodes)"
        
        # Calculate basic metrics for all nodes
        print("Calculating node metrics...")
        degree_dict = dict(G.degree())
        
        # Calculate PageRank
        print("Calculating PageRank...")
        pagerank = nx.pagerank(G, alpha=0.85, max_iter=100)
        
        # Calculate Eigenvector Centrality
        print("Calculating Eigenvector Centrality...")
        eigenvector = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-6)
        
        # Create a dataframe with node information
        nodes = pd.DataFrame({
            'node': list(G.nodes()),
            'degree': [degree_dict[n] for n in G.nodes()],
            'in_degree': [G.in_degree(n) for n in G.nodes()],
            'out_degree': [G.out_degree(n) for n in G.nodes()],
            'community': [node_to_community.get(n, -1) for n in G.nodes()],
            'pagerank': [pagerank.get(n, 0.0) for n in G.nodes()],
            'eigenvector': [eigenvector.get(n, 0.0) for n in G.nodes()],
        })
        
        print(f"Node dataframe created with {len(nodes)} nodes")
        print("All data loaded and ready!")

        # Update the community dropdown with real communities
        ui.update_select(
            "community",
            choices=community_choices,
            selected="all"
        )
        
        ui.update_select(
                "metric",
                choices={
                    "degree": "Degree",
                    "pagerank": "PageRank",
                    "eigenvector": "Eigenvector Centrality",
                }
        )

        return nodes

    # Reactive helper: build a filtered dataframe based on inputs
    def get_filtered_df():
        # Call load_data() to get the dataframe (cached after first call)
        nodes = load_data()
        
        df = nodes.copy()
        comm = input.community()
        if comm and comm != "all":
            # Use the community ID directly
            df = df[df["community"] == int(comm)]
        # sort by chosen metric and take top N
        metric = input.metric()
        top_n = int(input.top_n() or 20)
        df = df.sort_values(metric, ascending=False).head(top_n)
        return df

    # Graph UI: render a Plotly scatter and embed as HTML into the Shiny UI
    @output
    @render.ui
    def graph_ui():
        # Call get_filtered_df first to establish reactive dependency (same as table)
        df = get_filtered_df()
        
        metric = input.metric()
        
        # Create a simple bar chart showing top nodes by selected metric
        fig = px.bar(
            df.head(20),
            x='node',
            y=metric,
            color=metric,
            title=f"Top nodes by {metric}",
            labels={'node': 'Node ID', metric: metric.capitalize()}
        )
        fig.update_xaxes(type='category')
        
        # Plotly figure converted to HTML fragment; Shiny will render it in the UI.
        html_fragment = fig.to_html(include_plotlyjs="cdn", full_html=False)
        return ui.HTML(html_fragment)

    # Summary UI: small metric cards (simple HTML) showing top numbers
    @output
    @render.ui
    def summary_ui():
        df = get_filtered_df()
        n_nodes = len(df)
        avg_degree = float(df["degree"].mean()) if n_nodes else 0.0
        html = f"""
        <div>
          <p><strong>Shown nodes:</strong> {n_nodes}</p>
          <p><strong>Avg degree (shown):</strong> {avg_degree:.2f}</p>
          <p><em>Click/drag to zoom the graph. Use controls on the left to change metric/community.</em></p>
        </div>
        """
        return ui.HTML(html)

    @output
    @render.data_frame
    def results_table():
        df = get_filtered_df()
        result_df = df[["node", "community", "degree", "in_degree", "out_degree", "pagerank", "eigenvector"]].copy()
        
        # Format numeric columns to avoid scientific notation
        result_df['pagerank'] = result_df['pagerank'].apply(lambda x: f"{x:.8f}")
        result_df['eigenvector'] = result_df['eigenvector'].apply(lambda x: f"{x:.8f}")
        
        return result_df.reset_index(drop=True)


app = App(app_ui, server)
