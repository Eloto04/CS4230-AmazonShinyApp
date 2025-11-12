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
import plotly.graph_objects as go
import networkx as nx
from infomap import Infomap

print("Loading Amazon graph from AmazonGraph.txt...")
G = nx.read_edgelist('AmazonGraph.txt', 
                        create_using=nx.DiGraph(), 
                        nodetype=int,
                        data=False)
print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Load product metadata
print("Loading product metadata...")
try:
    products = pd.read_csv('metadata.csv')
    print(f"Product columns: {products.columns.tolist()}")
    
    # Check if 'id' column exists, otherwise use first column
    if 'id' in products.columns:
        products = products.rename(columns={'id': 'node'})
    elif products.columns[0] != 'node':
        products = products.rename(columns={products.columns[0]: 'node'})
    
    # Keep node, title, group, total_review_count, and average_rating columns
    columns_to_keep = ['node']
    if 'title' in products.columns:
        columns_to_keep.append('title')
    if 'group' in products.columns:
        columns_to_keep.append('group')
    if 'total_review_count' in products.columns:
        columns_to_keep.append('total_review_count')
    if 'average_rating' in products.columns:
        columns_to_keep.append('average_rating')
    
    products = products[columns_to_keep]
    
    print(f"Product metadata loaded: {len(products)} products")
except Exception as e:
    print(f"Warning: Could not load product metadata: {e}")
    products = pd.DataFrame(columns=['node', 'title', 'group', 'total_review_count', 'average_rating'])
    
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

# Merge with product metadata to get product names and other info
nodes = nodes.merge(products, on='node', how='left')
# Fill missing values
nodes['title'] = nodes['title'].fillna('Unknown Product')
nodes['group'] = nodes['group'].fillna('Unknown')
nodes['total_review_count'] = nodes['total_review_count'].fillna(0)
nodes['average_rating'] = nodes['average_rating'].fillna(0.0)

print(f"Node dataframe created with {len(nodes)} nodes")
print("All data loaded and ready!")

# User interface (layout)
app_ui = ui.page_fluid(
    ui.h2("Amazon Graph — Interactive Dashboard"),
    # Sidebar for inputs and filters
    ui.layout_sidebar(
        ui.sidebar(
            ui.h4("Filters & Controls"),
            # Visualization type selection
            ui.input_radio_buttons(
                "viz_type",
                "Visualization:",
                choices={
                    "bar": "Bar Chart",
                    "network": "Network Graph"
                },
                selected="bar"
            ),
            # Metric selection
            ui.input_select(
                "metric",
                "Color / metric:",
                choices={
                    "degree": "Degree",
                    "pagerank": "PageRank",
                    "eigenvector": "Eigenvector Centrality",
                    "total_review_count": "Total Review Count"
                },
                selected="degree"
            ),
            # Community selection
            ui.input_select(
                "community",
                "Community:",
                choices=community_choices,
                selected="all"
            ),
            # Number of top nodes to show
            ui.input_numeric(
                "top_n",
                "Show top N nodes:",
                value=20,
                min=5,
                max=1000
            ),
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
                    "total_review_count": "Total Review Count"
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
        viz_type = input.viz_type()
        
        if viz_type == "bar":
            # Create a bar chart showing top nodes by selected metric (limit to 20 for readability)
            fig = px.bar(
                df.head(20),
                x='node',
                y=metric,
                color=metric,
                title=f"Top 20 nodes by {metric}",
                labels={'node': 'Node ID', metric: metric.capitalize()}
            )
            fig.update_xaxes(type='category')
        else:
            # Create an interactive network graph showing all filtered nodes
            # Get the subgraph containing all the filtered nodes
            top_nodes = df['node'].tolist()
            subgraph = G.subgraph(top_nodes).copy()
            
            # Use spring layout for positioning
            pos = nx.spring_layout(subgraph, k=0.5, iterations=50)
            
            # Create edge traces with arrows for directed edges
            edge_traces = []
            for edge in subgraph.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                
                # Create individual edge trace with arrow
                edge_trace = go.Scatter(
                    x=[x0, x1],
                    y=[y0, y1],
                    mode='lines',
                    line=dict(width=0.5, color='#888'),
                    hoverinfo='none',
                    showlegend=False
                )
                edge_traces.append(edge_trace)
            
            # Create annotations for arrows
            annotations = []
            for edge in subgraph.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                
                annotations.append(
                    dict(
                        ax=x0, ay=y0,
                        x=x1, y=y1,
                        xref='x', yref='y',
                        axref='x', ayref='y',
                        showarrow=True,
                        arrowhead=3,
                        arrowsize=2,
                        arrowwidth=2,
                        arrowcolor='#555',
                        opacity=0.8
                    )
                )
            
            # Create node trace
            node_x = []
            node_y = []
            node_text = []
            node_color = []
            
            for node in subgraph.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_info = df[df['node'] == node].iloc[0]
                product_name = node_info['title']
                
                # Format metric value based on type
                if metric == 'total_review_count':
                    metric_value = f"{int(node_info[metric])}"
                else:
                    metric_value = f"{node_info[metric]:.4f}"
                
                node_text.append(
                    f"{product_name}<br>"
                    f"Node ID: {node}<br>"
                    f"Group: {node_info['group']}<br>"
                    f"{metric}: {metric_value}<br>"
                    f"Avg Rating: {node_info['average_rating']:.2f}<br>"
                    f"Total Reviews: {int(node_info['total_review_count'])}<br>"
                    f"Community: {node_info['community']}"
                )
                node_color.append(node_info[metric])
            
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers',
                hoverinfo='text',
                text=node_text,
                marker=dict(
                    showscale=True,
                    colorscale='Viridis',
                    color=node_color,
                    size=10,
                    colorbar=dict(
                        thickness=15,
                        title=metric.capitalize(),
                        xanchor='left'
                    ),
                    line_width=2))
            
            fig = go.Figure(data=edge_traces + [node_trace],
                          layout=go.Layout(
                              title=f'Network Graph colored by {metric}',
                              showlegend=False,
                              hovermode='closest',
                              margin=dict(b=0, l=0, r=0, t=40),
                              annotations=annotations,
                              xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                              yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                          )
        
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
        result_df = df[["node", "title", "group", "community", "degree", "in_degree", "out_degree", 
                        "pagerank", "eigenvector", "total_review_count", "average_rating"]].copy()
        
        # Format numeric columns to avoid scientific notation
        result_df['pagerank'] = result_df['pagerank'].apply(lambda x: f"{x:.8f}")
        result_df['eigenvector'] = result_df['eigenvector'].apply(lambda x: f"{x:.8f}")
        result_df['average_rating'] = result_df['average_rating'].apply(lambda x: f"{x:.2f}")
        result_df['total_review_count'] = result_df['total_review_count'].apply(lambda x: f"{int(x)}")
        
        # Rename columns for display
        result_df = result_df.rename(columns={
            'title': 'Product Name',
            'group': 'Group',
            'total_review_count': 'Total Reviews',
            'average_rating': 'Avg Rating'
        })
        
        return result_df.reset_index(drop=True)


app = App(app_ui, server)
