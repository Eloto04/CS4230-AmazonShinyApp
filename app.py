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

# Load graph at startup (fast operation)
print("Loading Amazon graph from AmazonGraph.txt...")
G = nx.read_edgelist('AmazonGraph.txt', 
                        create_using=nx.DiGraph(), 
                        nodetype=int,
                        data=False)
print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Load product metadata at startup (fast operation)
print("Loading product metadata...")
try:
    products = pd.read_csv('metadata.csv')
    
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

print("Website ready! Heavy computations will run in background...")

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
                "Color / Metric:",
                choices={
                    "degree": "Degree Centrality",
                    "pagerank": "PageRank",
                    "eigenvector": "Eigenvector Centrality",
                    "total_review_count": "Total Review Count"
                },
                selected="degree"
            ),
            # Community selection (will be updated after metrics load)
            ui.input_select(
                "community",
                "Community:",
                choices={"all": "All communities"},
                selected="all"
            ),
            # Number of top nodes to show
            ui.input_numeric(
                "top_n",
                "Show Top N Nodes:",
                value=20,
                min=5,
                max=1000
            ),
        ),
        # Main area with graph + summary + table
        ui.row(
            ui.column(
                8,
                ui.output_ui("graph_ui"),
                ui.h4("Results Table", style="margin-top: 20px;"),
                ui.output_data_frame("results_table"),
            ),
            ui.column(
                4,
                ui.panel_well(
                    ui.h4("Summary Metrics"),
                    ui.output_ui("summary_ui"),
                ),
                ui.panel_well(
                    ui.h4("Category Distribution in Shown Nodes"),
                    ui.output_ui("category_pie_chart"),
                ),
                ui.panel_well(
                    ui.h4("Word Distribution in Shown Nodes (Frequency > 1)"),
                    ui.output_ui("common_words_list"),
                ),
            ),
        ),
    ),
)


def server(input, output, session):
    # Reactive value to store selected node
    selected_node = reactive.Value(None)
    
    # Cache for graph layout positions
    layout_cache = {}
    
    # Use reactive.calc to compute metrics once and cache them
    @reactive.calc
    def load_data():
        print("Computing metrics in background...")
        
        # Run Infomap community detection
        print("Running Infomap community detection...")
        im = Infomap(directed=True, silent=True)
        for u, v in G.edges():
            im.add_link(u, v)
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
        
        # Build community choices for dropdown
        community_list = [(comm_id, len(nodes)) for comm_id, nodes in communities.items()]
        community_list.sort(key=lambda x: x[0])
        community_choices = {"all": "All communities"}
        for comm_id, size in community_list:
            community_choices[str(comm_id)] = f"Community {comm_id} ({size} nodes)"
        
        # Calculate metrics
        print("Calculating PageRank...")
        pagerank = nx.pagerank(G, alpha=0.85, max_iter=100)
        
        print("Calculating Eigenvector Centrality...")
        eigenvector = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-6)
        
        # Create dataframe
        degree_dict = dict(G.degree())
        nodes = pd.DataFrame({
            'node': list(G.nodes()),
            'degree': [degree_dict[n] for n in G.nodes()],
            'in_degree': [G.in_degree(n) for n in G.nodes()],
            'out_degree': [G.out_degree(n) for n in G.nodes()],
            'community': [node_to_community.get(n, -1) for n in G.nodes()],
            'pagerank': [pagerank.get(n, 0.0) for n in G.nodes()],
            'eigenvector': [eigenvector.get(n, 0.0) for n in G.nodes()],
        })
        
        # Merge with product metadata
        nodes = nodes.merge(products, on='node', how='left')
        nodes['title'] = nodes['title'].fillna('Unknown Product')
        nodes['group'] = nodes['group'].fillna('Unknown')
        nodes['total_review_count'] = nodes['total_review_count'].fillna(0)
        nodes['average_rating'] = nodes['average_rating'].fillna(0.0)
        
        print("All metrics computed!")
        
        # Update the community dropdown
        ui.update_select(
            "community",
            choices=community_choices,
            selected="all"
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
        try:
            # Call get_filtered_df first to establish reactive dependency (same as table)
            df = get_filtered_df()
        except:
            # Show loading message while metrics are being computed
            return ui.HTML("""
                <div style="height: 400px; display: flex; align-items: center; justify-content: center; 
                            border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9;">
                    <p style="color: #666; font-size: 16px;">Computing metrics... This may take a minute.</p>
                </div>
            """)
        
        metric = input.metric()
        viz_type = input.viz_type()
        top_n = int(input.top_n() or 20)
        
        if viz_type == "bar":
            # Create a bar chart showing top nodes by selected metric
            # Add highlighting for selected node
            selected = selected_node()
            colors = ['red' if node == selected else 'blue' for node in df['node']]
            
            fig = go.Figure(data=[
                go.Bar(
                    x=df['node'],
                    y=df[metric],
                    marker=dict(
                        color=df[metric],
                        colorscale='Viridis',
                        line=dict(
                            color=colors,
                            width=3
                        )
                    ),
                    text=df['node'],
                    hovertemplate='Node: %{text}<br>' + metric + ': %{y}<extra></extra>'
                )
            ])
            fig.update_layout(
                title=f"Top {top_n} nodes by {metric}",
                xaxis_title='Node ID',
                yaxis_title=metric.capitalize()
            )
            fig.update_xaxes(type='category')
        else:
            # Create an interactive network graph showing all filtered nodes
            # Get the subgraph containing all the filtered nodes
            top_nodes = df['node'].tolist()
            subgraph = G.subgraph(top_nodes).copy()
            
            # Create cache key based on the nodes in this subgraph
            cache_key = tuple(sorted(top_nodes))
            
            # Use cached layout if available, otherwise compute and cache it
            if cache_key not in layout_cache:
                layout_cache[cache_key] = nx.spring_layout(subgraph, k=0.5, iterations=50, seed=42)
            pos = layout_cache[cache_key]
            
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
            node_size = []
            
            selected = selected_node()
            
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
                
                # Highlight selected node
                if node == selected:
                    node_size.append(30)
                else:
                    node_size.append(15)
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
                    size=node_size,
                    colorbar=dict(
                        thickness=15,
                        title=metric.capitalize(),
                        xanchor='left'
                    ),
                    line=dict(
                        width=2,
                        color=['red' if n == selected else 'white' for n in subgraph.nodes()]
                    )
                )
            )
            
            fig = go.Figure(data=edge_traces + [node_trace],
                          layout=go.Layout(
                              title=f'Network Graph colored by {metric}. Click/drag to zoom the graph.',
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
        try:
            all_nodes = load_data()
            df = get_filtered_df()
        except:
            return ui.HTML("<p>Loading...</p>")
        
        # === ENTIRE GRAPH METRICS ===
        total_nodes = G.number_of_nodes()
        total_edges = G.number_of_edges()
        entire_avg_degree = 2 * total_edges / total_nodes if total_nodes > 0 else 0.0
        
        # Calculate average in/out degree for entire graph
        entire_avg_in = float(all_nodes["in_degree"].mean()) if len(all_nodes) > 0 else 0.0
        entire_avg_out = float(all_nodes["out_degree"].mean()) if len(all_nodes) > 0 else 0.0
        
        entire_density = nx.density(G)
        
        if entire_density < 0.1:
            entire_density_label = "Low"
        elif entire_density < 0.3:
            entire_density_label = "Medium"
        else:
            entire_density_label = "High"
        
        # Calculate modularity (using community structure from all_nodes)
        if len(all_nodes) > 0 and 'community' in all_nodes.columns:
            communities_dict = {}
            for _, row in all_nodes.iterrows():
                comm_id = row['community']
                if comm_id not in communities_dict:
                    communities_dict[comm_id] = []
                communities_dict[comm_id].append(row['node'])
            communities_list = list(communities_dict.values())
            modularity = nx.algorithms.community.modularity(G, communities_list)
            
            # Classify modularity: low < 0.3, medium 0.3-0.7, high > 0.7
            if modularity < 0.3:
                modularity_label = "Low"
            elif modularity < 0.7:
                modularity_label = "Medium"
            else:
                modularity_label = "High"
        else:
            modularity = 0.0
            modularity_label = "N/A"
        
        # === SELECTED COMMUNITY METRICS ===
        community_selected = input.community()
        if community_selected == "all":
            community_section = """
        <div style="background-color: #f0f8ff; padding: 10px; margin: 10px 0; border-radius: 5px;">
          <h5 style="margin-top: 0;">Selected Community</h5>
          <p style="color: #666; font-style: italic;">Select a community to show community metrics</p>
        </div>"""
        else:
            community_df = all_nodes[all_nodes['community'] == int(community_selected)]
            community_nodes_count = len(community_df)
            community_avg_degree = float(community_df["degree"].mean()) if community_nodes_count > 0 else 0.0
            community_avg_in = float(community_df["in_degree"].mean()) if community_nodes_count > 0 else 0.0
            community_avg_out = float(community_df["out_degree"].mean()) if community_nodes_count > 0 else 0.0
            
            # Create subgraph for this community
            community_node_list = community_df['node'].tolist()
            community_subgraph = G.subgraph(community_node_list)
            
            # Calculate density for community
            if len(community_node_list) > 1:
                community_density = nx.density(community_subgraph)
                if community_density < 0.1:
                    community_density_label = "Low"
                elif community_density < 0.3:
                    community_density_label = "Medium"
                else:
                    community_density_label = "High"
            else:
                community_density = 0.0
                community_density_label = "N/A"
            
            # Calculate clustering for community
            try:
                community_clustering = nx.average_clustering(community_subgraph.to_undirected())
                if community_clustering < 0.3:
                    community_clustering_label = "Low"
                elif community_clustering < 0.6:
                    community_clustering_label = "Medium"
                else:
                    community_clustering_label = "High"
            except:
                community_clustering = 0.0
                community_clustering_label = "N/A"
            
            # Calculate reciprocity for community
            community_edges = community_subgraph.number_of_edges()
            if community_edges > 0:
                reciprocal_edges = sum(1 for u, v in community_subgraph.edges() if community_subgraph.has_edge(v, u)) / 2
                community_reciprocity = (reciprocal_edges / community_edges) * 100
            else:
                community_reciprocity = 0.0
            
            # Top products in this community by different metrics
            community_top_products = ""
            if len(community_df) > 0:
                # Top by Degree
                if 'degree' in community_df.columns:
                    top_degree_idx = community_df['degree'].idxmax()
                    top_degree_product = community_df.loc[top_degree_idx, 'title']
                    top_degree_value = community_df.loc[top_degree_idx, 'degree']
                    community_top_products += f"""
          <p><strong>Top by Degree:</strong> {top_degree_product[:35]}{'...' if len(str(top_degree_product)) > 35 else ''} ({top_degree_value})</p>"""
                
                # Top by PageRank
                if 'pagerank' in community_df.columns:
                    top_pagerank_idx = community_df['pagerank'].idxmax()
                    top_pagerank_product = community_df.loc[top_pagerank_idx, 'title']
                    top_pagerank_value = community_df.loc[top_pagerank_idx, 'pagerank']
                    community_top_products += f"""
          <p><strong>Top by PageRank:</strong> {top_pagerank_product[:35]}{'...' if len(str(top_pagerank_product)) > 35 else ''} ({top_pagerank_value:.6f})</p>"""
                
                # Top by Eigenvector
                if 'eigenvector' in community_df.columns:
                    top_eigen_idx = community_df['eigenvector'].idxmax()
                    top_eigen_product = community_df.loc[top_eigen_idx, 'title']
                    top_eigen_value = community_df.loc[top_eigen_idx, 'eigenvector']
                    community_top_products += f"""
          <p><strong>Top by Eigenvector:</strong> {top_eigen_product[:35]}{'...' if len(str(top_eigen_product)) > 35 else ''} ({top_eigen_value:.6f})</p>"""
            
            community_section = f"""
        <div style="background-color: #f0f8ff; padding: 10px; margin: 10px 0; border-radius: 5px;">
          <h5 style="margin-top: 0;">Selected Community</h5>
          <p><strong>Nodes:</strong> {community_nodes_count:,}</p>
          <p><strong>Avg degree:</strong> {community_avg_degree:.2f} (in: {community_avg_in:.2f}, out: {community_avg_out:.2f})</p>
          <p><strong>Density:</strong> {community_density:.4f} <span style="color: #666;">({community_density_label})</span></p>
          <p><strong>Avg clustering:</strong> {community_clustering:.4f} <span style="color: #666;">({community_clustering_label})</span></p>
          <p><strong>Reciprocity:</strong> {community_reciprocity:.1f}% bidirectional</p>{community_top_products}
        </div>"""
        
        html = f"""
        <div style="font-size: 13px;">
          <div style="background-color: #fff3cd; padding: 10px; margin-bottom: 10px; border-radius: 5px;">
            <h5 style="margin-top: 0;">Entire Graph</h5>
            <p><strong>Nodes:</strong> {total_nodes:,}</p>
            <p><strong>Edges:</strong> {total_edges:,}</p>
            <p><strong>Avg degree:</strong> {entire_avg_degree:.2f} (in: {entire_avg_in:.2f}, out: {entire_avg_out:.2f})</p>
            <p><strong>Density:</strong> {entire_density:.6f} <span style="color: #666;">({entire_density_label})</span></p>
            <p><strong>Modularity:</strong> {modularity:.4f} <span style="color: #666;">({modularity_label})</span></p>
          </div>
          {community_section}
          <p style="margin-top: 10px; font-size: 12px;"></p>
        </div>
        """
        return ui.HTML(html)
    
    # Category pie chart
    @output
    @render.ui
    def category_pie_chart():
        try:
            df = get_filtered_df()
        except:
            return ui.HTML("<p>Loading...</p>")
        
        if len(df) > 0 and 'group' in df.columns:
            category_counts = df['group'].value_counts().head(10)  # Top 10 categories
            
            fig = go.Figure(data=[go.Pie(
                labels=category_counts.index,
                values=category_counts.values,
                textinfo='label+percent',
                hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percent: %{percent}<extra></extra>'
            )])
            
            fig.update_layout(
                height=350,
                margin=dict(l=0, r=0, t=20, b=0),
                showlegend=False
            )
            
            html_fragment = fig.to_html(include_plotlyjs="cdn", full_html=False)
            return ui.HTML(html_fragment)
        else:
            return ui.HTML("<p>No category data available</p>")
    
    # Common words list
    @output
    @render.ui
    def common_words_list():
        try:
            df = get_filtered_df()
        except:
            return ui.HTML("<p>Loading...</p>")
        
        if len(df) > 0 and 'title' in df.columns:
            # Extract all words from titles (lowercase, filter out common stop words)
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                         'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'be', 'been',
                         'unknown', 'product', '-', 'that', 'it', 'this', 'not', 'have', 'has',
                         'will', 'can', 'all', 'up', 'out', 'so', 'no', 'if', 'my', 'me', 'about'}
            all_words = []
            for title in df['title'].dropna():
                words = str(title).lower().split()
                # Filter out stop words and short words
                filtered_words = [w.strip('.,!?()[]{}:;"\'-') for w in words 
                                 if len(w) > 2 and w.lower() not in stop_words]
                all_words.extend(filtered_words)
            
            if all_words:
                from collections import Counter
                word_counts = Counter(all_words)
                # Only include words that appear more than once
                common_words = [(word, count) for word, count in word_counts.most_common(50) if count > 1]
                
                if common_words:
                    words_html = "<ul style='max-height: 300px; overflow-y: auto; padding-left: 20px;'>"
                    for word, count in common_words:
                        words_html += f"<li>{word} <span style='color: #666;'>({count})</span></li>"
                    words_html += "</ul>"
                    return ui.HTML(words_html)
                else:
                    return ui.HTML("<p>No repeated words found</p>")
            else:
                return ui.HTML("<p>No words to analyze</p>")
        else:
            return ui.HTML("<p>No title data available</p>")

    @output
    @render.data_frame
    def results_table():
        try:
            df = get_filtered_df()
        except:
            # Return empty dataframe with message while loading
            return render.DataGrid(pd.DataFrame({"Status": ["Computing metrics... please wait"]}))
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
        
        return render.DataGrid(result_df.reset_index(drop=True), selection_mode="row", height="600px")
    
    # Observer to track table selection and update selected_node
    @reactive.effect
    def _():
        selection = results_table.cell_selection()
        if selection and "rows" in selection and len(selection["rows"]) > 0:
            try:
                df = get_filtered_df()
                selected_row = list(selection["rows"])[0]
                if selected_row < len(df):
                    node_id = df.iloc[selected_row]['node']
                    selected_node.set(node_id)
            except:
                pass


app = App(app_ui, server)
