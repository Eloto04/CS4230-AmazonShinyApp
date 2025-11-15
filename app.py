# app.py
"""
Interactive Shiny Dashboard for Amazon Product Co-Purchase Network Analysis

This application provides a comprehensive visualization and analysis interface for exploring
the Amazon co-purchase network, featuring:
- Interactive visualizations (bar charts, network graphs, and metric scatter plots)
- Community detection using the Infomap algorithm
- Centrality metrics (In-Degree, PageRank, Eigenvector Centrality)
- Dynamic filtering by community and top N nodes
- Comprehensive network statistics and summaries
"""
from shiny import App, ui, render, reactive
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from infomap import Infomap

# ==================== DATA LOADING ====================
# Load the graph structure at startup for fast initial page rendering
print("Loading Amazon graph from AmazonGraph.txt...")
G = nx.read_edgelist('AmazonGraph.txt', 
                        create_using=nx.DiGraph(), 
                        nodetype=int,
                        data=False)
print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Load product metadata (titles, categories, reviews) at startup
print("Loading product metadata...")
try:
    products = pd.read_csv('metadata.csv')
    
    # Normalize column names: ensure the primary key is named 'node'
    if 'id' in products.columns:
        products = products.rename(columns={'id': 'node'})
    elif products.columns[0] != 'node':
        products = products.rename(columns={products.columns[0]: 'node'})
    
    # Select relevant columns for the dashboard
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

# ==================== USER INTERFACE ====================
# Define the application layout with sidebar controls and main content area
app_ui = ui.page_fluid(
    ui.h2("Amazon Graph — Interactive Dashboard"),
    # Sidebar: User controls for filtering and visualization options
    ui.layout_sidebar(
        ui.sidebar(
            ui.h4("Filters & Controls"),
            # Select visualization type: Bar Chart, Network Graph, or Experimental Plot
            ui.input_radio_buttons(
                "viz_type",
                "Visualization:",
                choices={
                    "bar": "Bar Chart",
                    "network": "Network Graph",
                    "metric": "Experimental Plot"
                },
                selected="bar"
            ),
            # Select which centrality metric to analyze and visualize
            ui.input_select(
                "metric",
                "Metric:",
                choices={
                    "in_degree": "In-Degree Centrality",
                    "pagerank": "PageRank",
                    "eigenvector": "Eigenvector Centrality",
                    "total_review_count": "Total Review Count"
                },
                selected="in_degree"
            ),
            # Community filter (dynamically populated after Infomap completes)
            ui.input_select(
                "community",
                "Community:",
                choices={"all": "All communities"},
                selected="all"
            ),
            # Set the number of top-ranked nodes to display
            ui.input_numeric(
                "top_n",
                "Show Top N Nodes:",
                value=20,
                min=5,
                max=1000
            ),
        ),
        # Main content area: visualization, results table, and statistics panels
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
                    ui.h4("Summative Statistics"),
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

# ==================== SERVER LOGIC ====================
def server(input, output, session):
    # Human-readable metric names for display in visualizations
    metric_titles = {
        'in_degree': 'In-Degree Centrality',
        'pagerank': 'PageRank',
        'eigenvector': 'Eigenvector Centrality',
        'total_review_count': 'Total Review Count'
    }
    
    # Reactive state: tracks the currently selected node from table interactions
    selected_node = reactive.Value(None)
    
    # Performance optimization: cache network layout positions to prevent recalculation on re-renders
    layout_cache = {}
    
    # Performance optimization: store pre-computed community structure and graph-level metrics
    community_data = {}
    
    # Reactive computation: performs expensive calculations once and caches results
    @reactive.calc
    def load_data():
        print("Computing metrics in background...")
        
        # Step 1: Community Detection using Infomap algorithm
        print("Running Infomap community detection...")
        im = Infomap(directed=True, silent=True)
        for u, v in G.edges():
            im.add_link(u, v)
        im.run()
        
        # Extract community assignments for each node
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
        
        # Build dropdown options showing community ID and size
        community_list = [(comm_id, len(nodes)) for comm_id, nodes in communities.items()]
        community_list.sort(key=lambda x: x[0])
        community_choices = {"all": "All communities"}
        for comm_id, size in community_list:
            community_choices[str(comm_id)] = f"Community {comm_id} ({size} nodes)"
        
        # Step 2: Calculate centrality metrics for all nodes
        print("Calculating PageRank...")
        pagerank = nx.pagerank(G, alpha=0.85, max_iter=100)
        
        print("Calculating Eigenvector Centrality...")
        eigenvector = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-6)
        
        # Step 3: Build comprehensive node dataframe with all metrics
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
        
        # Step 4: Enrich with product metadata (titles, categories, reviews)
        nodes = nodes.merge(products, on='node', how='left')
        nodes['title'] = nodes['title'].fillna('Unknown Product')
        nodes['group'] = nodes['group'].fillna('Unknown')
        nodes['total_review_count'] = nodes['total_review_count'].fillna(0)
        nodes['average_rating'] = nodes['average_rating'].fillna(0.0)
        
        print("All metrics computed!")
        
        # Cache community structure for efficient reuse in summary statistics
        community_data['communities_dict'] = communities
        community_data['node_to_community'] = node_to_community
        
        # Pre-calculate expensive graph-level metrics once
        community_data['density'] = nx.density(G)
        community_data['modularity'] = nx.algorithms.community.modularity(G, list(communities.values()))
        
        # Dynamically populate community dropdown with detected communities
        ui.update_select(
            "community",
            choices=community_choices,
            selected="all"
        )
        
        return nodes

    # Helper function: applies user-selected filters and returns top N nodes
    def get_filtered_df():
        # Retrieve cached data from reactive computation
        nodes = load_data()
        
        df = nodes.copy()
        comm = input.community()
        if comm and comm != "all":
            # Filter to selected community only
            df = df[df["community"] == int(comm)]
        # Sort by selected metric and return top N nodes
        metric = input.metric()
        top_n = int(input.top_n() or 20)
        df = df.sort_values(metric, ascending=False).head(top_n)
        return df

    # Render the main visualization based on selected type (bar/network/metric plot)
    @output
    @render.ui
    def graph_ui():
        try:
            # Establish reactive dependency on filtered data
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
            # Bar Chart: Visualize top nodes ranked by selected metric
            # Apply red outline to highlight user-selected node
            selected = selected_node()
            colors = ['red' if node == selected else 'rgba(0,0,0,0)' for node in df['node']]
            widths = [3 if node == selected else 0 for node in df['node']]
            
            fig = go.Figure(data=[
                go.Bar(
                    x=df['node'],
                    y=df[metric],
                    marker=dict(
                        color=df[metric],
                        colorscale='Viridis',
                        line=dict(
                            color=colors,
                            width=widths
                        )
                    ),
                    text=df['node'],
                    hovertemplate='Node: %{text}<br>' + metric + ': %{y}<extra></extra>'
                )
            ])
            fig.update_layout(
                title=f"Top {top_n} Nodes by {metric_titles[metric]}",
                xaxis_title='Node ID',
                yaxis_title=metric.capitalize()
            )
            fig.update_xaxes(type='category')
        elif viz_type == "metric":
            # Scatter Plot: Explore relationship between centrality metric and review count
            selected = selected_node()
            
            # Apply visual highlighting to selected node
            colors_list = ['red' if node == selected else 'rgba(100, 149, 237, 0.6)' for node in df['node']]
            sizes = [15 if node == selected else 8 for node in df['node']]
            
            fig = go.Figure(data=[
                go.Scatter(
                    x=df[metric],
                    y=df['total_review_count'],
                    mode='markers',
                    marker=dict(
                        size=sizes,
                        color=df[metric],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title=metric.capitalize()),
                        line=dict(
                            color=colors_list,
                            width=2
                        )
                    ),
                    text=df['node'],
                    hovertemplate='Node: %{text}<br>' + metric + ': %{x}<br>Total Reviews: %{y}<extra></extra>'
                )
            ])
            fig.update_layout(
                title=f"{metric_titles[metric]} vs. Review Count (Top {top_n} Nodes)",
                xaxis_title=metric_titles[metric],
                yaxis_title='Review Count',
                height=500
            )
        else:
            # Network Graph: Interactive force-directed visualization of node relationships
            # Extract subgraph containing only the filtered nodes
            top_nodes = df['node'].tolist()
            subgraph = G.subgraph(top_nodes).copy()
            
            # Generate unique cache key from node set to reuse layouts
            cache_key = tuple(sorted(top_nodes))
            
            # Retrieve or compute spring layout (prevents layout jumping on re-renders)
            if cache_key not in layout_cache:
                layout_cache[cache_key] = nx.spring_layout(subgraph, k=0.5, iterations=50, seed=42)
            pos = layout_cache[cache_key]
            
            # Build edge traces for all connections (optimized with list comprehension)
            edge_traces = [
                go.Scatter(
                    x=[pos[edge[0]][0], pos[edge[1]][0]],
                    y=[pos[edge[0]][1], pos[edge[1]][1]],
                    mode='lines',
                    line=dict(width=0.5, color='#888'),
                    hoverinfo='none',
                    showlegend=False
                )
                for edge in subgraph.edges()
            ]
            
            # Add directional arrow annotations to show edge direction
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
            
            # Build node trace with positions, colors, sizes, and hover text (optimized)
            selected = selected_node()
            nodes_list = list(subgraph.nodes())
            
            # Create lookup dictionary for O(1) node data access
            node_info_dict = {row['node']: row for _, row in df.iterrows()}
            
            node_x = [pos[node][0] for node in nodes_list]
            node_y = [pos[node][1] for node in nodes_list]
            
            node_text = []
            for node in nodes_list:
                node_info = node_info_dict[node]
                metric_value = f"{int(node_info[metric])}" if metric == 'total_review_count' else f"{node_info[metric]:.4f}"
                node_text.append(
                    f"{node_info['title']}<br>"
                    f"Node ID: {node}<br>"
                    f"Group: {node_info['group']}<br>"
                    f"{metric}: {metric_value}<br>"
                    f"Avg Rating: {node_info['average_rating']:.2f}<br>"
                    f"Total Reviews: {int(node_info['total_review_count'])}<br>"
                    f"Community: {node_info['community']}"
                )
            
            node_size = [30 if node == selected else 15 for node in nodes_list]
            node_color = [node_info_dict[node][metric] for node in nodes_list]
            
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
                              title=f'Network graph colored by {metric_titles[metric]}. Click/drag to zoom the graph.',
                              showlegend=False,
                              hovermode='closest',
                              margin=dict(b=0, l=0, r=0, t=40),
                              annotations=annotations,
                              xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                              yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                          )
        
        # Convert Plotly figure to HTML and embed in Shiny UI
        html_fragment = fig.to_html(include_plotlyjs="cdn", full_html=False)
        return ui.HTML(html_fragment)

    # Render comprehensive network statistics for entire graph and selected community
    @output
    @render.ui
    def summary_ui():
        try:
            all_nodes = load_data()
            df = get_filtered_df()
        except:
            return ui.HTML("<p>Loading...</p>")
        
        # ========== ENTIRE GRAPH STATISTICS ==========
        total_nodes = G.number_of_nodes()
        total_edges = G.number_of_edges()
        entire_avg_degree = 2 * total_edges / total_nodes if total_nodes > 0 else 0.0
        
        # Calculate average in/out degree for entire graph
        entire_avg_in = float(all_nodes["in_degree"].mean()) if len(all_nodes) > 0 else 0.0
        entire_avg_out = float(all_nodes["out_degree"].mean()) if len(all_nodes) > 0 else 0.0
        
        # Retrieve pre-computed density and modularity from cache
        entire_density = community_data.get('density', nx.density(G))
        modularity = community_data.get('modularity', 0.0)
        
        # Classify density
        if entire_density < 0.1:
            entire_density_label = "Low"
        elif entire_density < 0.3:
            entire_density_label = "Medium"
        else:
            entire_density_label = "High"
        
        # Classify modularity
        if modularity < 0.3:
            modularity_label = "Low"
        elif modularity < 0.7:
            modularity_label = "Medium"
        else:
            modularity_label = "High"
        
        # ========== SELECTED COMMUNITY STATISTICS ==========
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
            
            # Extract subgraph for community-specific calculations
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
            
            # Calculate average clustering coefficient (measures local connectivity)
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
            
            # Calculate reciprocity (percentage of bidirectional edges)
            community_edges = community_subgraph.number_of_edges()
            if community_edges > 0:
                reciprocal_edges = sum(1 for u, v in community_subgraph.edges() if community_subgraph.has_edge(v, u)) / 2
                community_reciprocity = (reciprocal_edges / community_edges) * 100
            else:
                community_reciprocity = 0.0
            
            # Identify most influential products by different centrality measures
            community_top_products = ""
            if len(community_df) > 0:
                # Top by Degree
                if 'degree' in community_df.columns:
                    top_degree_idx = community_df['degree'].idxmax()
                    top_degree_product = community_df.loc[top_degree_idx, 'title']
                    top_degree_value = community_df.loc[top_degree_idx, 'in_degree']
                    community_top_products += f"""
          <p><strong>Top Product by In-Degree:</strong> {top_degree_product[:35]}{'...' if len(str(top_degree_product)) > 35 else ''} ({top_degree_value})</p>"""
                
                # Top by PageRank
                if 'pagerank' in community_df.columns:
                    top_pagerank_idx = community_df['pagerank'].idxmax()
                    top_pagerank_product = community_df.loc[top_pagerank_idx, 'title']
                    top_pagerank_value = community_df.loc[top_pagerank_idx, 'pagerank']
                    community_top_products += f"""
          <p><strong>Top Product by PageRank:</strong> {top_pagerank_product[:35]}{'...' if len(str(top_pagerank_product)) > 35 else ''} ({top_pagerank_value:.6f})</p>"""
                
                # Top by Eigenvector
                if 'eigenvector' in community_df.columns:
                    top_eigen_idx = community_df['eigenvector'].idxmax()
                    top_eigen_product = community_df.loc[top_eigen_idx, 'title']
                    top_eigen_value = community_df.loc[top_eigen_idx, 'eigenvector']
                    community_top_products += f"""
          <p><strong>Top Product by Eigenvector:</strong> {top_eigen_product[:35]}{'...' if len(str(top_eigen_product)) > 35 else ''} ({top_eigen_value:.6f})</p>"""
            
            community_section = f"""
        <div style="background-color: #f0f8ff; padding: 10px; margin: 10px 0; border-radius: 5px;">
          <h5 style="margin-top: 0;">Selected Community</h5>
          <p><strong>Nodes:</strong> {community_nodes_count:,}</p>
          <p><strong>Average In-Degree:</strong> {community_avg_in:.2f}</p>
          <p><strong>Density:</strong> {community_density:.4f} <span style="color: #666;">({community_density_label})</span></p>
          <p><strong>Average Clustering:</strong> {community_clustering:.4f} <span style="color: #666;">({community_clustering_label})</span></p>
          <p><strong>Reciprocity:</strong> {community_reciprocity:.1f}% bidirectional</p>{community_top_products}
        </div>"""
        
        html = f"""
        <div style="font-size: 13px;">
          <div style="background-color: #fff3cd; padding: 10px; margin-bottom: 10px; border-radius: 5px;">
            <h5 style="margin-top: 0;">Entire Graph</h5>
            <p><strong>Nodes:</strong> {total_nodes:,}</p>
            <p><strong>Edges:</strong> {total_edges:,}</p>
            <p><strong>Average In-Degree:</strong> {entire_avg_in:.2f}</p>
            <p><strong>Density:</strong> {entire_density:.6f} <span style="color: #666;">({entire_density_label})</span></p>
            <p><strong>Modularity:</strong> {modularity:.4f} <span style="color: #666;">({modularity_label})</span></p>
          </div>
          {community_section}
          <p style="margin-top: 10px; font-size: 12px;"></p>
        </div>
        """
        return ui.HTML(html)
    
    # Render pie chart showing distribution of product categories in filtered nodes
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
    
    # Extract and display most frequent words from product titles
    @output
    @render.ui
    def common_words_list():
        try:
            df = get_filtered_df()
        except:
            return ui.HTML("<p>Loading...</p>")
        
        if len(df) > 0 and 'title' in df.columns:
            # Tokenize titles and filter out stop words and short words
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

    # Render interactive data table with node details and metrics
    @output
    @render.data_frame
    def results_table():
        try:
            df = get_filtered_df()
        except:
            # Return empty dataframe with message while loading
            return render.DataGrid(pd.DataFrame({"Status": ["Computing metrics... please wait"]}))
        result_df = df[["node", "title", "group", "community", "in_degree", "out_degree", 
                        "pagerank", "eigenvector", "total_review_count", "average_rating"]].copy()
        
        # Format numeric values for readability (avoid scientific notation)
        result_df['pagerank'] = result_df['pagerank'].apply(lambda x: f"{x:.8f}")
        result_df['eigenvector'] = result_df['eigenvector'].apply(lambda x: f"{x:.8f}")
        result_df['average_rating'] = result_df['average_rating'].apply(lambda x: f"{x:.2f}")
        result_df['total_review_count'] = result_df['total_review_count'].apply(lambda x: f"{int(x)}")
        
        # Rename columns for display
        result_df = result_df.rename(columns={
            'node': 'Node ID',
            'community': 'Community',
            'title': 'Product Name',
            'group': 'Category',
            'total_review_count': 'Review Count',
            'average_rating': 'Average Rating',
            'in_degree': 'In-Degree',
            'out_degree': 'Out-Degree',
            'pagerank': 'PageRank',
            'eigenvector': 'Eigenvector Centrality'
        })
        
        return render.DataGrid(result_df.reset_index(drop=True), selection_mode="row", height="600px")
    
    # Reactive observer: sync table row selection to visualization highlighting
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
