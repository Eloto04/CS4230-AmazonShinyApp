# app.py
"""
Preliminary Shiny dashboard layout for the Amazon graph project.

This file defines a simple sidebar + main layout with:
- controls for metric/community selection
- an interactive graph area (placeholder populated with a small example Plotly chart)
- a dynamic summary / metric box
- a results table

The actual graph/metric logic can be swapped in later where indicated.
"""
from shiny import App, ui, render, reactive
import pandas as pd
import numpy as np
import plotly.express as px


# ----------------------
# User interface (layout)
# ----------------------
app_ui = ui.page_fluid(
    ui.h2("Amazon Graph — Interactive Dashboard"),
    # Sidebar for inputs and filters
    ui.layout_sidebar(
        ui.sidebar(
            ui.h4("Filters & Controls"),
            # Select the metric that will determine node color/size in the graph
            ui.input_select(
                "metric",
                "Color / metric:",
                {
                    "degree": "Degree",
                    "betweenness": "Betweenness (placeholder)",
                    "pagerank": "PageRank (placeholder)",
                },
                selected="degree",
            ),
            # Select community (or "All") -- placeholder choices for now
            ui.input_select(
                "community",
                "Community:",
                {
                    "all": "All",
                    "c1": "Community 1",
                    "c2": "Community 2",
                    "c3": "Community 3",
                },
                selected="all",
            ),
            # Number of top nodes to show (example numeric control)
            ui.input_slider("top_n", "Show top N nodes:", min=5, max=50, value=20),
            # Manual refresh button (useful when swapping heavy visualizations)
            ui.input_action_button("refresh", "Refresh view"),
            ui.markdown(
                """
            **Notes:**
            - The graph area below is a placeholder using Plotly.
            - Replace the plotting code in `server()` with your real network visualization.
            """
            ),
        ),
        # Main area with graph + summary + table
        # (ui.layout_sidebar takes content directly after sidebar, no wrapper needed)
        ui.row(
            ui.column(8, ui.output_ui("graph_ui")),
            ui.column(
                4,
                ui.panel_well(
                    ui.h4("Summary metrics"),
                    # small dynamic HTML block to hold metrics / key numbers
                    ui.output_ui("summary_ui"),
                ),
            ),
        ),
        ui.h4("Results table"),
        # Data table showing filtered rows (Pandas DataFrame can be returned from render.table)
        ui.output_table("results_table"),
    ),
)


# ----------------------
# Server logic
# ----------------------
def server(input, output, session):
    # Create a small synthetic dataset for demonstration.
    # In your real app replace this with your graph nodes/metrics dataset.
    rng = np.random.default_rng(123)
    nodes = pd.DataFrame(
        {
            "node": [f"n{i}" for i in range(1, 101)],
            "x": rng.normal(size=100),
            "y": rng.normal(size=100),
            "degree": rng.integers(1, 50, size=100),
            "betweenness": rng.random(size=100),
            "pagerank": rng.random(size=100),
            "community": rng.choice(["c1", "c2", "c3"], size=100),
        }
    )

    # Reactive helper: build a filtered dataframe based on inputs
    def get_filtered_df():
        df = nodes.copy()
        comm = input.community()
        if comm and comm != "all":
            df = df[df["community"] == comm]
        # sort by chosen metric and take top N
        metric = input.metric() or "degree"
        top_n = int(input.top_n() or 20)
        df = df.sort_values(metric, ascending=False).head(top_n)
        return df

    # Graph UI: render a Plotly scatter and embed as HTML into the Shiny UI
    @output
    @render.ui
    def graph_ui():
        # Build a small interactive scatter that demonstrates zoom/pan and coloring.
        # Replace this with your network visualization (e.g., vis.js, pyvis, or a Plotly network layout)
        df = get_filtered_df()
        metric = input.metric() or "degree"
        fig = px.scatter(
            df,
            x="x",
            y="y",
            color=metric,
            hover_name="node",
            size=metric,
            title=f"Graph placeholder — colored by {metric}",
        )
        # Plotly figure converted to HTML fragment; Shiny will render it in the UI.
        html_fragment = fig.to_html(include_plotlyjs="cdn", full_html=False)
        return ui.HTML(html_fragment)

    # Summary UI: small metric cards (simple HTML) showing top numbers
    @output
    @render.ui
    def summary_ui():
        df = get_filtered_df()
        # Example summary values (replace with your real metrics)
        n_nodes = len(df)
        avg_degree = float(df["degree"].mean()) if n_nodes else 0.0
        # Simple HTML snippet — keep it minimal and easy to style later
        html = f"""
        <div>
          <p><strong>Shown nodes:</strong> {n_nodes}</p>
          <p><strong>Avg degree (shown):</strong> {avg_degree:.2f}</p>
          <p><em>Click/drag to zoom the graph. Use controls on the left to change metric/community.</em></p>
        </div>
        """
        return ui.HTML(html)

    # Results table: return the filtered DataFrame so Shiny will render it as a table
    @output
    @render.table
    def results_table():
        df = get_filtered_df()
        # return a small subset of columns for readability
        return df[["node", "community", "degree", "betweenness", "pagerank"]]

    # Optional: react to the refresh button (keeps app responsive when swapping heavy visualizations)
    @reactive.effect
    @reactive.event(input.refresh)
    def _on_refresh():
        # When user clicks refresh we trigger a no-op but it can be used to re-run expensive ops.
        # For now we just print to the console (visible when running locally).
        print("Refresh requested — you can reload heavy visualizations here.")


# Create the Shiny app object
app = App(app_ui, server)
