# app.py
from shiny import App, render, ui

# Define the user interface
app_ui = ui.page_fluid(
    ui.h2("Hello, Python Shiny!"),
    ui.input_text("name", "Enter your name:", value="Noah"),
    ui.input_slider("age", "Select your age:", 0, 100, 25),
    ui.output_text("greeting"),
)

# Define server logic
def server(input, output, session):
    @output
    @render.text
    def greeting():
        return f"Hello, {input.name()}! You are {input.age()} years old."

# Create and run the app
app = App(app_ui, server)
