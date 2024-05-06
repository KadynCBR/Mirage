from dash import Dash, html, dcc, callback, Output, Input, State
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import pickle
from mirage.plotting import plot_homogeneous_skeleton3D_animation
from mirage.skeleton import SkeletonDetection3D
import dash_player as dp
from flask import Flask, Response
import os
import base64
import subprocess
from glob import glob

filename = "Eva02"
server = Flask(__name__)
app = Dash(__name__, server=server, external_stylesheets=[dbc.themes.SLATE])
lazyreconstruct = pickle.load(open(f"static/{filename}.sk3d", "rb"))
fig = plot_homogeneous_skeleton3D_animation(lazyreconstruct)
processed_files_list = [g.split("/")[-1].split(".")[0] for g in glob("static/*.sk3d")]

app.layout = html.Div(
    [
        dcc.Location(id="url", refresh=False),
        dbc.NavbarSimple(
            children=[],
            brand="Mirage Low Cost Motion Capture",
            brand_href="#",
            color="primary",
            dark=True,
        ),
        dbc.Row(
            [
                dcc.Upload(
                    id="upload-data",
                    children=html.Div(["Drag and Drop or ", html.A("Select Video File")]),
                    style={
                        "width": "75%",
                        "height": "100px",
                        "lineHeight": "100px",
                        "borderWidth": "1px",
                        "borderStyle": "dashed",
                        "borderRadius": "5px",
                        "textAlign": "center",
                        "marginLeft": "auto",
                        "marginRight": "auto",
                        "marginTop": "10px",
                        "marginBottom": "10px",
                    },
                    # Allow multiple files to be uploaded
                    multiple=False,
                ),
                html.Div(id="output-data-upload"),
            ],
            align="center",
        ),
        dbc.Row(
            [dcc.Dropdown(processed_files_list, processed_files_list[0], id="data-selection")],
            align="center",
            style={"marginBottom": "10px", "marginTop": "10px"},
        ),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Row(
                        [
                            html.Video(
                                id="originalvideo",
                                src=f"/static/{filename}.mp4",
                                preload="auto",
                                muted=True,
                                controls=True,
                            )
                        ]
                    ),
                    width=4,
                ),
                dbc.Col(
                    dbc.Row(
                        [
                            html.Video(
                                id="processedvideo",
                                src=f"/static/{filename}Processed.mp4",
                                preload="auto",
                                muted=True,
                                controls=True,
                            )
                        ]
                    ),
                    width=4,
                ),
                dbc.Col(dcc.Graph(figure=fig, id="sk3d-graph"), width=4),
            ],
            align="center",
        ),
        dbc.Row([html.Button("Play/Pause Animations", id="startAnimationButton")]),
        html.P(id="placeholder", hidden=True),  # plotly doesn't like client callbacks without output.
    ],
)


@server.route("/static/<path:path>")
def serve_static(path):
    root_dir = os.getcwd()
    return Flask.send_from_directory(os.path.join(root_dir, "static"), path)


@callback(
    [Output("originalvideo", "src"), Output("processedvideo", "src"), Output("sk3d-graph", "figure")],
    Input("data-selection", "value"),
)
def update_filename(new_fn):
    if new_fn is None:
        return ""
    print(f"hello {new_fn}")
    lr = pickle.load(open(f"static/{filename}.sk3d", "rb"))
    figr = plot_homogeneous_skeleton3D_animation(lr)
    return (f"/static/{filename}.mp4", f"/static/{filename}Processed.mp4", figr)


@callback(
    Output("output-data-upload", "children"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
)
def update_output(contents, filename):
    if contents is not None:
        parse_contents(contents, filename)


app.clientside_callback(
    """
    function startVideos(n_clicks) {
        if (n_clicks > 0) {
        
            console.log(dash_clientside.callback_context);
            var originalVideo = document.getElementById('originalvideo');
            var processedVideo = document.getElementById('processedvideo');
            if (originalVideo.paused) {
                document.querySelector("g.updatemenu-header-group > g:nth-child(1)").dispatchEvent(new Event('click'))
                originalVideo.play();
                processedVideo.play();
            } else {
                originalVideo.pause();
                processedVideo.pause();
                document.querySelector("g.updatemenu-header-group > g:nth-child(2)").dispatchEvent(new Event('click'))
            }
        }
        return false;
    }
    """,
    Output("placeholder", "hidden"),
    Input("startAnimationButton", "n_clicks"),
)


def parse_contents(contents, filename):
    content_type, content_string = contents.split(",")
    try:
        if "mp4" in filename:
            decoded = base64.b64decode(content_string)
            print(f"Writing file {filename}")
            with open(f"static/{filename}", "wb") as f:
                f.write(decoded)
            print("spinning up process.")
            # spin up process to operate on static folder and save outputs to static folder
            subprocess.run(["python", "../mirage/core.py", f"static/{filename}", "-o", "static/"])
            return html.Div(f"File: {filename} uploaded! When processing is done, click nav button.")
    except Exception as e:
        print(e)


# @app.route("/<videotitle>")
# def display_processed_data(videotitle):
#     lazyreconstruct = pickle.load(open(f"static/{filename}.sk3d", "rb"))
#     fig = plot_homogeneous_skeleton3D_animation(lazyreconstruct)
#     return "hi"


if __name__ == "__main__":
    app.run(debug=True, host="neutralcity")
