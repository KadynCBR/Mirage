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

filename = "Dancing"
server = Flask(__name__)
app = Dash(__name__, server=server, external_stylesheets=[dbc.themes.SLATE], suppress_callback_exceptions=True)
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
            [dcc.Dropdown(processed_files_list, "", id="data-selection", clearable=True)],
            align="center",
            style={"marginBottom": "10px", "marginTop": "10px"},
        ),
        html.Div(id="page-content"),
        html.P(id="placeholder", hidden=True),  # plotly doesn't like client callbacks without output.
        html.P(id="placeholder2", hidden=True),  # plotly doesn't like client callbacks without output.
    ],
)


@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def display_page(pathname):
    if pathname is None:
        return "404 - not found"
    else:
        filename = pathname.lstrip("/")
        return generate_content(filename)


def generate_content(fn):
    if fn == "":
        return html.Div([html.H1("Select Data From Drop Down", style={"marginLeft": "auto", "marginRight": "auto"})])
    else:
        lr = pickle.load(open(f"static/{fn}.sk3d", "rb"))
        figr = plot_homogeneous_skeleton3D_animation(lr)
        return html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Row(
                                [
                                    html.Video(
                                        id="originalvideo",
                                        src=f"/static/{fn}.mp4",
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
                                        src=f"/static/{fn}Processed.mp4",
                                        preload="auto",
                                        muted=True,
                                        controls=True,
                                    )
                                ]
                            ),
                            width=4,
                        ),
                        dbc.Col(dcc.Graph(figure=figr, id="sk3d-graph"), width=4),
                    ],
                    align="center",
                ),
                dbc.Row([html.Button("Play/Pause Animations", id="startAnimationButton")]),
            ]
        )


@server.route("/static/<path:path>")
def serve_static(path):
    root_dir = os.getcwd()
    return Flask.send_from_directory(os.path.join(root_dir, "static"), path)


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


app.clientside_callback(
    """
    function(value) {
        if (value != null && value.length > 0) {
            window.location.href = value;
        }
    }
    return true;
    """,
    Output("placeholder2", "hidden"),
    Input("data-selection", "value"),
)


if __name__ == "__main__":
    app.run(debug=True, host="neutralcity")
