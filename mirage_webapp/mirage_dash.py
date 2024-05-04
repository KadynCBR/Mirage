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

filename = "Dancing"

server = Flask(__name__)
app = Dash(__name__, server=server, external_stylesheets=[dbc.themes.SLATE])
lazyreconstruct = pickle.load(open(f"static/{filename}.sk3d", "rb"))

fig = go.Figure(data=[go.Scatter(x=[1, 2, 3], y=[4, 1, 2])])

app.layout = html.Div(
    [
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
            [
                dbc.Col(
                    dbc.Row(
                        [html.Video(id="originalvideo", src=f"/static/{filename}.mp4", autoPlay=False, controls=True)]
                    ),
                    width=4,
                ),
                dbc.Col(
                    dbc.Row(
                        [
                            html.Video(
                                id="processedvideo",
                                src=f"/static/{filename}Processed.mp4",
                                autoPlay=False,
                                controls=True,
                            )
                        ]
                    ),
                    width=4,
                ),
                dbc.Col(dcc.Graph(figure=plot_homogeneous_skeleton3D_animation(lazyreconstruct)), width=4),
            ],
            align="center",
        ),
        dbc.Row([html.Button("Start Animations", id="startAnimationButton")]),
    ],
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


# @app.callback(
#     [Output("originalvideo", "autoPlay"), Output("processedvideo", "autoPlay")],
#     Input("startAnimationButton", "n_clicks"),
# )
# def start_animation(n_clicks):
#     print("Clicked")
#     autoPlay = True
#     return autoPlay, autoPlay

app.clientside_callback(
    """
    function startVideos() {
        var originalVideo = document.getElementById('originalvideo');
        var processedVideo = document.getElementById('processedvideo');
        originalVideo.play();
        processedVideo.play();
    }
    """,
    Output("originalvideo", "autoPlay"),
    Output("processedvideo", "autoPlay"),
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
            # save video to static folder
            # spin up process to operate on static folder and save outputs to static folder
            return html.Div(f"File: {filename} uploaded! When processing is done, click nav button.")
    except Exception as e:
        print(e)


if __name__ == "__main__":
    app.run(debug=True, host="neutralcity")
