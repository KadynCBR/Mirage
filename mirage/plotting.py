import plotly
import plotly.graph_objs as go
from mirage.skeleton import SkeletonDetection3D
from mirage.mirage_helpers import KeypointEdges


def get_homogeneous_skeleton3D_plot_data(
    skeleton_3d_model: SkeletonDetection3D, frame_number: int, zmin: int = -40, zmax: int = 120
):
    def zoff(z: int):
        return 1280 - z - 720

    X_IND = 0
    Y_IND = 4
    Z_IND = 2
    joint_saved_states = [j.get_estimate_at(frame_number) for j in skeleton_3d_model.joints.values()]
    # Plot joint points 3D
    trace = go.Scatter3d(
        x=[j[X_IND] for j in joint_saved_states] + [0, 1280],
        y=[j[Y_IND] for j in joint_saved_states] + [0, 1280],
        z=[zoff(j[Z_IND]) for j in joint_saved_states] + [zmin, zmax],  # <-- Make min Z based on estimates?
        mode="markers",
        marker={
            "size": 3,
            "opacity": 0.8,
        },
    )
    # Plot joint connections 3D
    linetraces = []
    for p1, p2 in KeypointEdges.keys():
        p1Joint = skeleton_3d_model.joints[p1]
        p2Joint = skeleton_3d_model.joints[p2]
        linetrace = go.Scatter3d(
            x=[p1Joint.get_estimate_at(frame_number)[X_IND]] + [p2Joint.get_estimate_at(frame_number)[X_IND]],
            y=[p1Joint.get_estimate_at(frame_number)[Y_IND]] + [p2Joint.get_estimate_at(frame_number)[Y_IND]],
            z=[zoff(p1Joint.get_estimate_at(frame_number)[Z_IND])]
            + [zoff(p2Joint.get_estimate_at(frame_number)[Z_IND])],
            mode="lines",
            marker={
                "size": 3,
                "opacity": 0.8,
            },
        )
        linetraces.append(linetrace)
    data = [trace] + linetraces
    return data


def plot_homogeneous_skeleton3D_animation(skeleton_3d_model: SkeletonDetection3D, zmin: int = -40, zmax: int = 120):
    total_frames = len(skeleton_3d_model.joints[0].saved_states)
    frames = []
    for frame_num in range(0, total_frames, 1):
        frame_data = get_homogeneous_skeleton3D_plot_data(skeleton_3d_model, frame_num, zmin, zmax)
        frames.append(go.Frame(data=frame_data))
    layout = go.Layout(
        margin={"l": 0, "r": 0, "b": 0, "t": 0},
        height=720,
        updatemenus=[
            {
                "type": "buttons",
                "buttons": [
                    {
                        "args": [
                            None,
                            {
                                "frame": {"duration": 26, "redraw": True},
                                "fromcurrent": True,
                            },
                        ],
                        "label": "Play",
                        "method": "animate",
                    },
                    {
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                        "label": "Pause",
                        "method": "animate",
                    },
                ],
                "showactive": True,
                "x": 0.1,
                "y": 0,
                "yanchor": "bottom",
            }
        ],
    )
    initial_data = get_homogeneous_skeleton3D_plot_data(skeleton_3d_model, 0, zmin, zmax)
    plot_figure = go.Figure(data=initial_data, layout=layout, frames=frames)
    plot_figure.update_layout(
        showlegend=False,
        scene=dict(
            xaxis=dict(range=[0, 1280]),  # adjust x-axis range if necessary
            yaxis=dict(range=[0, 1280]),  # adjust y-axis range if necessary
            zaxis=dict(range=[-50, 720]),  # adjust z-axis range if necessary
            # xaxis=dict(visible=False),  # adjust x-axis range if necessary
            # yaxis=dict(visible=False),  # adjust y-axis range if necessary
            # zaxis=dict(visible=False),  # adjust z-axis range if necessary
            aspectmode="manual",
            aspectratio=dict(x=2, y=2, z=1),
            camera=dict(up=dict(x=0, y=0, z=1), eye=dict(x=-1.5, y=-1.5, z=0.5)),
        ),
    )
    return plot_figure
