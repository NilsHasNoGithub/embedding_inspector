from collections import defaultdict
from functools import partial
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, TypeVar, Union

import dash
import numpy as np
import numpy.typing as npt
import pandas as pd
import PIL.Image
import plotly.express as px
import plotly.io
from dash import dcc, html
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm

plotly.io.templates.default = "seaborn"
KwArgs = Dict[str, Any]


class WrongArrayShapeError(ValueError):
    """Raised when an array does not have the required shape"""


U = TypeVar("U")
L = TypeVar("L")


def run_embedding_inspection_app(
    unique_ids: Sequence[U],
    labels: Union[Sequence[L], Sequence[Tuple[L, L]]],
    load_embedding: Callable[[U], np.ndarray],
    load_image: Callable[[U], PIL.Image.Image],
    /,
    dim_reduction_fn: Optional[
        Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]
    ] = None,
    dash_app_kwargs: Optional[KwArgs] = None,
    dash_app_run_kwargs: Optional[KwArgs] = None,
    scatter_fig_kwargs: Optional[KwArgs] = None,
    tag: Optional[str] = None,
) -> None:
    """Run a small application which allows for inspecting (model) embeddings.
    The embeddings are first reduced in dimensionality, using PCA with 50 + TSNE with 2 components. This can changed by providing a different `dim_reduction_fn`.
    Next, a Dash server is created, which allows for hovering over points in the reduced space and inspecting the images in the embeddings.



    ## Parameters:
    - `unique_ids` (`Sequence[U]`): A series of unique IDs, each corresponding to one of the samples
    - `labels` (`Union[Sequence[L],Sequence[Tuple[L,L]]]`): A label, or 2 labels, corresponding to each samples. One for color and the other for shape. If one is provided, only color is used
    - `load_embedding` (`Callable[[U], np.ndarray]`): Function to generate the embedding for unique id, this function should return a 1 dimensional array
    - `load_image` (`Callable[[U], PIL.Image.Image]`): Load an image given a unique ID, used for displaying the images on hover/click
    - `dim_reduction_fn` (`Optional[Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]]`, optional): Optional function to replace the default dimensionality reduction technique. Defaults to None.
    - `dash_app_kwargs` (`Optional[KwArgs]`, optional): Keyword arguments to provide to the `dash.Dash` call. Defaults to None.
    - `dash_app_run_kwargs` (`Optional[KwArgs]`, optional): Extra keywords to provide to the `app.run` call, `app` being an instance of dash.Dash. Defaults to None.
    - `scatter_fig_kwargs` (`Optional[KwArgs]`, optional): Extra keyword arguments to provide to the `px.scatter(_3d)` call, useful for changing the colors, symbol or opacity. Defaults to None.
    - `tag` (`Optional[str]`, optional): Tag to add to the top of the webpage. Defaults to None.

    ## Raises
    - `ValueError`: when the number of unique ids does not match the number of labels
    - `ValueError`: when the number of unique ids is 0
    - `WrongArrayShapeError`: When the output of `dim_reduction_fn` is not of the correct shape

    ## Returns
    This function does not return!

    """

    if len(unique_ids) != len(labels):
        raise ValueError("Number of unique IDs and labels should be the same")

    if len(unique_ids) == 0:
        raise ValueError("There should be at least one sample to plot")

    if dash_app_kwargs is None:
        dash_app_kwargs = {}
    if dash_app_run_kwargs is None:
        dash_app_run_kwargs = {}
    if scatter_fig_kwargs is None:
        scatter_fig_kwargs = {}

    if dim_reduction_fn is None:
        dim_reduction_fn = lambda x: TSNE().fit_transform(
            PCA(n_components=50).fit_transform(x)
        )

    #### Create plotting data
    embeds = np.stack(
        [load_embedding(uid) for uid in tqdm(unique_ids, desc="Extracting embeddings")]
    )
    embeds_reduced = dim_reduction_fn(embeds)
    if embeds_reduced.shape not in [(len(unique_ids), 2), (len(unique_ids), 3)]:
        raise WrongArrayShapeError(
            f"Expected `dim_reduction_fn` to return an array of shape (len(unique_ids), 2/3), but got {embeds_reduced.shape}"
        )

    two_labels = isinstance(labels[0], tuple)
    is_3d = embeds_reduced.shape[1] == 3

    plot_data = defaultdict(list)

    for idx, (uid, label) in enumerate(zip(unique_ids, labels)):
        x, y, *z = embeds_reduced[idx, :]

        plot_data["uid"].append(uid)
        plot_data["x"].append(x)
        plot_data["y"].append(y)

        if is_3d:
            plot_data["z"].append(z[0])

        if two_labels:
            plot_data["color"].append(label[0])
            plot_data["symbol"].append(label[1])
        else:
            plot_data["color"].append(label)

    scatter_fn = partial(px.scatter_3d, z="z") if is_3d else px.scatter
    fig = scatter_fn(
        pd.DataFrame(plot_data),
        x="x",
        y="y",
        color="color",
        symbol="symbol" if two_labels else None,
        custom_data="uid",
        **scatter_fig_kwargs,
    )

    fig.update_xaxes(title_text="1st component")
    fig.update_yaxes(title_text="2nd component")
    if is_3d:
        fig.update_layout(scene=dict(zaxis=dict(title="3d component")))
    # fig.update_traces(marker={"size": 15})

    #### Create app
    app = dash.Dash(__name__, **dash_app_kwargs)

    app.layout = html.Div(
        ([html.H2(tag)] if tag is not None else [])
        + [
            dcc.Graph(
                id="scatter-plot",
                figure=fig,
                style={"height": "50vh"},
            ),
            html.Div(
                id="image-display-fixed",
                style={"width": "50%", "display": "inline-block", "align": "center"},
            ),
            html.Div(
                id="image-display",
                style={"width": "50%", "display": "inline-block", "align": "center"},
            ),
        ],
        style={"height": "100%"},
    )

    # Define the callback function to display the image
    @app.callback(
        dash.dependencies.Output("image-display-fixed", "children"),
        [dash.dependencies.Input("scatter-plot", "clickData")],
    )
    def display_fixed_image(click_data):
        if click_data is not None:
            return html.Img(
                src=load_image(click_data["points"][0]["customdata"][0]), width="40%"
            )
        else:
            return ""
    
    @app.callback(
        dash.dependencies.Output("image-display", "children"),
        [dash.dependencies.Input("scatter-plot", "hoverData")],
    )
    def display_image(hover_data):
        if hover_data is not None:
            return html.Img(
                src=load_image(hover_data["points"][0]["customdata"][0]), width="40%"
            )
        else:
            return ""

    app.run(**dash_app_run_kwargs)
