import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

data = pd.read_csv("exercise_1_results.csv")
data["samples_per_second"] = data["n_sample"] / data["elapsed"]
data["error"] = np.log10(np.abs(data["mean_estimate"] - data["true_result"]))
data["stderr"] = data["var_estimate"] / np.sqrt(data["n_sample"])
data = data[data["name"] != "Importance Sampling (D_2)"]

methods = data["name"].unique()
fig = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=methods,
    horizontal_spacing=0.1,
    vertical_spacing=0.1,
)
for i_method, method in enumerate(methods):
    sub_data = data[data["name"] == method].sort_values("n_sample")
    fig.add_scatter(
        x=sub_data["n_sample"],
        y=sub_data["mean_estimate"],
        name=method,
        row=(i_method) // 2 + 1,
        col=(i_method) % 2 + 1,
        showlegend=False,
        line=dict(color="black", width=1.0),
        error_y=dict(
            type="data",
            array=sub_data["stderr"] * 1.96,
            visible=True,
            thickness=1.0,
            width=2.0,
        ),
    )
    fig.add_hline(
        y=sub_data["true_result"].iloc[0],
        line=dict(color="red", width=1.0),
    )
fig = fig.update_yaxes(range=[0, 0.02])
fig = fig.update_layout(
    template="plotly_white",
    margin=dict(t=30, b=0, r=0, l=0),
    width=800,
    height=400,
    font=dict(color="black"),
)
fig.show()

fig = px.box(data, x="name", y="variance_reduction")
fig = fig.update_layout(
    template="plotly_white",
    margin=dict(t=30, b=0, r=0, l=0),
    width=800,
    height=300,
    font=dict(color="black"),
)
fig = fig.update_traces(marker=dict(color="black"), fillcolor="rgba(0,0,0,0.0)")
fig = fig.update_yaxes(title="Variance Reduction")
fig = fig.update_xaxes(title="")
fig.show()


fig = px.scatter(data, x="n_sample", y="error", color="name")
fig = fig.update_layout(
    template="plotly_white",
    margin=dict(t=30, b=0, r=0, l=0),
    width=800,
    height=250,
    font=dict(color="black"),
)
fig = fig.update_traces(marker=dict(line=dict(width=1, color="black")), opacity=0.8)
fig = fig.update_xaxes(title="Sample Size")
fig = fig.update_yaxes(title="Log Absolute Error")
fig.show()


fig = px.box(data, y="name", x="samples_per_second")
fig = fig.update_layout(
    template="plotly_white",
    margin=dict(t=30, b=0, r=0, l=0),
    width=800,
    height=200,
    font=dict(color="black"),
)
fig = fig.update_traces(marker=dict(color="black"), fillcolor="rgba(0,0,0,0.0)")
fig = fig.update_xaxes(title="Speed(Samples per second)")
fig = fig.update_yaxes(title="")
fig.show()
