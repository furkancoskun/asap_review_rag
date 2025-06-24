import wandb
import plotly.graph_objects as go
import time 
import numpy as np

wandb.init(project="IS-584-Term-Project", name="ragas_visualization")

def bin_metric_values(values):
    bins = [0, 0.3, 0.75, 1.01]
    return np.clip(np.digitize(values, bins) - 1, 0, 2)

def log_radar_plot(metrics_data, model_pair):
    fig = go.Figure()
    metrics = list(metrics_data.keys())
    values = list(metrics_data.values())
    metrics += [metrics[0]]  # Close radar loop
    values += [values[0]]


    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=metrics,
        fill='toself',
        name=model_pair
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1], tickvals=[0, 0.5, 1])),
        title=dict(text=f"Radar Plot - {model_pair}", x=0.5, xanchor='center'),
        showlegend=True
    )


    # Save and log radar plot
    timestamp = int(time.time())
    radar_html_path = f"./radar_plot_{model_pair}_{timestamp}.html"
    fig.write_html(radar_html_path, auto_play=False)
    wandb.log({f"Radar Plot {model_pair}": wandb.Html(radar_html_path)})


def log_heatmap(heatmap_data, metric_name, gen_models, embed_models):
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=["Low", "Medium", "High"],
        y=[f"{gen}_{embed}" for gen, embed in zip(gen_models, embed_models)],
        colorscale='YlGnBu', showscale=True
    ))
    fig.update_layout(
        title=f"{metric_name} Heatmap",
        xaxis_title="Score Bin", yaxis_title="Model Pair"
    )

    heatmap_html_path = f"./heatmap_{metric_name}.html"
    fig.write_html(heatmap_html_path, auto_play=False)
    wandb.log({f"{metric_name} Heatmap": wandb.Html(heatmap_html_path)})
