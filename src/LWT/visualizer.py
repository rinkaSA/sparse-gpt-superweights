from plotly.subplots import make_subplots
import plotly.graph_objects as go


def plot_round_metrics_plotly(metrics_per_round, sw_preserved, sparsity_map):
    """
    Creates a Plotly figure with one subplot per round.
    Left y-axis: train & val loss; right y-axis: perplexity.
    Uses consistent colors/styles:
      - Perplexity: dashed blue
      - Train loss: solid orange
      - Val loss: dotted green
    One legend entry per metric type; clicking toggles all corresponding traces.
    Marks the final train-loss point in green if SW preserved, red if not.
    Includes sparsity level in subplot titles.

    Parameters
    ----------
    metrics_per_round : dict[int, dict]
        Mapping round -> {
            'step': list of step indices,
            'train_loss': list of train-loss values,
            'val_loss': list of val-loss values,
            'ppl': list of perplexity values
        }
    sw_preserved : dict[int, bool]
        Mapping round -> True if SW preserved, False otherwise.
    sparsity_map : dict[int, float]
        Mapping round -> sparsity percentage (e.g. 87.5 for 87.5%)

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    # color & style constants
    COL_PPL   = '#1f77b4'  # blue
    COL_TRAIN = '#ff7f0e'  # orange
    COL_VAL   = '#2ca02c'  # green

    DASH_PPL   = 'dash'    # dashed for ppl
    DASH_TRAIN = 'solid'   # solid for train
    DASH_VAL   = 'dot'     # dotted for val

    rounds = sorted(metrics_per_round.keys())
    n_rounds = len(rounds)

    # Build subplot titles including sparsity
    titles = [
        f"Round {r} (sparsity: {sparsity_map.get(r, 0):.1f}%)"
        for r in rounds
    ]

    fig = make_subplots(
        rows=n_rounds, cols=1,
        shared_xaxes=False,
        subplot_titles=titles,
        specs=[[{"secondary_y": True}] for _ in rounds]
    )

    # Flags to ensure single legend entry per metric
    for idx, rnd in enumerate(rounds, start=1):
        data = metrics_per_round[rnd]
        steps = data['step']

        # Only show legend on first subplot
        show_legend = (idx == 1)

        # Train loss (solid orange)
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=data['train_loss'],
                mode='lines',
                name='Train Loss',
                legendgroup='train',
                showlegend=show_legend,
                line=dict(color=COL_TRAIN, dash=DASH_TRAIN)
            ),
            row=idx, col=1, secondary_y=False
        )
        # Val loss (dotted green)
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=data['val_loss'],
                mode='lines',
                name='Val Loss',
                legendgroup='val',
                showlegend=show_legend,
                line=dict(color=COL_VAL, dash=DASH_VAL)
            ),
            row=idx, col=1, secondary_y=False
        )
        # Perplexity (dashed blue) on secondary axis
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=data['ppl'],
                mode='lines',
                name='Perplexity',
                legendgroup='ppl',
                showlegend=show_legend,
                line=dict(color=COL_PPL, dash=DASH_PPL)
            ),
            row=idx, col=1, secondary_y=True
        )
        # Final train-loss marker (marker group)
        last_step  = steps[-1]
        last_train = data['train_loss'][-1]
        mk_color   = 'green' if sw_preserved.get(rnd, False) else 'red'
        fig.add_trace(
            go.Scatter(
                x=[last_step],
                y=[last_train],
                mode='markers',
                name='Final Train',
                legendgroup='marker',
                showlegend=show_legend,
                marker=dict(color=mk_color, size=10)
            ),
            row=idx, col=1, secondary_y=False
        )

        # Axes labels
        fig.update_yaxes(title_text="Loss", row=idx, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Perplexity", row=idx, col=1, secondary_y=True)
        fig.update_xaxes(title_text="Step", row=idx, col=1)

    fig.update_layout(
        height=350 * n_rounds,
        width=900,
        title_text="Training Metrics per Round",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            groupclick="togglegroup"
        ),
        margin=dict(t=100)
    )
    return fig