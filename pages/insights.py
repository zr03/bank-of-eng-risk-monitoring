from typing import List
import dash
from dash import dcc, html, callback, Output, Input, State, ctx, dash_table
import dash_bootstrap_components as dbc
from dash.dash_table.Format import Format, Scheme
from dash.dash_table import FormatTemplate
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from dash.exceptions import PreventUpdate
import datetime
from dateutil.relativedelta import relativedelta
import pytz
import matplotlib.colors as mcolors
import app_config as config
import numpy as np

dash.register_page(__name__,
                   path='/',
                   name='Insights',
                   title='G-SIB Risk Insights',
                   description='Risk insights for G-SIBs',)

TOPIC_RELEVANCE_FPATH = config.TOPIC_RELEVANCE_FPATH
TOPIC_RELEVANCE_WEIGHTED_SENTIMENT_FPATH = config.TOPIC_RELEVANCE_WEIGHTED_SENTIMENT_FPATH
RISK_CATEGORY_MAPPING_FPATH = config.RISK_CATEGORY_MAPPING_FPATH
COLOURS_LIST = px.colors.qualitative.Pastel
Q_A_ANALYSIS = config.Q_A_ANALYSIS_FPATH

def read_insights_data():
    df_topic_relevance = pd.read_parquet(TOPIC_RELEVANCE_FPATH)
    df_topic_relevance_weighted_sentiment = pd.read_parquet(TOPIC_RELEVANCE_WEIGHTED_SENTIMENT_FPATH)
    df_risk_category_mapping = pd.read_parquet(RISK_CATEGORY_MAPPING_FPATH)
    df_q_a_analysis = pd.read_parquet(Q_A_ANALYSIS)

    return df_topic_relevance, df_topic_relevance_weighted_sentiment, df_risk_category_mapping, df_q_a_analysis

def read_summary_data():
    pass

def get_neutral_pastel_palette(n):
    safe_hex_colors = [
        "#aec6cf", "#cfcfc4", "#fdfd96", "#b39eb5", "#ffb347",
        "#dda0dd", "#b0e0e6", "#cdb5cd", "#fab57a", "#d1cfe2",
        "#e6e6fa", "#f5deb3", "#ccccff", "#e0bbE4", "#f7cac9"
    ]
    return safe_hex_colors[:n]

def sentiment_text_color(score):
    if score >= 0.05:
        return 'green'
    elif score <= -0.05:
        return 'red'
    else:
        return 'black'

SENTIMENT_COLOR_MAP = {
    "positive": "green",
    "neutral": "black",
    "negative": "red",
}

quarter_month_map = {'Q1': '03-31', 'Q2': '06-30', 'Q3': '09-30', 'Q4': '12-31'}
def quarter_to_date(qr):
    try:
        q, y = qr.split('_')
        return pd.to_datetime(f"{y}-{quarter_month_map.get(q, '12-31')}")
    except:
        return pd.NaT

def generate_multiline_chart(df, x_data_col, y_data_cols, x_title, y_title, plot_title=None, marker_colors=COLOURS_LIST, marker_size=1, line_colors=COLOURS_LIST, line_legend_labels=None):

    assert len(y_data_cols) <= len(line_colors), "Number of y_data_cols must not exceed number of line_colors"

    fig = go.Figure()

    for i, y_data_col in enumerate(y_data_cols):
        fig.add_trace(
            go.Scatter(
                x=df[x_data_col],
                y=df[y_data_col],
                mode='lines+markers',
                marker=dict(
                    color=marker_colors[i],
                    size=marker_size,
                ),
                line=dict(color=line_colors[i]),
                name=line_legend_labels[i],
                showlegend=True,
                yaxis='y')
        )

    fig.update_layout(
        # title=plot_title,
        xaxis=dict(title=x_title),
        yaxis=dict(title=y_title),
    )

    fig.update_layout(title_text=plot_title, title_x=0.5)

    fig.update_layout(
        legend=dict(
            xanchor="center",
            x=0.5,
            yanchor="bottom",
            y=-0.5,
            orientation='h',
        )
    )
    return fig

@callback(
    Output("topic-proportion-graph", "figure"),
    Output("topic-bank-dropdown", "options"),
    Output("topic-bank-dropdown", "value"),
    Input("topic-bank-dropdown", "value")
)
def update_topic_proportion(bank_selected):
    _, _, _, df = read_insights_data()

    # Dropdown options
    bank_options = sorted(df["bank"].dropna().unique())
    default_bank = bank_selected or bank_options[0]

    df = df[df["bank"] == default_bank]
    df["date_of_call"] = df["reporting_period"].apply(quarter_to_date)
    df["quarter"] = df["reporting_period"]
    df = df.dropna(subset=["quarter", "final_topic"])

    period_order = (
        df.groupby("quarter")["date_of_call"]
        .min()
        .sort_values()
        .index.tolist()
    )

    topic_summary = (
        df.groupby(["quarter", "final_topic"])
        .agg(count=("final_topic", "size"), avg_sentiment=("sentiment", "mean"))
        .reset_index()
    )

    topic_summary["total"] = topic_summary.groupby("quarter")["count"].transform("sum")
    topic_summary["proportion"] = topic_summary["count"] / topic_summary["total"]
    topic_summary["quarter"] = pd.Categorical(topic_summary["quarter"], categories=period_order, ordered=True)

    pivot_df = topic_summary.pivot(index="quarter", columns="final_topic", values="proportion").fillna(0)
    sentiment_lookup = topic_summary.set_index(["quarter", "final_topic"])["avg_sentiment"]

    topics = list(pivot_df.columns)
    pastel_colors = get_neutral_pastel_palette(len(topics))
    topic_color_map = dict(zip(topics, pastel_colors))

    fig = go.Figure()
    bottom = pd.Series(0, index=pivot_df.index)

    for topic in topics:
        heights = pivot_df[topic]
        fig.add_trace(go.Bar(
            x=pivot_df.index,
            y=heights,
            name=topic,
            marker=dict(color=topic_color_map[topic]),
            offsetgroup=0,
            base=bottom,
            hoverinfo="x+y+name",
        ))

        # Add sentiment annotations
        for i, quarter in enumerate(pivot_df.index):
            h = heights[quarter]
            if h > 0.03:
                sentiment = sentiment_lookup.get((quarter, topic), 0)
                color = sentiment_text_color(sentiment)
                fig.add_annotation(
                    x=quarter,
                    y=bottom[quarter] + h / 2,
                    text=f"{sentiment:+.2f}",
                    showarrow=False,
                    font=dict(size=10, color=color)
                )
        bottom += heights

    fig.update_layout(
        barmode="stack",
        title=f"{default_bank} – Topic Proportions per Quarter<br><sub>(Analyst question topic + Bank response sentiment)</sub>",
        xaxis_title="Quarter",
        yaxis_title="Proportion of Questions",
        legend_title="Topics",
        height=600,
        margin=dict(t=80, b=60),
    )

    return fig, bank_options, default_bank

@callback(
    Output("sentiment-distribution", "figure"),
    Output("sentiment-bank-dropdown", "options"),
    Output("sentiment-bank-dropdown", "value"),
    Output("sentiment-topic-dropdown", "options"),
    Output("sentiment-topic-dropdown", "value"),
    Input("sentiment-bank-dropdown", "value"),
    Input("sentiment-topic-dropdown", "value")
)
def q_a_sentiment(bank_selected, topic_selected):
    _, _, _, df = read_insights_data()

    # Get dropdown options
    bank_options = sorted(df["bank"].dropna().unique())
    default_bank = bank_selected or bank_options[0]

    df = df[df["bank"] == default_bank]

    topic_options = sorted(df["final_topic"].dropna().unique())
    default_topic = topic_selected or topic_options[0]

    df = df[df["final_topic"] == default_topic]

    if df.empty:
        return go.Figure(), bank_options, default_bank, topic_options, default_topic

    df["quarter"] = df["reporting_period"]

    sentiment_trend = (
        df.groupby("quarter")["sentiment_label"]
        .value_counts()
        .unstack(fill_value=0)
        .sort_index()
    )

    fig = go.Figure()
    colors = px.colors.diverging.Portland
    sentiment_labels = list(sentiment_trend.columns)

    for label in sentiment_labels:
        color = SENTIMENT_COLOR_MAP.get(label.lower(), "gray")  # fallback if unknown
        fig.add_trace(go.Bar(
            x=sentiment_trend.index,
            y=sentiment_trend[label],
            name=label.capitalize(),  # nicer formatting
            marker=dict(color=color),
            showlegend=True
        ))


    fig.update_layout(
        barmode="stack",
        title=f"Sentiment Over Time – {default_bank} – Topic: {default_topic}",
        xaxis_title="Reporting Period",
        yaxis_title="Count",
        legend_title="Sentiment",
        height=500,
        showlegend=True,
        margin=dict(t=60, b=60)
    )

    return fig, bank_options, default_bank, topic_options, default_topic

@callback(
    Output("kl-topic-summary-cards", "children"),
    Input("kl-topic-bank-dropdown", "value"),
    Input("kl-topic-quarter-dropdown", "value")
)
def update_kl_topic_summary_cards(bank_selected, quarter_selected):
    _, _, _, df = read_insights_data()
    if not bank_selected or not quarter_selected:
        return []

    df = df[df["bank"] == bank_selected]

    pivot = df.groupby(['reporting_period', 'final_topic']).size().unstack(fill_value=0)
    pivot_prop = pivot.div(pivot.sum(axis=1), axis=0).sort_index()

    quarters = pivot_prop.index.tolist()
    if quarter_selected not in quarters or quarters.index(quarter_selected) == 0:
        return []

    prev = pivot_prop.loc[quarters[quarters.index(quarter_selected) - 1]]
    curr = pivot_prop.loc[quarter_selected]
    diff = (curr - prev).sort_values(ascending=False)

    # Categorize
    top_up = diff[diff > 0].nlargest(3)
    top_down = diff[diff < 0].nsmallest(3)
    near_zero = diff[diff.abs() < 0.005].abs().sort_values().head(3).index
    near_zero = diff.loc[near_zero]


    def make_card(title, items, color):
        return html.Div(
            style={
                "background": color,
                "padding": "1rem",
                "borderRadius": "8px",
                "flex": "1",
                "color": "white",
                "minHeight": "130px"
            },
            children=[
                html.H5(title, style={"marginBottom": "0.5rem"}),
                *[
                    html.Div(f"{'↑' if val > 0 else '↓' if val < 0 else '→'} {topic}: {val:+.2%}",
                             style={"fontFamily": "monospace"})
                    for topic, val in items.items()
                ]
            ]
        )

    return [
        make_card("↑ Top Increases", top_up, "#28a745"),
        make_card("↓ Top Decreases", top_down, "#dc3545"),
        make_card("→ No Change", near_zero, "#6c757d"),
    ]

@callback(
    Output("kl-topic-bank-dropdown", "options"),
    Output("kl-topic-bank-dropdown", "value"),
    Input("kl-divergence-graph", "figure")  # dummy input to trigger once
)
def populate_kl_topic_bank_dropdown(_):
    _, _, _, df = read_insights_data()

    bank_options = sorted(df["bank"].dropna().unique())
    if not bank_options:
        return [], None

    return [{"label": b, "value": b} for b in bank_options], bank_options[0]

@callback(
    Output("kl-topic-quarter-dropdown", "options"),
    Output("kl-topic-quarter-dropdown", "value"),
    Input("kl-topic-bank-dropdown", "value")
)
def populate_kl_quarter_dropdown(bank_selected):
    _, _, _, df = read_insights_data()

    if not bank_selected:
        return [], None

    # Filter for selected bank
    df = df[df["bank"] == bank_selected]

    # Get quarters where the bank has meaningful topic data
    pivot = df.groupby(['reporting_period', 'final_topic']).size().unstack(fill_value=0)
    pivot = pivot[pivot.sum(axis=1) > 0]  # only keep quarters with topic data

    quarter_list = list(pivot.index)

    if len(quarter_list) < 2:
        return [], None

    # Skip the first quarter (no previous to compare)
    return [{"label": q, "value": q} for q in quarter_list[1:]], quarter_list[-1]

@callback(
    Output("kl-divergence-graph", "figure"),
    Output("kl-bank-dropdown", "options"),
    Output("kl-bank-dropdown", "value"),
    Input("kl-bank-dropdown", "value")
)
def update_kl_drift(selected_banks):
    _, _, _, df = read_insights_data()

    bank_options = sorted(df["bank"].dropna().unique())
    default_banks = selected_banks or bank_options[:5]  # default to showing first 5 banks

    kl_results = []

    for bank, bank_df in df.groupby('bank'):
        if bank not in default_banks:
            continue

        pivot = bank_df.groupby(['reporting_period', 'final_topic']).size().unstack(fill_value=0)
        pivot_prop = pivot.div(pivot.sum(axis=1), axis=0)

        if len(pivot_prop) < 2:
            continue

        for i in range(1, len(pivot_prop)):
            prev_dist = pivot_prop.iloc[i - 1]
            curr_dist = pivot_prop.iloc[i]
            kl_val = np.sum(prev_dist * np.log((prev_dist + 1e-10) / (curr_dist + 1e-10)))
            kl_results.append({
                'bank': bank,
                'quarter': pivot_prop.index[i],
                'KL Divergence': kl_val
            })

    kl_df = pd.DataFrame(kl_results)

    # Convert and sort quarters
    def quarter_to_date(qr):
        try:
            q, y = qr.split('_')
            month = {'Q1': '03-31', 'Q2': '06-30', 'Q3': '09-30', 'Q4': '12-31'}[q]
            return pd.to_datetime(f"{y}-{month}")
        except:
            return pd.NaT

    kl_df['quarter_date'] = kl_df['quarter'].apply(quarter_to_date)
    kl_df = kl_df.dropna(subset=['quarter_date'])
    kl_df = kl_df.sort_values('quarter_date')

    # Plot
    fig = go.Figure()
    for bank in kl_df['bank'].unique():
        bank_df = kl_df[kl_df['bank'] == bank]
        fig.add_trace(go.Bar(
            x=bank_df['quarter'],
            y=bank_df['KL Divergence'],
            name=bank
        ))

    fig.update_layout(
        barmode="group",
        title="Topic Distribution Drift Over Time by Bank (KL Divergence)",
        xaxis_title="Quarter",
        yaxis_title="KL Divergence",
        legend_title="Bank",
        height=500,
        margin=dict(t=60, b=60)
    )

    return fig, bank_options, default_banks

@callback(
    Output("peer-sentiment-heatmap", "figure"),
    Input("kl-divergence-graph", "figure")  # dummy input to trigger on load
)
def update_peer_heatmap(_):
    _, _, _, df = read_insights_data()

    # Build bank-topic sentiment pivot table
    pivot = df.groupby(["bank", "final_topic"])["sentiment"].mean().unstack()

    if pivot.empty:
        return go.Figure()

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale='RdYlGn',
        zmid=0,
        colorbar=dict(title="Avg Sentiment"),
        hovertemplate="Bank: %{y}<br>Topic: %{x}<br>Avg Sentiment: %{z:.2f}<extra></extra>",
    ))

    fig.update_layout(
        title="Average Sentiment per Topic by Bank",
        xaxis_title="Topic",
        yaxis_title="Bank",
        height=500,
        margin=dict(t=60, l=100, r=40, b=100)
    )

    return fig



def generate_layout():

    df_topic_relevance, df_topic_relevance_weighted_sentiment, df_risk_category_mapping, df_qa = read_insights_data()

    # Get the list of banks from the topic relevance data
    all_banks_list = sorted(df_topic_relevance['bank'].unique().tolist())

    # Get the quarters from the topic relevance data and set up the slider
    quarters = df_topic_relevance['reporting_period'].unique().tolist()
    slider_values = sorted(quarters)
    slider_indices = list(range(len(quarters)))
    slider_marks = {i: {'label': j, 'style': {'transform': 'rotate(30deg)', 'padding-right': '0px'}}
             for i, j in zip(slider_indices, slider_values)}
    max_value = len(slider_indices) - 1

    time_data_dict = {}
    time_data_dict['slider_indices'] = slider_indices
    time_data_dict['slider_values'] = slider_values

    time_store = dcc.Store(id="time-data", data=time_data_dict)

    # Get risk categories
    risk_categories_list = sorted(df_risk_category_mapping['risk_category'].unique().tolist())

    qa_topics_card = html.Div(
        className="card",
        children=[
            html.H4("Q&A Topics", className="card-header"),
            html.P("Stacked topic proportions with sentiment annotations."),
            dcc.Dropdown(
                id="topic-bank-dropdown",  # 👈 This is the input
                multi=False,
                placeholder="Select a bank",
            ),
            dcc.Graph(
                id="topic-proportion-graph",  # 👈 This is the output
                config={"displayModeBar": False},
            ),
        ]
    )

    sentiment_card = html.Div(
        className="card",
        children=[
            html.H4("Q&A Sentiment Distribution", className="card-header"),
            html.P("Stacked sentiment distribution of analyst Q&A over time."),
            dcc.Dropdown(
                id="sentiment-bank-dropdown",
                placeholder="Select a bank",
                multi=False,
            ),
            dcc.Dropdown(
                id="sentiment-topic-dropdown",
                placeholder="Select a topic",
                multi=False,
            ),
            dcc.Graph(id="sentiment-distribution"),
        ]
    )


    kl_summary_cards = html.Div(
        id="kl-topic-summary-cards",
        className="card-group flex-div3",  # Adjust your layout class if needed
        style={"display": "flex", "gap": "1rem", "margin-top": "20px"},
        children=[]  # We'll fill these in via callback
    )
    kl_drilldown_card = html.Div(
        className="card",
        children=[
            html.H4("KL Topic Change Drilldown", className="card-header"),
            html.P("Topic shifts vs. previous quarter. Arrows show increase (↑), decrease (↓), or no change (→)."),
            html.Div(
                style={"display": "flex", "gap": "1rem", "marginBottom": "1rem"},
                children=[
                    dcc.Dropdown(id="kl-topic-bank-dropdown", placeholder="Select a bank", style={"flex": "1"}),
                    dcc.Dropdown(id="kl-topic-quarter-dropdown", placeholder="Select a quarter", style={"flex": "1"}),
                ]
            ),
            html.Div(
                id="kl-topic-summary-cards",
                className="card-group flex-div3",
                style={"display": "flex", "gap": "1rem", "marginBottom": "1rem"}
            ),
            html.Div(id="kl-topic-drift-list", style={"padding": "10px"})
        ]
    )

    kl_combined_card = html.Div(
        className="card",
        children=[
            html.H4("Topic Drift and Change Analysis", className="card-header"),
            html.P("KL divergence tracks overall topic drift. Below it, explore the top topic increases, decreases, and stable ones per quarter."),


            # KL Divergence chart
            dcc.Dropdown(
                id="kl-bank-dropdown",
                placeholder="Select banks to display",
                multi=True,
                style={"marginBottom": "1rem"},
            ),
            dcc.Graph(id="kl-divergence-graph"),

            html.Ul([
                html.Li("KL < 1: Stable topic distribution"),
                html.Li("1 ≤ KL ≤ 3: Moderate shift in focus"),
                html.Li("KL > 3: Major change in topical priorities")
            ], style={"marginBottom": "2rem"}),

            html.Div(
                style={"display": "flex", "gap": "1rem", "marginBottom": "1rem"},
                children=[
                    dcc.Dropdown(id="kl-topic-bank-dropdown", placeholder="Select a bank", style={"flex": "1"}),
                    dcc.Dropdown(id="kl-topic-quarter-dropdown", placeholder="Select a quarter", style={"flex": "1"}),
                ]
            ),
            
            # Summary cards for ↑ ↓ →
            html.Div(
                id="kl-topic-summary-cards",
                className="card-group flex-div3",
                style={"display": "flex", "gap": "1rem", "marginBottom": "1rem"}
            ),

            # Optional drilldown list
            html.Div(id="kl-topic-drift-list", style={"padding": "10px"})
        ]
    )

    
    kl_drift_card = html.Div(
        className="card",
        children=[
            html.H4("Topic Drift (KL Divergence)", className="card-header"),
            html.P("KL divergence measures the shift in topic distributions between consecutive quarters. Larger values suggest a bank’s focus or concern areas changed significantly."),
            dcc.Dropdown(
                id="kl-bank-dropdown",
                placeholder="Select banks to display",
                multi=True,
            ),
            dcc.Graph(id="kl-divergence-graph"),
            html.Ul([
                html.Li("KL < 1: Stable topic distribution"),
                html.Li("1 ≤ KL ≤ 3: Moderate shift in focus"),
                html.Li("KL > 3: Major change in topical priorities")
            ])

        ]
    )

    peer_comparison_card = html.Div(
        className="card",
        children=[
            html.H4("Peer Comparison: Average Sentiment by Topic", className="card-header"),
            html.P("Compare banks by average sentiment across topics."),
            dcc.Graph(id="peer-sentiment-heatmap")
        ]
    )









    page_layout = html.Div(
        children=[
            html.Div(
                className="card",
                id='card-summary',
                children=[
                    html.Div(
                        children=[
                            html.H4(
                                className="card-header",
                                children="Executive Summary"
                            ),
                            html.P(
                                className="explanation",
                                children="The table below provides a summary of trending risk factors across the G-SIBs."
                            ),
                            html.Br(),

                        ]
                    ),

                ]
            ),
            html.Div(
                className="card card-controls",
                id="curt-card-controls",
                children=[
                    html.H4(
                        className="card-header",
                        children="Controls"
                    ),
                    # html.Br(),
                    html.Div(
                        className='control-div',
                        children=[
                            html.Div(
                                className="control-dropdown",
                                children=[
                                    html.Label(
                                        className="control-label",
                                        children="Bank:",
                                        htmlFor='bank-dropdown-comp'
                                    ),
                                    dcc.Dropdown(
                                        id='bank-dropdown-comp',
                                        options=all_banks_list,
                                        value=all_banks_list,
                                        multi=True,
                                        clearable=False,
                                    ),
                                ]
                            ),
                            html.Div(
                                className="control-slider",
                                children=[
                                    html.Label(
                                        className="control-label",
                                        children="Date range:",
                                        htmlFor='date-slider-comp'
                                    ),
                                    # Consider persisting values
                                    dcc.RangeSlider(
                                        id='date-slider-comp',
                                        className='date-slider',
                                        max=max_value,
                                        step=1,
                                        marks=slider_marks,
                                        value=[0, max_value],
                                    ),
                                ]
                            ),
                        ],
                    )

                ]
            ),
            html.Div(
                className="card-group flex-div2",
                children=[
                    html.Div(
                        className="card",
                        id='left-card-curt',
                        children=[
                            html.Div(
                                children=[
                                    html.H4(
                                        className="card-header",
                                        children="Sentiment Trends"
                                    ),
                                    html.P(
                                        className="explanation",
                                        children="The chart below shows how net sentiment across different risk categories has been evolving over recent quarters."
                                    ),
                                    html.Label(
                                        className="control-label",
                                        children="Risk category:",
                                        htmlFor='risk-category-dropdown'
                                    ),
                                    dcc.Dropdown(
                                        id='risk-category-dropdown',
                                        options=risk_categories_list,
                                        value=risk_categories_list[0],
                                        clearable=False,
                                    ),
                                    dcc.Graph(
                                        # className='graphic',
                                        id='line-fig-sentiment',
                                    ),
                                ]
                            ),

                        ]
                    ),
                    html.Div(
                        className="card data-card",
                        id='right-card-curt',
                        children=[
                            html.Div(
                                children=[
                                    html.Div(
                                        children=[
                                            html.H4(
                                                children="Data Summary",
                                                className="data-card-heading",
                                                id="sentiment-eff-heading",
                                            ),
                                            html.H4(
                                                className="data-card-value",
                                                id="sentiment-eff",
                                            )
                                        ]
                                    ),
                                    html.Br(),
                                    html.Div(
                                        children=[
                                            html.H4(
                                                className="data-card-heading",
                                                id="sentiment-bank-heading",
                                            ),
                                            html.H4(
                                                className="data-card-value",
                                                id="sentiment-bank",
                                            )
                                        ]
                                    ),


                                ]

                            ),



                        ]
                    )
                ]
            ),
            qa_topics_card,
            html.Div(
                style={"display": "flex", "gap": "1rem"},
                children=[
                    html.Div(sentiment_card, style={"flex": "1"}),
                    html.Div(peer_comparison_card, style={"flex": "1"}),
                ]
            ),
            kl_combined_card,
                time_store,
        ]

    )

    return page_layout


def layout():
    return generate_layout()


@callback(
    Output(component_id='line-fig-sentiment', component_property='figure'),
    Input(component_id='bank-dropdown-comp', component_property='value'),
    Input(component_id='date-slider-comp', component_property='value'),
    Input(component_id='risk-category-dropdown', component_property='value'),
    State(component_id='time-data', component_property='data'),
    # prevent_initial_call=True,
    # background=True
)
def update_agg_figs(banks, date_range_indices, risk_category, time_data):
    # Read in the curtailment data
    df_topic_relevance, df_topic_relevance_weighted_sentiment, df_risk_category_mapping, _ = read_insights_data()

    # Filter the data based on the selected bank(s)
    df_topic_relevance_weighted_sentiment = df_topic_relevance_weighted_sentiment[df_topic_relevance_weighted_sentiment['bank'].isin(banks)]

    # Filter the data based on the selected date range
    all_quarters = time_data['slider_values']
    start_idx = date_range_indices[0]
    end_idx = date_range_indices[1]
    quarters_selected = all_quarters[start_idx:end_idx + 1]
    df_topic_relevance_weighted_sentiment = df_topic_relevance_weighted_sentiment[df_topic_relevance_weighted_sentiment['reporting_period'].isin(quarters_selected)]


    # Get risk category and subtopic column sets
    risk_category_cols = df_risk_category_mapping['risk_category'].unique().tolist()
    subtopic_cols = df_risk_category_mapping['subtopic'].unique().tolist()

    # Aggregate by bank and reporting period
    df_all_text_topics_relevance_sentiment_quarter_agg = df_topic_relevance_weighted_sentiment.groupby(
        ['bank', 'reporting_period']
    )[subtopic_cols + risk_category_cols].sum().reset_index()

    # Normalize the net sentiment scores
    df_all_text_topics_relevance_sentiment_quarter_agg_norm = df_all_text_topics_relevance_sentiment_quarter_agg.copy()
    df_all_text_topics_relevance_sentiment_quarter_agg_norm[subtopic_cols] = df_all_text_topics_relevance_sentiment_quarter_agg_norm[subtopic_cols].div(
        df_all_text_topics_relevance_sentiment_quarter_agg_norm[subtopic_cols].abs().mean(axis=1), axis=0
    )
    df_all_text_topics_relevance_sentiment_quarter_agg_norm[risk_category_cols] = df_all_text_topics_relevance_sentiment_quarter_agg_norm[risk_category_cols].div(
        df_all_text_topics_relevance_sentiment_quarter_agg_norm[risk_category_cols].abs().mean(axis=1), axis=0
    )

    # Reshape data for plotting
    df_topics_relevance_sentiment_quarter_plotting = df_all_text_topics_relevance_sentiment_quarter_agg_norm.pivot(
        columns='bank',
        index='reporting_period',
    )
    df_topics_relevance_sentiment_quarter_plotting.columns = [', '.join(col).strip() for col in df_topics_relevance_sentiment_quarter_plotting.columns.values]
    df_topics_relevance_sentiment_quarter_plotting.reset_index(inplace=True)

    # Generate a multiline chart for the topic relevance scores
    cols_to_plot = [col for col in df_topics_relevance_sentiment_quarter_plotting.columns if col.startswith(risk_category)]
    legend_labels = [col.split(",")[-1].strip() for col in cols_to_plot]

    # Generate the multiline chart
    fig_topics_relevance_sentiment_quarter = generate_multiline_chart(
        df=df_topics_relevance_sentiment_quarter_plotting,
        x_data_col='reporting_period',
        y_data_cols=cols_to_plot,
        x_title='Reporting Period',
        y_title='Net Sentiment Score (Normalized)',
        plot_title=f"Quarterly Net Sentiment Scores: {risk_category}",
        marker_size=5,
        line_legend_labels=legend_labels,
    )

    return fig_topics_relevance_sentiment_quarter

# Sample component for layout
proportion_card = html.Div(
    className="card",
    children=[
        html.H4("Topic Proportions per Quarter", className="card-header"),
        html.P("Select a bank to view topic distribution and sentiment evolution over time."),
        dcc.Dropdown(id="topic-bank-dropdown", multi=False),
        dcc.Graph(id="topic-proportion-graph")
    ]
)

