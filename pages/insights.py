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

import app_config as config

dash.register_page(__name__,
                   path='/',
                   name='Insights',
                   title='G-SIB Risk Insights',
                   description='Risk insights for G-SIBs',)

TOPIC_RELEVANCE_FPATH = config.TOPIC_RELEVANCE_FPATH
TOPIC_RELEVANCE_WEIGHTED_SENTIMENT_FPATH = config.TOPIC_RELEVANCE_WEIGHTED_SENTIMENT_FPATH
RISK_CATEGORY_MAPPING_FPATH = config.RISK_CATEGORY_MAPPING_FPATH
COLOURS_LIST = px.colors.qualitative.Pastel

def read_insights_data():
    df_topic_relevance = pd.read_parquet(TOPIC_RELEVANCE_FPATH)
    df_topic_relevance_weighted_sentiment = pd.read_parquet(TOPIC_RELEVANCE_WEIGHTED_SENTIMENT_FPATH)
    df_risk_category_mapping = pd.read_parquet(RISK_CATEGORY_MAPPING_FPATH)

    return df_topic_relevance, df_topic_relevance_weighted_sentiment, df_risk_category_mapping

def read_summary_data():
    pass

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

def generate_layout():

    df_topic_relevance, df_topic_relevance_weighted_sentiment, df_risk_category_mapping = read_insights_data()

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
    df_topic_relevance, df_topic_relevance_weighted_sentiment, df_risk_category_mapping = read_insights_data()

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