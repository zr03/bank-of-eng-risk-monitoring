import dash
from dash import dcc, html, callback, Output, Input, State, ctx, dash_table

dash.register_page(__name__,
                   path='/chat',
                   name='Chat',
                   title='G-SIB Risk Chat',
                   description='Risk Analysis Chatbot for G-SIBs',)



def generate_layout():

    page_layout = html.Div(
        children=[
            html.Div(
                className="card",
                id='chat-card',
                children=[
                    html.Div(
                        children=[
                            html.H4(
                                className="card-header",
                                children="Chat"
                            ),
                            html.P(
                                className="explanation",
                                children="Start chatting!"
                            ),
                            html.Br(),

                        ]
                    ),

                ]
            ),
        ]

    )

    return page_layout


def layout():
    return generate_layout()


