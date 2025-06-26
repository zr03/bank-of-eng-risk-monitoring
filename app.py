import os
import datetime

import dash
from dash import CeleryManager, DiskcacheManager
from dash import html, dcc
from celery import Celery

APP_ENV = os.environ.get("DASH_APP_NAME", "dev")
print(f"IMPORTANT: App is running in the {APP_ENV} environment.")
EXTERNAL_SCRIPTS = [
    {
        'src': "https://cdnjs.cloudflare.com/ajax/libs/jquery/3.1.1/jquery.min.js"
    },
    {
        'src': "https://cdnjs.cloudflare.com/ajax/libs/twitter-bootstrap/3.3.7/js/bootstrap.min.js"
    },
]

EXTERNAL_STYLESHEETS = [
    {
        'href': "https://cdnjs.cloudflare.com/ajax/libs/bootswatch/3.3.7/paper/bootstrap.min.css",
        'rel': "stylesheet"
    },
]

WINDOWS = "nt"
if os.name == WINDOWS:
    path_prefix = '/'
else:
    dash_app_name = os.environ.get('DASH_APP_NAME',None)
    if dash_app_name is not None:
        path_prefix = f"/{dash_app_name}/"
    elif dash_app_name is None :
        path_prefix = '/'

app = dash.Dash(
    __name__,
    use_pages=True,
    external_scripts=EXTERNAL_SCRIPTS,
    external_stylesheets=EXTERNAL_STYLESHEETS,
    suppress_callback_exceptions=True,
    assets_folder="./assets",
    # assets_ignore="(custom_script.js)",  # Added as a deferred script
    # background_callback_manager=background_callback_manager
)

server = app.server  # expose server variable for Procfile

background = html.Div(
    className="background_img"
)

header = html.Div(
    className="app-header",
    children=[
        html.Div(id='script-container'),
        html.H1(
            className="app-header-title",
            children='RADAR',
        ),
        html.H3(
            className="app-header-subtitle",
            children='G-SIB Risk Insights',
            style={'font-family': 'Quicksand'}
        ),
    ]
)

header_nav = html.Div(
    className="app-header-nav",
    children=[
        html.H1(
            className="app-header-title-nav",
            children='RADAR',
        ),
    ]
)


sidebar = html.Nav(
    className="navbar navbar-inverse navbar-fixed-left",
    children=[
        # html.A(
        #     id="left-slide",
        #     children=[
        #         caret_left
        #     ]
        # ),
        html.Div(
            className="container",
            children=[
                html.Div(
                    id="scroll-pane",
                    className="navbar-collapse collapse",
                    children=[
                        html.Div(
                            className="side",
                            children=[
                                html.Div(
                                    className="dropdown",
                                    children=[
                                        html.A(
                                            href=path_prefix+'',
                                            children=[
                                                html.Span(className="helper"),
                                                # html.Div(className="sidebar_home_logo"),
                                                html.Img(
                                                    className="home_icon_svg",
                                                    id="home_icon",
                                                    title="Home Page"
                                                ),
                                                html.Br(),
                                                # html.Span("Home")
                                            ]
                                        )
                                    ]
                                )
                            ]
                        ),

                        html.Div(
                            className="side",
                            children=[
                                html.Div(
                                    className="dropdown",
                                    children=[
                                        html.A(
                                            href=path_prefix+'leaderboard',
                                            className="dropdown-toggle",
                                            # **{"data-toggle": "dropdown",
                                            #    "aria-haspopup": "true"},
                                            children=[
                                                html.Span(className="helper"),
                                                html.Img(
                                                    className="laep_icon_svg",
                                                    id="laep_icon",
                                                    title="LLM Leaderboard"
                                                ),
                                                # html.Span(
                                                #     className="caret"
                                                # )
                                            ]
                                        ),
                                    ]
                                )
                            ]
                        ),

                    ]
                )
            ]
        ),
        # html.A(
        #     id="right-slide",
        #     children=[
        #         caret_right
        #     ]
        # ),
    ]
)


app.layout = html.Div(
    className="wrapper",
    children=[
        html.Div(
            className="navigation",
            children=[
                header_nav,
                sidebar,
            ]
        ),
        html.Main(
            className="main",
            children=[
                background,
                header,
                # sidebar,
                html.Div(
                    className="content",
                    children=[
                        dash.page_container
                    ]
                )
            ],
        ),
    ]
)


if __name__ == '__main__':
    if APP_ENV=="dev":
        app.run(host="localhost", debug=True,use_reloader=True, threaded=False)
    else:
        app.run(debug=False, threaded=False)

