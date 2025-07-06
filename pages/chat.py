import time
import os

from dotenv import load_dotenv
import dash
from dash import dcc, html, callback, Output, Input, State, ctx, dash_table, clientside_callback, ClientsideFunction
import dash_mantine_components as dmc
from dash_socketio import DashSocketIO
from flask_socketio import SocketIO, emit
from openai import OpenAI
from openai.types.responses.response_text_delta_event import ResponseTextDeltaEvent
from dash_chat import ChatComponent

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)

dash.register_page(__name__,
                   path='/chat',
                   name='Chat',
                   title='G-SIB Risk Chat',
                   description='Risk Analysis Chatbot for G-SIBs',)

# app = dash.get_app()
# socketio = SocketIO(app.server)


# @socketio.on("connect")
# def on_connect():
#     print("Client connected")

# @socketio.on("disconnect")
# def on_disconnect():
#     print("Client disconnected")

# def notify(socket_id, message, color=None):
#     emit(
#         "notification",
#         dmc.Notification(
#             message=message,
#             action="show",
#             id=uuid.uuid4().hex,
#             color=color,
#         ).to_plotly_json(),
#         namespace="/",
#         to=socket_id,
#     )


def stream_llm_response(prompt):
    stream = client.responses.create(
        model="gpt-4o-mini",
        input=[{"role": "user", "content": prompt}],
        stream=True,
    )
    return stream

def generate_layout():

    page_layout = html.Div(
        children=[
            # html.Div(
            #     className="card",
            #     id='chat-card',
            #     children=[
            #         html.Div(
            #             children=[
            #                 html.H4(
            #                     className="card-header",
            #                     children="Chat"
            #                 ),
            #                 html.P(
            #                     className="explanation",
            #                     children="Start chatting!"
            #                 ),
            #                 html.Br(),

            #             ]
            #         ),
            #         ChatComponent(
            #             id="chat-component",
            #             messages=[],  # Initialize empty since we'll load from storage
            #             fill_height=True, # Not sure what effect this has

            #         ),
            #         html.Div(id="notification_wrapper"),
            #     ]
            # ),
            # html.Div(
            #     # className="card",
            #     id='chat-card',
            #     children=[
            #         # ChatComponent(
            #         #     id="chat-component",
            #         #     messages=[],  # Initialize empty since we'll load from storage
            #         #     fill_height=True, # Not sure what effect this has

            #         # ),
            #     ]
            # ),
            # dmc.MantineProvider(
            #     children=[dmc.NotificationProvider(position="top-right"),
            #     html.Div(id="notification_wrapper"),]
            # ),
            # DashSocketIO(id='socketio', eventNames=["notification", "stream"]),
            dcc.Store(id="chatHistoryLocal", storage_type="local", data=[]),
            dcc.Store(id="chatHistoryServer", data=[]),
            dcc.Store(id="dummy", data=""),
        ]

    )

    return page_layout


def layout():
    return generate_layout()

# @callback(
#     Output("chatHistoryServer", "data", allow_duplicate=True),
#     Output("user-updates", "data"),
#     Input("chat-component", "new_message"), # Always a user message
#     State("chatHistoryServer", "data"),
#     State("user-updates", "data"),
#     prevent_initial_call=True,
# )
def store_chat(new_message, existing_messages, user_updates):
    user_updates += 1
    if not new_message:
        return no_update
    updated_messages = existing_messages + [new_message]
    return updated_messages, user_updates


# @callback(
#     Output("chatHistoryServer", "data", allow_duplicate=True),
#     Input("user-updates", "data"),
#     State("chatHistoryServer", "data"),
#     State("socketio", "socketId"),
#     prevent_initial_call=True,
# )
def stream_response(user_updates, chat_history, socket_id):
    if not chat_history:
        return no_update

    latest_message = chat_history[-1]
    if latest_message["role"] != "user": # This condition should always be False and we move on
        return no_update

    # Collect full response from the LLM
    llm_response = []
    for event in stream_llm_response(latest_message["content"]):
        if isinstance(event, ResponseTextDeltaEvent):
            token = event.delta
            emit("stream", token, namespace="/", to=socket_id)
            time.sleep(0.03)  # Simulate delay for streaming effect
            llm_response.append(token)

    print(''.join(llm_response))
    bot_response = {"role": "assistant", "content": ''.join(llm_response)}

    return chat_history + [bot_response] # Update the server side chat history with the bot response

# clientside_callback(
#     ClientsideFunction(
#         namespace='clientside',
#         function_name='add_user_msg'
#     ),
#     Output("dummy", "data"),
#     Input("chat-component", "new_message"),
#     prevent_initial_call=True,
# )

# clientside_callback(
#     """(notification) => {
#         if (!notification) return dash_clientside.no_update
#         return notification
#     }""",
#     Output("notification_wrapper", "children", allow_duplicate=True),
#     Input("socketio", "data-notification"),
#     prevent_initial_call=True,
# )


# clientside_callback(
#     ClientsideFunction(
#         namespace='clientside',
#         function_name='build_response'
#     ),
#     Output("chat-component", "messages"),
#     Input("socketio", "data-stream"),
#     prevent_initial_call=True,
# )
