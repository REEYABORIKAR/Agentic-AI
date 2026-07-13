import queue
import uuid
from datetime import datetime

import streamlit as st
from langgraph_mcp_backend import chatbot, retrieve_all_threads, submit_async_task
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

# =========================================================================
#  Page config (must be the first Streamlit call)
# =========================================================================
st.set_page_config(
    page_title="LangGraph MCP Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================================
#  Global styling
# =========================================================================
st.markdown(
    """
    <style>
        /* Tighten default padding */
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 900px;
        }

        /* Header */
        .app-header {
            display: flex;
            align-items: center;
            gap: 0.6rem;
            margin-bottom: 0.25rem;
        }
        .app-header h1 {
            font-size: 1.6rem;
            margin: 0;
        }
        .app-subtitle {
            color: #8a8f98;
            font-size: 0.92rem;
            margin-bottom: 1.2rem;
        }

        /* Sidebar thread buttons */
        section[data-testid="stSidebar"] .stButton button {
            text-align: left;
            justify-content: flex-start;
            border-radius: 8px;
            font-size: 0.85rem;
            padding: 0.45rem 0.7rem;
        }
        section[data-testid="stSidebar"] .stButton button:hover {
            border-color: #6c63ff;
            color: #6c63ff;
        }

        /* New chat button emphasis */
        div[data-testid="stSidebarUserContent"] > div:first-child .stButton button {
            background-color: #6c63ff;
            color: white;
            border: none;
            font-weight: 600;
        }
        div[data-testid="stSidebarUserContent"] > div:first-child .stButton button:hover {
            background-color: #5a52d5;
            color: white;
        }

        /* Chat bubbles get a little breathing room */
        div[data-testid="stChatMessage"] {
            padding: 0.35rem 0;
        }

        .thread-caption {
            color: #8a8f98;
            font-size: 0.75rem;
            margin: 0.6rem 0 0.2rem 0.2rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }

        .empty-state {
            text-align: center;
            color: #8a8f98;
            padding: 3rem 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

AVATARS = {"user": "🧑‍💻", "assistant": "🤖"}


# =========================================================================
#  Utilities
# =========================================================================
def generate_thread_id():
    return uuid.uuid4()


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])


def thread_label(thread_id, messages=None):
    """Build a short, human-friendly label for a thread."""
    if messages:
        for msg in messages:
            if isinstance(msg, HumanMessage) and msg.content:
                text = msg.content.strip().replace("\n", " ")
                return text[:32] + ("…" if len(text) > 32 else "")
    return f"Chat {str(thread_id)[:8]}"


def switch_thread(thread_id):
    st.session_state["thread_id"] = thread_id
    messages = load_conversation(thread_id)

    temp_messages = []
    for msg in messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        temp_messages.append({"role": role, "content": msg.content})
    st.session_state["message_history"] = temp_messages


# =========================================================================
#  Session initialization
# =========================================================================
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

if "thread_previews" not in st.session_state:
    st.session_state["thread_previews"] = {}

add_thread(st.session_state["thread_id"])

# =========================================================================
#  Sidebar
# =========================================================================
with st.sidebar:
    st.markdown("### 🤖 LangGraph MCP Chatbot")
    st.caption("Multi-tool assistant, powered by LangGraph + MCP")

    if st.button("➕  New chat", use_container_width=True):
        reset_chat()
        st.rerun()

    st.markdown('<div class="thread-caption">Conversations</div>', unsafe_allow_html=True)

    threads = st.session_state["chat_threads"][::-1]
    if not threads:
        st.caption("No conversations yet.")
    else:
        for thread_id in threads:
            is_active = thread_id == st.session_state["thread_id"]
            label = st.session_state["thread_previews"].get(thread_id)
            if label is None:
                # Only load history lazily / once per thread to keep the sidebar snappy
                try:
                    label = thread_label(thread_id, load_conversation(thread_id))
                except Exception:
                    label = thread_label(thread_id)
                st.session_state["thread_previews"][thread_id] = label

            icon = "💬" if is_active else "🗨️"
            if st.button(
                f"{icon} {label}",
                key=f"thread_{thread_id}",
                use_container_width=True,
                type="secondary" if not is_active else "primary",
            ):
                switch_thread(thread_id)
                st.rerun()

    st.markdown("---")
    with st.expander("⚙️ Session info"):
        st.caption(f"Thread ID: `{str(st.session_state['thread_id'])[:13]}...`")
        st.caption(f"Messages: {len(st.session_state['message_history'])}")

# =========================================================================
#  Main header
# =========================================================================
st.markdown(
    """
    <div class="app-header"><h1>LangGraph MCP Chatbot</h1></div>
    <div class="app-subtitle">Ask anything — I can search the web, check stock prices, and use connected MCP tools.</div>
    """,
    unsafe_allow_html=True,
)

# =========================================================================
#  Render history
# =========================================================================
if not st.session_state["message_history"]:
    st.markdown(
        """
        <div class="empty-state">
            <div style="font-size:2.2rem;">💬</div>
            <div>Start a conversation below.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    for message in st.session_state["message_history"]:
        with st.chat_message(message["role"], avatar=AVATARS.get(message["role"])):
            st.markdown(message["content"])

user_input = st.chat_input("Type your message…")

if user_input:
    # Show user's message
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar=AVATARS["user"]):
        st.markdown(user_input)

    # Update the sidebar preview for the current (possibly first) message
    st.session_state["thread_previews"][st.session_state["thread_id"]] = thread_label(
        None, [HumanMessage(content=user_input)]
    )

    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {"thread_id": st.session_state["thread_id"]},
        "run_name": "chat_turn",
    }

    # Assistant streaming block
    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        status_holder = {"box": None}
        error_holder = {"err": None}

        def ai_only_stream():
            event_queue: queue.Queue = queue.Queue()

            async def run_stream():
                try:
                    async for message_chunk, metadata in chatbot.astream(
                        {"messages": [HumanMessage(content=user_input)]},
                        config=CONFIG,
                        stream_mode="messages",
                    ):
                        event_queue.put((message_chunk, metadata))
                except Exception as exc:
                    event_queue.put(("error", exc))
                finally:
                    event_queue.put(None)

            submit_async_task(run_stream())

            while True:
                item = event_queue.get()
                if item is None:
                    break
                message_chunk, metadata = item
                if message_chunk == "error":
                    error_holder["err"] = metadata
                    break

                # Lazily create & update the SAME status container when any tool runs
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                # Stream ONLY assistant tokens
                if isinstance(message_chunk, AIMessage) and message_chunk.content:
                    yield message_chunk.content

        try:
            ai_message = st.write_stream(ai_only_stream())
        except Exception as exc:
            ai_message = None
            error_holder["err"] = exc

        # Finalize tool status
        if status_holder["box"] is not None:
            if error_holder["err"] is None:
                status_holder["box"].update(
                    label="✅ Tool finished", state="complete", expanded=False
                )
            else:
                status_holder["box"].update(
                    label="⚠️ Tool step interrupted", state="error", expanded=False
                )

        if error_holder["err"] is not None:
            st.error(f"Something went wrong while generating a response: {error_holder['err']}")

    # Save assistant message (only if we actually got one)
    if ai_message:
        st.session_state["message_history"].append(
            {"role": "assistant", "content": ai_message}
        )