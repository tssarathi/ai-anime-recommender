import streamlit as st

from src.pipeline.pipeline import AnimeRecommendationPipeline

st.set_page_config(
    page_title="Anime Recommendation System",
    page_icon=":tv:",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading...")
def load_pipeline():
    return AnimeRecommendationPipeline(persist_dir="data/gold/")


pipeline = load_pipeline()

st.title("Anime Recommendation System")
st.caption(
    "Are you an anime fan searching for your next binge? Use the chat below to get new recommendations."
)


if "chat" not in st.session_state:
    st.session_state.chat = [
        {
            "role": "assistant",
            "content": "Hi! I'm an anime recommender. How can I help you today?",
        }
    ]

query = st.chat_input(
    "What are your anime preferences? Ex: I like anime with a lot of action and comedy."
)

if query:
    st.session_state.chat = [
        {
            "role": "assistant",
            "content": "Hi! I'm an anime recommender. How can I help you today?",
        },
        {"role": "user", "content": query},
    ]

    with st.spinner("Generating recommendations..."):
        response, source_docs = pipeline.recommend(query)

    st.session_state.chat.append({"role": "assistant", "content": response})
    st.session_state.source_docs = source_docs

for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if "source_docs" in st.session_state and st.session_state.source_docs:
    with st.expander("View documents sent to the LLM"):
        for i, doc in enumerate(st.session_state.source_docs, 1):
            st.markdown(f"**Document {i}**")
            st.text(doc.page_content)
            if doc.metadata:
                st.caption(f"Metadata: {doc.metadata}")
            st.divider()
