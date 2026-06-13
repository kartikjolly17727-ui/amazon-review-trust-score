import streamlit as st

from model import predict_review, train_model
from trust_score import build_trust_score


st.set_page_config(
    page_title="Amazon Review Trust Score",
    page_icon="?",
    layout="centered",
)


@st.cache_resource(show_spinner="Training review model...")
def load_model():
    return train_model()


st.title("Amazon Review Trust Score")
st.caption("Check whether a product review looks real or computer generated.")

try:
    model, metrics = load_model()
except Exception as exc:
    st.error(str(exc))
    st.stop()

review_text = st.text_area(
    "Paste an Amazon review",
    height=180,
    placeholder="Example: The product arrived quickly and worked exactly as described...",
)

analyze = st.button("Analyze review", type="primary")

if analyze:
    try:
        prediction = predict_review(model, review_text)
        trust_score = build_trust_score(prediction)
    except ValueError as exc:
        st.warning(str(exc))
    else:
        st.metric("Trust score", f"{trust_score['score']} / 100")
        st.subheader(trust_score["verdict"])
        st.write(trust_score["summary"])

        real_percent = trust_score["real_percent"]
        fake_percent = trust_score["fake_percent"]
        st.progress(trust_score["score"] / 100)
        st.write(f"Model real confidence: {real_percent}%")
        st.write(f"Model generated confidence: {fake_percent}%")

with st.expander("Model details"):
    st.write(f"Training reviews: {metrics['train_rows']}")
    st.write(f"Testing reviews: {metrics['test_rows']}")
    st.write(f"Test accuracy: {metrics['accuracy']:.2%}")
