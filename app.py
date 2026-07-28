"""
비전 품질 검사 PoC - Streamlit App

배포 구조 (GitHub: visionapp)
    visionapp/
    ├── app.py
    ├── requirements.txt
    └── lesson06_vision_model.joblib
"""

import io
import os
import tempfile
import urllib.request
from pathlib import Path

import joblib
import numpy as np
import streamlit as st
from PIL import Image

# --- 0. 경로 설정 --------------------------------------------------------
# __file__ 기준 상대경로 → 로컬/Cloud 어디서 실행해도 동일하게 동작
APP_DIR = Path(__file__).resolve().parent
MODEL_FILENAME = "lesson06_vision_model.joblib"
MODEL_PATH = APP_DIR / MODEL_FILENAME

REQUIRED_KEYS = {
    "model",
    "feature_size",
    "operating_threshold",
    "quality_limits",
    "class_names",
}


def _get_model_url() -> str:
    """모델 파일을 리포지토리에 직접 못 넣는 경우(용량 등)를 위한 폴백 URL."""
    try:
        if "MODEL_URL" in st.secrets:
            return str(st.secrets["MODEL_URL"])
    except Exception:
        pass  # secrets.toml 이 없으면 무시
    return os.environ.get("MODEL_URL", "")


# --- 1. 모델 로드 (캐시) -------------------------------------------------
@st.cache_resource(show_spinner="모델을 불러오는 중입니다...")
def load_model_bundle():
    path = MODEL_PATH

    if not path.exists():
        url = _get_model_url()
        if not url:
            raise FileNotFoundError(
                f"'{MODEL_FILENAME}' 을(를) 찾을 수 없습니다.\n"
                f"확인한 경로: {path}\n\n"
                "리포지토리 루트(app.py와 같은 위치)에 모델 파일을 커밋했는지, "
                "또는 Secrets에 MODEL_URL 을 설정했는지 확인하세요."
            )
        # 리포지토리 외부에서 내려받는 경우 임시 디렉터리에 저장
        path = Path(tempfile.gettempdir()) / MODEL_FILENAME
        if not path.exists():
            urllib.request.urlretrieve(url, path)

    bundle = joblib.load(path)

    missing = REQUIRED_KEYS - set(bundle)
    if missing:
        raise KeyError(f"모델 번들에 다음 키가 없습니다: {sorted(missing)}")
    return bundle


try:
    model_bundle = load_model_bundle()
except Exception as exc:  # 사용자에게 원인을 그대로 보여줌
    st.set_page_config(page_title="비전 품질 검사 PoC", layout="wide")
    st.title("비전 품질 검사 PoC")
    st.error(f"모델 로드 실패\n\n{exc}")
    st.stop()

model = model_bundle["model"]
FEATURE_SIZE = tuple(model_bundle["feature_size"])  # (width, height)
OPERATING_THRESHOLD = float(model_bundle["operating_threshold"])
quality_limits = model_bundle["quality_limits"]
class_names = model_bundle["class_names"]


# --- 2. 특징 추출 / 품질 게이트 ------------------------------------------
# ※ 학습 노트북과 반드시 동일한 로직이어야 합니다. 수정 금지.
def quality_metrics(image: Image.Image) -> dict:
    a = np.asarray(image.resize(FEATURE_SIZE), dtype=np.float32)
    gx = np.diff(a, axis=1, prepend=a[:, :1])
    gy = np.diff(a, axis=0, prepend=a[:1, :])
    lap = -4 * a + np.roll(a, 1, 0) + np.roll(a, -1, 0) + np.roll(a, 1, 1) + np.roll(a, -1, 1)
    return {
        "brightness": float(a.mean()),
        "contrast": float(a.std()),
        "sharpness": float(lap.var()),
        "mean_gradient": float(np.hypot(gx, gy).mean()),
    }


def extract_features(image: Image.Image) -> np.ndarray:
    a = np.asarray(
        image.resize(FEATURE_SIZE, Image.Resampling.BILINEAR), dtype=np.float32
    ) / 255
    gx = np.diff(a, axis=1, prepend=a[:, :1])
    gy = np.diff(a, axis=0, prepend=a[:1, :])
    mag = np.hypot(gx, gy)
    ori = (np.degrees(np.arctan2(gy, gx)) + 180) % 180

    hog, bins = [], np.linspace(0, 180, 10)
    for row in range(0, FEATURE_SIZE[1], 8):
        for col in range(0, FEATURE_SIZE[0], 8):
            hist, _ = np.histogram(
                ori[row:row + 8, col:col + 8],
                bins=bins,
                weights=mag[row:row + 8, col:col + 8],
            )
            hog.extend(hist / (hist.sum() + 1e-6))

    intensity, _ = np.histogram(a, bins=16, range=(0, 1), density=True)
    percentiles = np.percentile(a, [1, 5, 25, 50, 75, 95, 99])
    extra = [
        a.mean(),
        a.std(),
        mag.mean(),
        np.percentile(mag, 90),
        np.percentile(mag, 99),
    ]
    return np.concatenate([hog, intensity, percentiles, extra])


def quality_ok(q: dict) -> bool:
    return (
        quality_limits["brightness_low"] <= q["brightness"] <= quality_limits["brightness_high"]
        and q["contrast"] >= quality_limits["contrast_low"]
        and q["sharpness"] >= quality_limits["sharpness_low"]
    )


# --- 3. UI ---------------------------------------------------------------
st.set_page_config(page_title="비전 품질 검사 PoC", layout="wide")
st.title("비전 품질 검사 PoC")
st.caption("이미지를 업로드하면 불량 여부를 판정합니다.")

with st.sidebar:
    st.header("설정")
    threshold = st.slider(
        "판정 임계값",
        min_value=0.0,
        max_value=1.0,
        value=OPERATING_THRESHOLD,
        step=0.01,
        help=f"모델 기본 운영 임계값: {OPERATING_THRESHOLD:.4f}",
    )
    st.divider()
    st.caption(f"입력 크기: {FEATURE_SIZE[0]}×{FEATURE_SIZE[1]}")
    st.caption(f"클래스: {', '.join(map(str, class_names))}")

uploaded_file = st.file_uploader(
    "이미지를 업로드하세요", type=["jpg", "jpeg", "png", "bmp"]
)

if uploaded_file is None:
    st.info("이미지를 업로드하여 판정을 시작해주세요.")
    st.stop()

image = Image.open(io.BytesIO(uploaded_file.getvalue())).convert("L")

col1, col2 = st.columns([1, 1])
with col1:
    st.image(image, caption="업로드된 이미지 (Grayscale)", use_container_width=True)

with col2:
    if not st.button("이미지 판정하기", type="primary"):
        st.stop()

    features = extract_features(image)
    q = quality_metrics(image)

    probability = float(model.predict_proba(features.reshape(1, -1))[0, 1])
    prediction = int(probability >= threshold)
    predicted_class = class_names[prediction]

    gate_pass = quality_ok(q)
    routing_status = (
        "RECAPTURE_OR_HUMAN_REVIEW"
        if not gate_pass
        else "DEFECT_CANDIDATE_REVIEW"
        if prediction
        else "POLICY_PASS"
    )

    st.subheader("판정 결과")

    color = "#d62728" if str(predicted_class).upper() == "DEFECT" else "#2ca02c"
    st.markdown(
        f"### <span style='color:{color}'>{predicted_class}</span>",
        unsafe_allow_html=True,
    )

    m1, m2 = st.columns(2)
    m1.metric("불량 확률", f"{probability:.4f}")
    m2.metric("품질 게이트", "통과" if gate_pass else "미달")

    st.write(f"**라우팅 상태:** `{routing_status}`")
    st.progress(min(max(probability, 0.0), 1.0))

    st.markdown("#### 품질 기준")
    st.table(
        {
            "항목": ["밝기", "대비", "선명도", "평균 그래디언트"],
            "측정값": [
                f"{q['brightness']:.2f}",
                f"{q['contrast']:.2f}",
                f"{q['sharpness']:.2f}",
                f"{q['mean_gradient']:.2f}",
            ],
            "기준": [
                f"{quality_limits['brightness_low']:.2f} ~ {quality_limits['brightness_high']:.2f}",
                f"≥ {quality_limits['contrast_low']:.2f}",
                f"≥ {quality_limits['sharpness_low']:.2f}",
                "-",
            ],
        }
    )
