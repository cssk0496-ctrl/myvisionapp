import streamlit as st
import joblib
from PIL import Image
import numpy as np
import requests
from io import BytesIO

# --------------------------------------------------
# 기본 설정
# --------------------------------------------------
st.set_page_config(
    page_title="불량 검사 AI 모델 데모",
    page_icon="🔍",
    layout="wide"
)

FEATURE_SIZE = (64, 160)

# GitHub에 저장된 모델의 Raw 주소로 변경하세요.
MODEL_FILENAME = "lesson06_vision_model.joblib"
)


# --------------------------------------------------
# 특징 추출 및 품질 측정 함수
# --------------------------------------------------
def quality_metrics(image):
    a = np.asarray(
        image.resize(FEATURE_SIZE),
        dtype=np.float32
    )

    gx = np.diff(a, axis=1, prepend=a[:, :1])
    gy = np.diff(a, axis=0, prepend=a[:1, :])

    lap = (
        -4 * a
        + np.roll(a, 1, 0)
        + np.roll(a, -1, 0)
        + np.roll(a, 1, 1)
        + np.roll(a, -1, 1)
    )

    return {
        "brightness": float(a.mean()),
        "contrast": float(a.std()),
        "sharpness": float(lap.var()),
        "mean_gradient": float(np.hypot(gx, gy).mean())
    }


def extract_features(image):
    a = np.asarray(
        image.resize(
            FEATURE_SIZE,
            Image.Resampling.BILINEAR
        ),
        dtype=np.float32
    ) / 255.0

    gx = np.diff(a, axis=1, prepend=a[:, :1])
    gy = np.diff(a, axis=0, prepend=a[:1, :])

    mag = np.hypot(gx, gy)
    ori = (np.degrees(np.arctan2(gy, gx)) + 180) % 180

    hog = []
    bins = np.linspace(0, 180, 10)

    for row in range(0, 160, 8):
        for col in range(0, 64, 8):
            hist, _ = np.histogram(
                ori[row:row + 8, col:col + 8],
                bins=bins,
                weights=mag[row:row + 8, col:col + 8]
            )
            hog.extend(hist / (hist.sum() + 1e-6))

    intensity, _ = np.histogram(
        a,
        bins=16,
        range=(0, 1),
        density=True
    )

    percentiles = np.percentile(
        a,
        [1, 5, 25, 50, 75, 95, 99]
    )

    extra = [
        a.mean(),
        a.std(),
        mag.mean(),
        np.percentile(mag, 90),
        np.percentile(mag, 99)
    ]

    return np.concatenate([
        hog,
        intensity,
        percentiles,
        extra
    ])


# --------------------------------------------------
# GitHub 모델 로드
# --------------------------------------------------
@st.cache_resource(show_spinner="GitHub에서 AI 모델을 불러오는 중입니다...")
def load_model_bundle():
    try:
        response = requests.get(
            MODEL_URL,
            timeout=60
        )
        response.raise_for_status()

        # Git LFS 포인터 파일이 내려온 경우 확인
        if response.content.startswith(
            b"version https://git-lfs.github.com/spec"
        ):
            raise RuntimeError(
                "실제 모델 파일이 아닌 Git LFS 포인터가 내려왔습니다. "
                "GitHub Release 또는 일반 파일 다운로드 주소를 사용해주세요."
            )

        return joblib.load(BytesIO(response.content))

    except requests.RequestException as error:
        raise RuntimeError(
            f"GitHub에서 모델 파일을 다운로드하지 못했습니다: {error}"
        ) from error

    except Exception as error:
        raise RuntimeError(
            f"모델 파일을 불러오는 중 오류가 발생했습니다: {error}"
        ) from error


try:
    model_bundle = load_model_bundle()

    required_keys = {
        "model",
        "operating_threshold",
        "quality_limits",
        "class_names"
    }

    missing_keys = required_keys - set(model_bundle.keys())

    if missing_keys:
        raise KeyError(
            f"모델 번들에 필요한 항목이 없습니다: {sorted(missing_keys)}"
        )

    model = model_bundle["model"]
    operating_threshold = float(
        model_bundle["operating_threshold"]
    )
    quality_limits = model_bundle["quality_limits"]
    class_names = model_bundle["class_names"]

except Exception as error:
    st.error(str(error))
    st.info(
        "MODEL_URL이 GitHub의 Raw 파일 주소인지 확인해주세요."
    )
    st.stop()


def quality_ok(q):
    return (
        quality_limits["brightness_low"]
        <= q["brightness"]
        <= quality_limits["brightness_high"]
        and q["contrast"]
        >= quality_limits["contrast_low"]
        and q["sharpness"]
        >= quality_limits["sharpness_low"]
    )


# --------------------------------------------------
# Streamlit 화면
# --------------------------------------------------
st.title("🔍 불량 검사 AI 모델 데모")
st.write(
    "새로운 이미지를 업로드하면 AI가 불량 여부와 "
    "이미지 품질을 분석합니다."
)

uploaded_file = st.file_uploader(
    "검사할 이미지 파일을 선택하세요.",
    type=["jpg", "jpeg", "png", "bmp"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("L")
    except Exception:
        st.error("이미지를 열 수 없습니다. 정상적인 이미지 파일인지 확인해주세요.")
        st.stop()

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("업로드된 이미지")
        st.image(
            image,
            caption="업로드된 원본 이미지",
            use_container_width=True
        )

    with col2:
        st.subheader("분석 결과")

        try:
            features = extract_features(image).reshape(1, -1)
            quality = quality_metrics(image)

            probabilities = model.predict_proba(features)[0]
            defect_probability = float(probabilities[1])

            prediction_label = (
                1
                if defect_probability >= operating_threshold
                else 0
            )

            predicted_class = class_names[prediction_label]
            gate_pass = quality_ok(quality)

            if not gate_pass:
                routing_status = "RECAPTURE_OR_HUMAN_REVIEW"
            elif prediction_label == 1:
                routing_status = "DEFECT_CANDIDATE_REVIEW"
            else:
                routing_status = "POLICY_PASS"

            st.metric(
                "불량 확률",
                f"{defect_probability:.1%}"
            )

            st.markdown(
                f"**운영 임계값:** `{operating_threshold:.2f}`"
            )
            st.markdown(
                f"**AI 판정:** `{predicted_class}`"
            )
            st.markdown(
                f"**품질 게이트 통과:** `{gate_pass}`"
            )
            st.markdown(
                f"**최종 라우팅 상태:** `{routing_status}`"
            )

            st.subheader("이미지 품질 지표")

            metric_col1, metric_col2 = st.columns(2)

            with metric_col1:
                st.metric(
                    "밝기",
                    f"{quality['brightness']:.2f}"
                )
                st.metric(
                    "선명도",
                    f"{quality['sharpness']:.2f}"
                )

            with metric_col2:
                st.metric(
                    "대비",
                    f"{quality['contrast']:.2f}"
                )
                st.metric(
                    "평균 경사도",
                    f"{quality['mean_gradient']:.2f}"
                )

            if not gate_pass:
                st.warning(
                    "이미지 품질이 기준에 미달하여 "
                    "재촬영 또는 사람의 검토가 필요합니다."
                )
            elif prediction_label == 1:
                st.info(
                    "불량 후보로 감지되었습니다. 추가 검토가 필요합니다."
                )
            else:
                st.success("정상 이미지로 판정되었습니다.")

        except Exception as error:
            st.error(f"이미지 분석 중 오류가 발생했습니다: {error}")
