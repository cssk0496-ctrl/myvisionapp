import io
import os
import tempfile
import urllib.request
from pathlib import Path

import joblib
import numpy as np
import streamlit as st
from PIL import Image

# 반드시 첫 번째 Streamlit 명령으로 실행
st.set_page_config(
    page_title="비전 품질 검사 PoC",
    page_icon="🔍",
    layout="wide"
)

APP_DIR = Path(__file__).resolve().parent
MODEL_FILENAME = "lesson06_vision_model.joblib"
MODEL_PATH = APP_DIR / MODEL_FILENAME

REQUIRED_KEYS = {
    "model",
    "operating_threshold",
    "quality_limits",
    "class_names",
}


def _get_model_url() -> str:
    """모델 파일이 저장소에 없을 경우 사용할 다운로드 주소."""
    try:
        return str(st.secrets.get("MODEL_URL", ""))
    except Exception:
        return os.environ.get("MODEL_URL", "")


@st.cache_resource(show_spinner="모델을 불러오는 중입니다...")
def load_model_bundle():
    path = MODEL_PATH

    if not path.exists():
        url = _get_model_url()

        if not url:
            raise FileNotFoundError(
                f"{MODEL_FILENAME} 파일을 찾을 수 없습니다.\n\n"
                f"확인한 경로: {path}\n\n"
                "app.py와 같은 위치에 모델 파일을 올리거나 "
                "Streamlit Secrets에 MODEL_URL을 설정해주세요."
            )

        path = Path(tempfile.gettempdir()) / MODEL_FILENAME

        try:
            urllib.request.urlretrieve(url, path)
        except Exception as error:
            raise RuntimeError(
                f"모델 다운로드에 실패했습니다: {error}"
            ) from error

    # Git LFS 포인터 파일인지 검사
    try:
        with path.open("rb") as file:
            first_bytes = file.read(100)

        if b"git-lfs.github.com/spec" in first_bytes:
            raise RuntimeError(
                "GitHub에서 실제 모델이 아닌 Git LFS 포인터 파일을 "
                "가져왔습니다. 모델의 직접 다운로드 주소가 필요합니다."
            )
    except RuntimeError:
        raise
    except Exception as error:
        raise RuntimeError(
            f"모델 파일 확인 중 오류가 발생했습니다: {error}"
        ) from error

    try:
        bundle = joblib.load(path)
    except Exception as error:
        raise RuntimeError(
            f"joblib 모델을 읽지 못했습니다: {error}"
        ) from error

    if not isinstance(bundle, dict):
        raise TypeError(
            "모델 파일의 최상위 데이터가 dictionary 형식이 아닙니다."
        )

    missing = REQUIRED_KEYS - set(bundle.keys())

    if missing:
        raise KeyError(
            f"모델 번들에 다음 키가 없습니다: {sorted(missing)}\n"
            f"현재 저장된 키: {sorted(bundle.keys())}"
        )

    return bundle


try:
    model_bundle = load_model_bundle()
except Exception as exc:
    st.title("비전 품질 검사 PoC")
    st.error(f"모델 로드 실패\n\n{exc}")
    st.stop()


model = model_bundle["model"]

# 기존 모델 파일에 feature_size가 없으면 기본값 적용
FEATURE_SIZE = tuple(
    model_bundle.get("feature_size", (64, 160))
)

OPERATING_THRESHOLD = float(
    model_bundle["operating_threshold"]
)

quality_limits = model_bundle["quality_limits"]
class_names = model_bundle["class_names"]
