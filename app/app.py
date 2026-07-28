import requests

# GitHub Raw 파일 링크로 변경 (예시: 'YOUR_USERNAME', 'YOUR_REPO', 'main' 또는 'master' 브랜치, 'models' 폴더 등)
# 실제 경로와 파일명을 맞춰주세요.
GITHUB_MODEL_URL = "https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/models/lesson06_vision_model.joblib"
LOCAL_MODEL_PATH = "lesson06_vision_model.joblib" # Streamlit 앱이 실행되는 환경에 저장될 이름

# 모델 파일 다운로드
@st.cache_resource # Streamlit에서 모델을 한 번만 로드하도록 캐시
def load_model_from_github():
    try:
        response = requests.get(GITHUB_MODEL_URL)
        response.raise_for_status() # HTTP 오류 발생 시 예외 발생
        with open(LOCAL_MODEL_PATH, "wb") as f:
            f.write(response.content)
        return joblib.load(LOCAL_MODEL_PATH)
    except requests.exceptions.RequestException as e:
        st.error(f"모델 파일을 다운로드할 수 없습니다: {e}")
        st.stop()
    except Exception as e:
        st.error(f"모델을 로드하는 중 오류가 발생했습니다: {e}")
        st.stop()

model_bundle = load_model_from_github()
model = model_bundle["model"]
operating_threshold = model_bundle["operating_threshold"]
quality_limits = model_bundle["quality_limits"]
class_names = model_bundle["class_names"]
