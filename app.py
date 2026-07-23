import streamlit as st
import pandas as pd
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai

# -----------------------------------------------------------------------------
# 1. 페이지 및 기본 레이아웃 설정
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="모여라! 취향 맞춤 동호회 추천 플랫폼",
    page_icon="🧩",
    layout="wide"
)

st.title("🧩 나의 성향 맞춤 동호회 찾기")
st.caption("Hugging Face Nemotron-Personas-Korea 데이터셋 기반 AI 동호회 매칭 시뮬레이터")

# -----------------------------------------------------------------------------
# 2. 사이드바 - 유저 프로필 입력
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ 유저 프로필 & 설정")

# Gemini API Key (선택)
gemini_api_key = st.sidebar.text_input("Gemini API Key (선택: AI 맞춤 추천서)", type="password")

st.sidebar.subheader("1. 기본 정보")
user_region = st.sidebar.selectbox(
    "거주 / 활동 지역",
    ["전체", "서울특별시", "경기도", "인천광역시", "부산광역시", "대구광역시", "대전광역시", "광주광역시", "울산광역시", "강원특별자치도", "충청북도", "충청남도", "전라북도", "전라남도", "경상북도", "경상남도", "제주특별자치도"]
)
user_age = st.sidebar.slider("연령대", 18, 70, 28)

st.sidebar.subheader("2. 관심 분야")
interest_sports = st.sidebar.text_input("⚽ 스포츠 / 운동", "러닝, 플러깅, 등산")
interest_arts = st.sidebar.text_input("🎨 예술 / 문화", "전시회 관람, 사진 촬영")
interest_travel = st.sidebar.text_input("✈️ 여행 / 야외활동", "주말 캠핑, 드라이브")
interest_culinary = st.sidebar.text_input("☕ 음식 / 카페 / 술", "수제맥주, 맛집 탐방")

st.sidebar.subheader("3. 나의 성향 요약")
user_bio = st.sidebar.text_area(
    "라이프스타일 및 모임 성향",
    "퇴근 후나 주말에 가볍게 사람들과 어울리고, 운동 후 맛있는 소주나 맥주 한잔하는 것을 즐깁니다. 친근하고 부담 없는 분위기를 원해요."
)

# -----------------------------------------------------------------------------
# 3. 데이터셋 및 임베딩 모델 로드 (캐싱 적용)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner="한국인 페르소나 데이터셋을 로드하는 중입니다...")
def load_persona_data():
    # Streamlit Cloud 메모리 한계를 고려하여 2,000건 샘플링 로드
    ds = load_dataset("nvidia/Nemotron-Personas-Korea", split="train[:2000]")
    records = []
    for idx, item in enumerate(ds):
        interests = []
        if item.get("sports"): interests.append(f"스포츠: {item['sports']}")
        if item.get("arts"): interests.append(f"예술: {item['arts']}")
        if item.get("travel"): interests.append(f"여행: {item['travel']}")
        if item.get("culinary"): interests.append(f"음식: {item['culinary']}")
        
        interest_text = " / ".join(interests) if interests else "일상 대화 및 친목"
        concise = item.get("concise", "")
        location = item.get("location", "전국")
        age = item.get("age", 30)
        
        # 벡터 검색 대상 텍스트 생성
        matching_text = f"지역: {location} | 연령: {age}세 | 관심사: {interest_text} | 성향: {concise}"
        
        records.append({
            "id": f"persona_{idx}",
            "age": age,
            "location": location,
            "occupation": item.get("occupation", "직장인"),
            "interests": interest_text,
            "summary": concise,
            "matching_text": matching_text
        })
    return pd.DataFrame(records)

@st.cache_resource(show_spinner="AI 매칭 임베딩 모델을 준비하는 중입니다...")
def load_embedding_model():
    return SentenceTransformer("jhgan/ko-sroberta-multitask")

df_personas = load_persona_data()
embed_model = load_embedding_model()

@st.cache_data(show_spinner="페르소나 벡터 인덱스를 생성 중입니다...")
def get_persona_embeddings(_model, texts):
    return _model.encode(texts, show_progress_bar=False)

persona_embeddings = get_persona_embeddings(embed_model, df_personas["matching_text"].tolist())

# -----------------------------------------------------------------------------
# 4. 동호회 매칭 및 결과 출력
# -----------------------------------------------------------------------------
if st.button("🚀 나에게 꼭 맞는 동호회 찾기", type="primary"):
    with st.spinner("유저 성향 분석 및 동호회 매칭 진행 중..."):
        # 1. 유저 쿼리 벡터 변환
        user_query_text = f"지역: {user_region} | 연령: {user_age}세 | 관심사: {interest_sports}, {interest_arts}, {interest_travel}, {interest_culinary} | 성향: {user_bio}"
        user_vector = embed_model.encode([user_query_text])
        
        # 2. 유사도 계산
        sim_scores = cosine_similarity(user_vector, persona_embeddings)[0]
        
        df_result = df_personas.copy()
        df_result["match_score"] = (sim_scores * 100).round(1)
        
        # 3. 하드 필터링 (지역 필터)
        if user_region != "전체":
            df_filtered = df_result[df_result["location"].str.contains(user_region, na=False)]
            if len(df_filtered) >= 3:
                df_result = df_filtered
        
        top_matches = df_result.sort_values(by="match_score", ascending=False).head(5)
        
        st.success("🎯 회원님의 성향과 가장 매칭률이 높은 동호회를 찾았습니다!")
        
        col1, col2 = st.columns([2, 1])
        
        # 동호회 매칭 리스트
        with col1:
            st.subheader("👥 추천 동호회 TOP 3")
            for idx, (_, row) in enumerate(top_matches.head(3).iterrows()):
                with st.container(border=True):
                    c1, c2 = st.columns([4, 1])
                    with c1:
                        st.markdown(f"### 🏆 {idx+1}위 추천 동호회 (적합도: `{row['match_score']}%`)")
                        st.markdown(f"**📍 주요 활동지:** {row['location']} | **👤 주요 연령대:** {row['age']}세 ({row['occupation']})")
                        st.markdown(f"**🎨 동호회 메인 관심사:** {row['interests']}")
                        st.markdown(f"**💬 모임 성향/분위기:** {row['summary']}")
                    with c2:
                        st.metric(label="매칭률", value=f"{row['match_score']}%")
                        if st.button(f"가입 신청 #{idx+1}", key=f"btn_{idx}"):
                            st.toast(f"'{row['interests']}' 동호회에 가입 신청 메시지를 보냈습니다!", icon="🎉")

        # LLM 추천서
        with col2:
            st.subheader("🤖 AI 맞춤 동호회 추천서")
            top_match = top_matches.iloc[0]
            
            if gemini_api_key:
                try:
                    genai.configure(api_key=gemini_api_key)
                    llm = genai.GenerativeModel("gemini-1.5-flash")
                    prompt = f"""
                    당신은 국내 최고 맞춤형 동호회 매칭 전문가입니다.
                    유저 프로필과 매칭 1위 동호회 정보를 바탕으로 설레는 초대장과 추천 사유를 작성해주세요.

                    [유저 프로필]
                    - 연령/지역: {user_age}세 / {user_region}
                    - 관심사: {interest_sports}, {interest_arts}, {interest_travel}, {interest_culinary}
                    - 성향: {user_bio}

                    [매칭 1위 동호회]
                    - 지역: {top_match['location']}
                    - 관심사: {top_match['interests']}
                    - 분위기: {top_match['summary']}
                    - 매칭률: {top_match['match_score']}%

                    [작성 가이드]
                    1. 동호회 이름 제안 (센스있고 위트있게)
                    2. 이 동호회를 추천하는 이유 3가지
                    3. 첫 모임 추천 활동 코스
                    """
                    response = llm.generate_content(prompt)
                    st.info(response.text)
                except Exception as e:
                    st.error(f"Gemini API 호출 중 오류 발생: {e}")
            else:
                st.warning("👈 사이드바에 Gemini API Key를 입력하면 AI가 생성한 맞춤형 초대장을 받아보실 수 있습니다.")
                st.markdown(f"""
                **[1위 매칭 동호회 요약]**
                * **모임 스타일:** {top_match['location']} {top_match['interests'].split('/')[0]} 크루
                * **추천 사유:** 입력하신 라이프스타일과 동호회 멤버들의 성향 유사도가 **{top_match['match_score']}%**로 가장 높습니다.
                """)
else:
    st.info("👈 왼쪽 사이드바에서 프로필 정보를 입력하고 **'나에게 꼭 맞는 동호회 찾기'** 버튼을 누르면 매칭이 시작됩니다.")