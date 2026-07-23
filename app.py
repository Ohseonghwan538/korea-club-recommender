import streamlit as st
import pandas as pd
import requests
import json
import random
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai

# ==========================================
# 1. 페이지 설정 및 기본 레이아웃
# ==========================================
st.set_page_config(
    page_title="취향 콕! 성향 맞춤 한국 여행 코스 추천 시뮬레이터",
    page_icon="✈️",
    layout="wide"
)

st.title("✈️ 성향 맞춤 한국 여행 코스 추천 시뮬레이터")
st.caption("Nemotron-Personas-Korea 인구 페르소나 X 한국관광공사 TourAPI 4.0 실시간 데이터 기반 AI 여행 플래너")

# ==========================================
# 2. Secrets 환경변수 확인
# ==========================================
TOUR_API_KEY = st.secrets.get("TOUR_API_KEY", "")
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")

# 지역 코드 매핑 (한국관광공사 TourAPI 기준)
AREA_CODES = {
    "서울특별시": "1", "인천광역시": "2", "대전광역시": "3", "대구광역시": "4",
    "광주광역시": "5", "부산광역시": "6", "울산광역시": "7", "세종특별자치시": "8",
    "경기도": "31", "강원특별자치도": "32", "충청북도": "33", "충청남도": "34",
    "전라북도": "35", "전라남도": "36", "경상북도": "37", "경상남도": "38", "제주특별자치도": "39"
}

# ==========================================
# 3. 사이드바 - 유저 여행 성향 입력
# ==========================================
st.sidebar.header("⚙️ 유저 여행 성향 설정")

st.sidebar.subheader("1. 여행 기본 정보")
selected_region = st.sidebar.selectbox("여행 희망 지역", list(AREA_CODES.keys()), index=0)
travel_duration = st.sidebar.radio("여행 일정", ["당일치기", "1박 2일"], index=0)
user_age = st.sidebar.slider("연령대", 18, 70, 28)
companion = st.sidebar.selectbox("동행인", ["혼자", "연인/배우자", "친구들", "가족/아이와 함께", "부모님과 함께"])

st.sidebar.subheader("2. 세부 취향 (관심사)")
interest_travel = st.sidebar.text_input("🏕️ 여행 스타일", "힐링, 오션뷰 카페, 트레킹")
interest_culinary = st.sidebar.text_input("🍱 미식 / 식음료", "현지 로컬 맛집, 정갈한 한식, 디저트")
interest_arts = st.sidebar.text_input("🖼️ 문화 / 예술", "미술관, 한옥 마을, 역사 탐방")
interest_sports = st.sidebar.text_input("🚴 액티비티 / 레포츠", "가벼운 산책, 해양 레저")

st.sidebar.subheader("3. 성향 요약")
user_bio = st.sidebar.text_area(
    "나의 라이프스타일 & 여행관",
    "너무 빽빽한 일정보다는 여유롭게 풍경을 감상하고, 맛있는 음식을 먹으며 힐링하는 여행을 좋아합니다."
)

# Secrets 미설정시 안내 박스
if not TOUR_API_KEY or not GEMINI_API_KEY:
    st.sidebar.warning(
        "💡 Secrets 설정 필요:\n"
        "- TOUR_API_KEY (한국관광공사 API Key)\n"
        "- GEMINI_API_KEY (Google Gemini API Key)\n"
        "설정되지 않은 경우 시뮬레이션용 데모 데이터로 작동합니다."
    )

# ==========================================
# 4. 데이터 로드 및 한국관광공사 TourAPI 연동 함수
# ==========================================
@st.cache_data(show_spinner="Nemotron 페르소나 데이터셋을 로드 중입니다...")
def load_persona_data():
    ds = load_dataset("nvidia/Nemotron-Personas-Korea", split="train[:1500]")
    records = []
    for idx, item in enumerate(ds):
        travel = item.get("travel", "")
        culinary = item.get("culinary", "")
        concise = item.get("concise", "")
        matching_text = f"여행취향: {travel} | 미식: {culinary} | 라이프스타일: {concise}"
        records.append({
            "id": f"persona_{idx}",
            "age": item.get("age", 30),
            "location": item.get("location", "전국"),
            "travel": travel,
            "culinary": culinary,
            "summary": concise,
            "matching_text": matching_text
        })
    return pd.DataFrame(records)

@st.cache_resource(show_spinner="AI 매칭 임베딩 모델을 준비 중입니다...")
def load_embedding_model():
    return SentenceTransformer("jhgan/ko-sroberta-multitask")

df_personas = load_persona_data()
embed_model = load_embedding_model()

@st.cache_data(show_spinner="페르소나 벡터 인덱스를 생성 중입니다...")
def get_persona_embeddings(_model, texts):
    return _model.encode(texts, show_progress_bar=False)

persona_embeddings = get_persona_embeddings(embed_model, df_personas["matching_text"].tolist())

def fetch_tour_api_places(area_code, num_rows=30):
    """한국관광공사 국문 관광정보 서비스 (KorService1) - areaBasedList1 연동"""
    if not TOUR_API_KEY:
        return get_mock_places(selected_region)
    
    url = "http://apis.data.go.kr/B551011/KorService1/areaBasedList1"
    params = {
        "serviceKey": TOUR_API_KEY,
        "numOfRows": num_rows,
        "pageNo": 1,
        "MobileOS": "ETC",
        "MobileApp": "PersonaTravelApp",
        "_type": "json",
        "areaCode": area_code,
        "arrange": "O" # 대표이미지 있는 항목 우선
    }
    
    try:
        response = requests.get(url, params=params, timeout=8)
        if response.status_code == 200:
            data = response.json()
            items = data.get("response", {}).get("body", {}).get("items", {}).get("item", [])
            if items:
                cleaned_places = []
                for item in items:
                    cleaned_places.append({
                        "title": item.get("title", "관광지"),
                        "addr": item.get("addr1", "주소 정보 없음"),
                        "image": item.get("firstimage", "https://via.placeholder.com/400x250?text=No+Image"),
                        "content_type": item.get("contenttypeid", "12"),
                        "tel": item.get("tel", "")
                    })
                return cleaned_places
    except Exception as e:
        st.error(f"TourAPI 호출 중 오류 발생 (데모 데이터로 대체): {e}")
    
    return get_mock_places(selected_region)

def get_mock_places(region_name):
    """API Key 미설정 또는 장애 시 사용하는 가상 장소 데이터"""
    return [
        {"title": f"{region_name} 대표 힐링 수목원", "addr": f"{region_name} 중앙로 102", "image": "https://images.unsplash.com/photo-1507525428034-b723cf961d3e?w=500", "content_type": "12", "tel": "02-123-4567"},
        {"title": f"{region_name} 감성 오션뷰/전망 카페", "addr": f"{region_name} 해안도로 45", "image": "https://images.unsplash.com/photo-1554118811-1e0d58224f24?w=500", "content_type": "39", "tel": "02-234-5678"},
        {"title": f"{region_name} 로컬 한식 미식 거리", "addr": f"{region_name} 맛집길 12", "image": "https://images.unsplash.com/photo-1555396273-367ea4eb4db5?w=500", "content_type": "39", "tel": "02-345-6789"},
        {"title": f"{region_name} 역사 & 문화 미술관", "addr": f"{region_name} 문화로 88", "image": "https://images.unsplash.com/photo-1518998053901-5348d3961a04?w=500", "content_type": "14", "tel": "02-456-7890"},
        {"title": f"{region_name} 야경 명소 & 산책로", "addr": f"{region_name} 야경길 77", "image": "https://images.unsplash.com/photo-1519501025264-65ba15a82390?w=500", "content_type": "12", "tel": "02-567-8901"}
    ]

# ==========================================
# 5. 추천 실행 및 화면 구성
# ==========================================
if st.button("✈️ 나의 성향 맞춤 여행 코스 설계하기", type="primary"):
    with st.spinner("1️⃣ Nemotron 인구 데이터 기반 나의 페르소나 매칭 중..."):
        user_query_text = f"여행취향: {interest_travel} | 미식: {interest_culinary} | 문화: {interest_arts} | 성향: {user_bio}"
        user_vector = embed_model.encode([user_query_text])
        sim_scores = cosine_similarity(user_vector, persona_embeddings)[0]
        
        df_personas_matched = df_personas.copy()
        df_personas_matched["match_score"] = (sim_scores * 100).round(1)
        matched_persona = df_personas_matched.sort_values(by="match_score", ascending=False).iloc[0]

    with st.spinner(f"2️⃣ 한국관광공사 TourAPI에서 [{selected_region}] 실시간 관광/맛집 정보 조회 중..."):
        area_code = AREA_CODES.get(selected_region, "1")
        real_places = fetch_tour_api_places(area_code)

    st.success(f"🎉 Nemotron 페르소나 매칭도 {matched_persona['match_score']}%! [{selected_region}] 맞춤 코스가 생성되었습니다.")

    # 매칭된 페르소나 브리핑
    with st.expander("🔍 AI가 분석한 나의 한국인 페르소나 유형", expanded=False):
        st.write(f"**- 가장 가까운 인구 페르소나:** {matched_persona['age']}세 / {matched_persona['location']} 거주")
        st.write(f"**- 페르소나 라이프스타일:** {matched_persona['summary']}")
        st.write(f"**- 선호 여행/미식 스타일:** {matched_persona['travel']} / {matched_persona['culinary']}")

    # Gemini LLM 기반 여행 코스 구성
    with st.spinner("3️⃣ AI 여행 플래너가 실시간 관광지 데이터로 여행 코스를 디자인하고 있습니다..."):
        ai_itinerary = ""
        if GEMINI_API_KEY:
            try:
                genai.configure(api_key=GEMINI_API_KEY)
                llm = genai.GenerativeModel("gemini-3.1-flash-lite")
                places_text = "\n".join([f"- {p['title']} ({p['addr']})" for p in real_places[:12]])
                
                prompt = f"""
                당신은 국내 최고 맞춤형 여행 코스 컨설턴트입니다.
                아래 유저 프로필과 한국관광공사 실시간 장소 목록을 바탕으로 [{travel_duration}] 완벽 맞춤 코스를 구성해 주세요.

                [유저 성향]
                - 여행지/일정: {selected_region} / {travel_duration}
                - 동행인: {companion}
                - 여행/미식 취향: {interest_travel}, {interest_culinary}, {interest_arts}
                - 라이프스타일: {user_bio}
                - 매칭된 한국인 페르소나 특징: {matched_persona['summary']}

                [한국관광공사 실시간 장소 후보]
                {places_text}

                [작성 요청사항]
                1. 유저의 성향에 어울리는 감성적인 여행 타이틀 (예: "여유와 미식이 함께하는 {selected_region} 힐링 로드")
                2. [{travel_duration}] 시간 순서별 (오전/점심/오후/저녁 등) 상세 동선 및 코스 구성
                3. 각 장소별로 "왜 이 유저의 성향과 어울리는지" 1-2문장 명확한 추천 이유 작성
                4. 이동 동선 및 미식 팁 제공
                """
                response = llm.generate_content(prompt)
                ai_itinerary = response.text
            except Exception as e:
                st.error(f"Gemini AI 코스 생성 중 오류: {e}")

    # UI 출력
    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.subheader(f"🗺️ [{selected_region}] {travel_duration} 추천 여행 코스")
        if ai_itinerary:
            st.markdown(ai_itinerary)
        else:
            st.info("""
            **[기본 추천 동선 예시]**
            * **오전 (10:00):** 지역 대표 힐링 산책로 탐방 (자연 속 힐링)
            * **점심 (12:30):** 로컬 맛집 거리에서 한상 차림 미식 체험
            * **오후 (14:30):** 감성 오션뷰/전망 카페에서 여유로운 티타임
            * **저녁 (17:30):** 야경 명소 및 문화 거리 산책
            *(💡 Secrets에 GEMINI_API_KEY를 등록하시면 실시간 AI 코스 설명서가 자동 작성됩니다.)*
            """)

    with col_right:
        st.subheader("📸 코스 포함 대표 관광지 / 맛집 정보")
        for idx, place in enumerate(real_places[:5]):
            with st.container(border=True):
                c_img, c_info = st.columns([1, 1.5])
                with c_img:
                    st.image(place["image"], use_container_width=True)
                with c_info:
                    st.markdown(f"**{idx+1}. {place['title']}**")
                    st.caption(f"📍 {place['addr']}")
                    if place['tel']:
                        st.caption(f"📞 {place['tel']}")

else:
    st.info("👈 사이드바에서 여행 희망 지역과 성향을 선택한 후 **'나의 성향 맞춤 여행 코스 설계하기'** 버튼을 누르면 추천이 시작됩니다.")
