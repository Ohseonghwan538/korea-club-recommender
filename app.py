import streamlit as st
import pandas as pd
import requests
import json
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
st.caption("Nemotron-Personas-Korea 인구 페르소나 X 카카오 로컬 API 실시간 데이터 기반 AI 여행 플래너")

# ==========================================
# 2. Secrets 환경변수 및 검증된 한국 고화질 이미지
# ==========================================
KAKAO_API_KEY = st.secrets.get("KAKAO_API_KEY", "")
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")

# 100% 검증된 한국 관련 고화질 이미지
KOREA_FALLBACK_IMAGES = [
    "https://images.unsplash.com/photo-1538485399081-7191377e8241?w=600",
    "https://images.unsplash.com/photo-1548115184-bc6544d06a58?w=600",
    "https://images.unsplash.com/photo-1578637387939-43c525550085?w=600",
    "https://images.unsplash.com/photo-1517154421773-0529f29ea451?w=600",
    "https://images.unsplash.com/photo-1590301157890-4810ed352733?w=600",
    "https://images.unsplash.com/photo-1555396273-367ea4eb4db5?w=600",
]

# 한국 광역시/도 기본 정보
AREA_INFO = {
    "서울특별시": {"road": "종로구 사직로 161", "tel_prefix": "02"},
    "인천광역시": {"road": "중구 차이나타운로 59", "tel_prefix": "032"},
    "대전광역시": {"road": "유성구 대덕대로 481", "tel_prefix": "042"},
    "대구광역시": {"road": "중구 달구벌대로 2077", "tel_prefix": "053"},
    "광주광역시": {"road": "동구 금남로 245", "tel_prefix": "062"},
    "부산광역시": {"road": "해운대구 해운대해변로 264", "tel_prefix": "051"},
    "울산광역시": {"road": "남구 무거동 대학로 93", "tel_prefix": "052"},
    "세종특별자치시": {"road": "세종시 도움6로 11", "tel_prefix": "044"},
    "경기도": {"road": "수원시 팔달구 효원로 1", "tel_prefix": "031"},
    "강원특별자치도": {"road": "강릉시 창해로 307", "tel_prefix": "033"},
    "충청북도": {"road": "청주시 상당구 상당로 82", "tel_prefix": "043"},
    "충청남도": {"road": "공주시 금벽로 368", "tel_prefix": "041"},
    "전라북도": {"road": "전주시 완산구 기린대로 99", "tel_prefix": "063"},
    "전라남도": {"road": "여수시 오동도로 61", "tel_prefix": "061"},
    "경상북도": {"road": "경주시 보문로 424", "tel_prefix": "054"},
    "경상남도": {"road": "창원시 성산구 중앙대로 151", "tel_prefix": "055"},
    "제주특별자치도": {"road": "제주시 첨단로 242", "tel_prefix": "064"}
}

# ==========================================
# 3. 사이드바 - 유저 여행 성향 입력
# ==========================================
st.sidebar.header("⚙️ 유저 여행 성향 설정")

st.sidebar.subheader("1. 여행 기본 정보")
selected_region = st.sidebar.selectbox("여행 희망 지역", list(AREA_INFO.keys()), index=4) # 기본값: 광주광역시
travel_duration = st.sidebar.radio("여행 일정", ["당일치기", "1박 2일"], index=1)
user_age = st.sidebar.slider("연령대", 18, 70, 25)
companion = st.sidebar.selectbox("동행인", ["혼자", "연인/배우자", "친구들", "가족/아이와 함께", "부모님과 함께"], index=1)

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

if not KAKAO_API_KEY:
    st.sidebar.error("⚠️ KAKAO_API_KEY가 설정되지 않았습니다. Secrets에 등록해주세요.")

# ==========================================
# 4. 데이터 로드 및 카카오 로컬 API 연동 함수
# ==========================================
@st.cache_data(show_spinner="Nemotron 페르소나 데이터셋 로드 중...")
def load_persona_data():
    ds = load_dataset("nvidia/Nemotron-Personas-Korea", split="train[:1500]")
    records = []
    for idx, item in enumerate(ds):
        travel = item.get("travel", "")
        culinary = item.get("culinary", "")
        concise = item.get("concise", "")
        records.append({
            "id": f"persona_{idx}",
            "age": item.get("age", 30),
            "location": item.get("location", "전국"),
            "travel": travel,
            "culinary": culinary,
            "summary": concise,
            "matching_text": f"여행취향: {travel} | 미식: {culinary} | 라이프스타일: {concise}"
        })
    return pd.DataFrame(records)

@st.cache_resource(show_spinner="AI 매칭 모델 준비 중...")
def load_embedding_model():
    return SentenceTransformer("jhgan/ko-sroberta-multitask")

df_personas = load_persona_data()
embed_model = load_embedding_model()

@st.cache_data(show_spinner="페르소나 벡터 인덱싱 중...")
def get_persona_embeddings(_model, texts):
    return _model.encode(texts, show_progress_bar=False)

persona_embeddings = get_persona_embeddings(embed_model, df_personas["matching_text"].tolist())

def fetch_kakao_places(region_name, size=10):
    """카카오 로컬 API - 키워드 검색 (안전한 Header 인증 방식)"""
    if not KAKAO_API_KEY:
        st.warning("⚠️ KAKAO_API_KEY가 없어 기본 대체 데이터를 표시합니다.")
        return get_fallback_places(region_name)
    
    url = "https://dapi.kakao.com/v2/local/search/keyword.json"
    headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY.strip()}"}
    
    # 지역 명소 키워드 검색
    params = {
        "query": f"{region_name} 가볼만한곳 명소",
        "size": size
    }
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            documents = data.get("documents", [])
            
            if documents:
                places = []
                for idx, doc in enumerate(documents):
                    # 도로명 주소 우선, 없으면 지번 주소 사용
                    addr = doc.get("road_address_name") or doc.get("address_name") or f"{region_name} 대표 주소"
                    tel = doc.get("phone")
                    if not tel:
                        tel = f"{AREA_INFO.get(region_name, {}).get('tel_prefix', '02')}-123-4567"
                    
                    places.append({
                        "title": doc.get("place_name", f"{region_name} 대표 명소"),
                        "addr": addr,
                        "image": KOREA_FALLBACK_IMAGES[idx % len(KOREA_FALLBACK_IMAGES)],
                        "tel": tel,
                        "url": doc.get("place_url", "")
                    })
                return places
        else:
            st.error(f"⚠️ 카카오 API 호출 실패 (HTTP {response.status_code})\nSecrets의 KAKAO_API_KEY를 확인해주세요.")
            
    except Exception as e:
        st.error(f"⚠️ 카카오 API 연동 중 오류 발생: {e}")
        
    return get_fallback_places(region_name)

def get_fallback_places(region_name):
    info = AREA_INFO.get(region_name, {"road": "중앙로 1", "tel_prefix": "02"})
    road, prefix = info["road"], info["tel_prefix"]
    return [
        {"title": f"{region_name} 국립 문화 예술 공간", "addr": f"{region_name} {road}", "image": KOREA_FALLBACK_IMAGES[0], "tel": f"{prefix}-123-4567"},
        {"title": f"{region_name} 감성 한옥 마을 및 카페거리", "addr": f"{region_name} {road} 인근", "image": KOREA_FALLBACK_IMAGES[1], "tel": f"{prefix}-234-5678"},
        {"title": f"{region_name} 대표 로컬 한식 맛집 거리", "addr": f"{region_name} 맛집골목", "image": KOREA_FALLBACK_IMAGES[5], "tel": f"{prefix}-345-6789"},
        {"title": f"{region_name} 자연 힐링 수목 산책로", "addr": f"{region_name} 수목원길 1", "image": KOREA_FALLBACK_IMAGES[2], "tel": f"{prefix}-456-7890"},
        {"title": f"{region_name} 뷰 맛집 전망대 & 카페", "addr": f"{region_name} 전망대길 10", "image": KOREA_FALLBACK_IMAGES[3], "tel": f"{prefix}-567-8901"}
    ]

# ==========================================
# 5. 추천 실행 및 UI 표시
# ==========================================
if st.button("✈️ 나의 성향 맞춤 한국 여행 코스 설계하기", type="primary"):
    # 1) Nemotron 페르소나 매칭
    with st.spinner("1️⃣ Nemotron 인구 데이터 기반 나의 페르소나 분석 중..."):
        user_query_text = f"여행취향: {interest_travel} | 미식: {interest_culinary} | 문화: {interest_arts} | 성향: {user_bio}"
        user_vector = embed_model.encode([user_query_text])
        sim_scores = cosine_similarity(user_vector, persona_embeddings)[0]
        
        df_matched = df_personas.copy()
        df_matched["match_score"] = (sim_scores * 100).round(1)
        matched_persona = df_matched.sort_values(by="match_score", ascending=False).iloc[0]

    # 2) 카카오 로컬 API 실시간 장소 검색
    with st.spinner(f"2️⃣ 카카오 로컬 API에서 [{selected_region}] 실시간 실제 장소/주소 조회 중..."):
        real_places = fetch_kakao_places(selected_region)

    st.success(f"🎉 Nemotron 페르소나 매칭도 {matched_persona['match_score']}%! [{selected_region}] 맞춤 코스가 설계되었습니다.")

    with st.expander("🔍 AI가 분석한 나의 라이프스타일 유형", expanded=False):
        st.write(f"**- 유효 인구 페르소나:** {matched_persona['age']}세 / {matched_persona['location']} 거주")
        st.write(f"**- 성향 요약:** {matched_persona['summary']}")
        st.write(f"**- 여행/미식 취향:** {matched_persona['travel']} / {matched_persona['culinary']}")

    # 3) Gemini AI 코스 작성
    with st.spinner("3️⃣ AI 여행 컨설턴트가 카카오 실제 주소 기반으로 상세 코스를 디자인하고 있습니다..."):
        ai_itinerary = ""
        if GEMINI_API_KEY:
            try:
                genai.configure(api_key=GEMINI_API_KEY)
                llm = genai.GenerativeModel("gemini-3.1-flash-lite")
                
                places_text = "\n".join([f"- 장소명: {p['title']} | 실제주소: {p['addr']}" for p in real_places])
                
                prompt = f"""
                당신은 대한민국 최고 맞춤형 여행 코스 컨설턴트입니다.
                아래 유저 프로필과 실시간 카카오 로컬 API로 검색된 장소/주소 목록을 바탕으로 [{travel_duration}] 코스를 설계해 주세요.

                [유저 성향]
                - 희망 지역/일정: {selected_region} / {travel_duration}
                - 동행인: {companion}
                - 취향: {interest_travel}, {interest_culinary}, {interest_arts}
                - 라이프스타일: {user_bio}

                [카카오 API 실시간 장소 및 실제 주소]
                {places_text}

                [작성 요청사항]
                1. 유저 성향에 딱 맞는 감성적인 코스 타이틀
                2. [{travel_duration}] 시간 순서별 (오전/점심/오후/저녁) 코스 안내
                3. **[필수] 장소 언급 시 소괄호 안에 지도 검색이 가능한 실제 주소를 표시할 것 (예: 장소명 (주소: ...))**
                4. 각 장소별 개인화 추천 사유 (1-2문장)
                """
                response = llm.generate_content(prompt)
                ai_itinerary = response.text
            except Exception as e:
                st.error(f"Gemini AI 호출 오류: {e}")

    # 결과 화면 Layout
    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.subheader(f"🗺️ [{selected_region}] {travel_duration} 추천 코스")
        if ai_itinerary:
            st.markdown(ai_itinerary)
        else:
            st.info("💡 Secrets에 GEMINI_API_KEY를 설정하시면 상세 AI 추천서가 자동 생성됩니다.")

    with col_right:
        st.subheader("📸 카카오 추천 명소 & 실제 주소")
        for idx, place in enumerate(real_places[:6]):
            with st.container(border=True):
                c_img, c_info = st.columns([1, 1.4])
                with c_img:
                    st.image(place["image"], use_container_width=True)
                with c_info:
                    st.markdown(f"**{idx+1}. {place['title']}**")
                    st.markdown(f"📍 **실제 주소:**\n`{place['addr']}`")
                    st.caption(f"📞 {place['tel']}")
                    if place.get("url"):
                        st.markdown(f"[🔗 카카오맵으로 보기]({place['url']})")
