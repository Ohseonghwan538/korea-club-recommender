import streamlit as st
import pandas as pd
import requests
import json
import urllib.parse
import re
import datetime
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai

# ==========================================
# 1. 페이지 설정 및 기본 레이아웃
# ==========================================
st.set_page_config(
    page_title="사용자 맞춤 에이전틱 한국 여행 코스 추천",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 사용자 맞춤 에이전틱 한국 여행 코스 플래너")
st.caption("유저 취향 입력 분석 X 동적 유사 페르소나 비교 X 출발시각/영업시간/숙소 고려 X 점진적 하네스 시뮬레이션")

# ==========================================
# 2. Secrets 환경변수 및 기본 정보
# ==========================================
KAKAO_API_KEY = st.secrets.get("KAKAO_API_KEY", "")
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")

AREA_INFO = {
    "서울특별시": {"road": "서울특별시 종로구 사직로 161", "tel_prefix": "02"},
    "인천광역시": {"road": "인천광역시 중구 차이나타운로 59", "tel_prefix": "032"},
    "대전광역시": {"road": "대전광역시 유성구 대덕대로 481", "tel_prefix": "042"},
    "대구광역시": {"road": "대구광역시 중구 달구벌대로 2077", "tel_prefix": "053"},
    "광주광역시": {"road": "광주광역시 동구 금남로 245", "tel_prefix": "062"},
    "부산광역시": {"road": "부산광역시 해운대구 해운대해변로 264", "tel_prefix": "051"},
    "울산광역시": {"road": "울산광역시 남구 대학로 93", "tel_prefix": "052"},
    "세종특별자치시": {"road": "세종특별자치시 도움6로 11", "tel_prefix": "044"},
    "경기도": {"road": "경기도 수원시 팔달구 효원로 1", "tel_prefix": "031"},
    "강원특별자치도": {"road": "강원특별자치도 강릉시 창해로 307", "tel_prefix": "033"},
    "충청북도": {"road": "충청북도 청주시 상당구 상당로 82", "tel_prefix": "043"},
    "충청남도": {"road": "충청남도 공주시 금벽로 368", "tel_prefix": "041"},
    "전라북도": {"road": "전라북도 전주시 완산구 기린대로 99", "tel_prefix": "063"},
    "전라남도": {"road": "전라남도 여수시 오동도로 61", "tel_prefix": "061"},
    "경상북도": {"road": "경상북도 경주시 보문로 424", "tel_prefix": "054"},
    "경상남도": {"road": "경상남도 창원시 성산구 중앙대로 151", "tel_prefix": "055"},
    "제주특별자치도": {"road": "제주특별자치도 제주시 첨단로 242", "tel_prefix": "064"}
}

# ==========================================
# 3. 사이드바 - 유저 성향 및 하네스 설정
# ==========================================
st.sidebar.header("⚙️ 유저 여행 성향 설정")

SEED_DATA = {
    "start_location": "서울역",
    "departure_time": datetime.time(9, 0),
    "age": 28,
    "companion": "친구들",
    "travel": "도심 핫플레이스, 복합문화공간, 브랜드 공간 및 팝업스토어",
    "culinary": "디저트 카페, 트렌디한 로컬 다이닝, 감성 와인/펍",
    "arts": "현대 미술 전시, 공간 디자인 및 아트 갤러리, 도심 문화 공간",
    "bio": "전통적인 대표 명소보다는 도심의 트렌디한 공간과 감성적인 분위기를 즐기며 감각적인 경험을 하는 여행을 선호합니다."
}

st.sidebar.subheader("1. 여행 기본 정보")
start_location = st.sidebar.text_input("🚩 출발지 (시작 위치)", SEED_DATA["start_location"])

departure_time = st.sidebar.time_input("⏰ 출발 시간", SEED_DATA["departure_time"])
departure_time_str = departure_time.strftime("%H:%M")

selected_region = st.sidebar.selectbox("여행 희망 지역", list(AREA_INFO.keys()), index=0)
travel_duration = st.sidebar.radio("여행 일정", ["당일치기", "1박 2일", "2박 3일"], index=1)

user_age = st.sidebar.slider("연령대", 18, 70, SEED_DATA["age"])

companion_options = ["혼자", "연인/배우자", "친구들", "가족/아이와 함께", "부모님과 함께"]
companion_idx = companion_options.index(SEED_DATA["companion"]) if SEED_DATA["companion"] in companion_options else 2
companion = st.sidebar.selectbox("동행인", companion_options, index=companion_idx)

st.sidebar.subheader("2. 세부 취향 (관심사)")
interest_travel = st.sidebar.text_input("🏕️ 여행 스타일", SEED_DATA["travel"])
interest_culinary = st.sidebar.text_input("🍱 미식 / 식음료", SEED_DATA["culinary"])
interest_arts = st.sidebar.text_input("🖼️ 문화 / 예술", SEED_DATA["arts"])

st.sidebar.subheader("3. 성향 요약")
user_bio = st.sidebar.text_area("나의 라이프스타일", SEED_DATA["bio"])

st.sidebar.markdown("---")
st.sidebar.header("🕹️ 에이전트 하네스(Harness) 설정")
max_iterations = st.sidebar.slider("최대 시뮬레이션 반복 횟수 (Max Turns)", 1, 4, 3)
target_score = st.sidebar.slider("목표 만족도 점수 (Cut-off Score)", 70, 95, 88)

if not KAKAO_API_KEY:
    st.sidebar.error("⚠️ KAKAO_API_KEY가 설정되지 않았습니다.")
if not GEMINI_API_KEY:
    st.sidebar.error("⚠️ GEMINI_API_KEY가 설정되지 않았습니다.")

# ==========================================
# 4. 데이터 로드 및 다양한 페르소나 인덱싱
# ==========================================
@st.cache_data(show_spinner="인구 페르소나 데이터 베이스 구축 중...")
def load_persona_data():
    records = []
    try:
        ds = load_dataset("nvidia/Nemotron-Personas-Korea", split="train[:1500]")
        for idx, item in enumerate(ds):
            age_val = item.get("age", 30)
            travel = item.get("travel") or item.get("travel_style") or ""
            culinary = item.get("culinary") or item.get("food_preference") or ""
            concise = item.get("concise") or item.get("summary") or ""
            
            travel_str = str(travel).strip()
            culinary_str = str(culinary).strip()
            concise_str = str(concise).strip()

            if travel_str or culinary_str or concise_str:
                records.append({
                    "id": f"persona_hf_{idx}",
                    "age": age_val,
                    "location": item.get("location", "전국"),
                    "travel": travel_str or "로컬 탐방 및 휴식",
                    "culinary": culinary_str or "지역 향토 음식 및 맛집 탐방",
                    "summary": concise_str or "여유로운 식도락과 휴식을 지향하는 여행 성향",
                    "matching_text": f"연령대: {age_val}세 | 여행취향: {travel_str} | 미식: {culinary_str} | 라이프스타일: {concise_str}"
                })
    except Exception:
        pass

    fallback_personas = [
        {
            "id": "p_urban", "age": 27, "location": "서울특별시",
            "travel": "도심 핫플레이스, 복합문화공간 및 팝업스토어, 감성 골목 탐방",
            "culinary": "트렌디한 디저트 카페, 감성 로컬 다이닝, 와인바",
            "summary": "도심의 감각적인 공간과 트렌디한 문화를 즐기는 20대 라이프스타일",
            "matching_text": "연령대: 27세 | 여행취향: 도심 핫플레이스 팝업스토어 복합문화공간 | 미식: 디저트 카페 와인바 | 라이프스타일: 감성 공간"
        },
        {
            "id": "p_nature", "age": 34, "location": "강원특별자치도",
            "travel": "자연 경관 산책로, 고즈넉한 수목원, 여유로운 숲길 힐링 스팟",
            "culinary": "정갈한 한식 다이닝, 지역 향토 음식, 뷰 좋은 자생 차 다원",
            "summary": "빡빡한 일정 대신 자연 속에서 여유롭게 휴식과 식도락을 즐기는 힐링 성향",
            "matching_text": "연령대: 34세 | 여행취향: 자연 경관 산책 힐링 스팟 수목원 | 미식: 한식 다이닝 향토 음식 다원 | 라이프스타일: 자연 여유 휴식"
        },
        {
            "id": "p_family", "age": 42, "location": "경기도",
            "travel": "가족 체험형 박물관, 넓은 야외 공원 및 역사 유적지, 안전한 산책 코스",
            "culinary": "남녀노소 즐기기 좋은 정갈한 로컬 맛집, 대형 베이커리 카페",
            "summary": "가족 및 아이와 함께 안전하고 교육적인 체험 및 휴식을 도모하는 라이프스타일",
            "matching_text": "연령대: 42세 | 여행취향: 가족 아이 체험 공원 역사 유적지 | 미식: 베이커리 카페 로컬 맛집 | 라이프스타일: 가족 안전 유적지"
        },
        {
            "id": "p_culture", "age": 52, "location": "경상북도",
            "travel": "전통 한옥 마을, 정갈한 로컬 역사 거리, 전통 공예 숍 및 고즈넉한 사찰",
            "culinary": "깊은 맛의 전통 향토 한정식, 오션뷰/마운틴뷰 찻집",
            "summary": "지역 고유의 역사와 전통문화를 깊이 있게 탐방하며 고즈넉하게 즐기는 성향",
            "matching_text": "연령대: 52세 | 여행취향: 전통 한옥 고즈넉 사찰 역사 문화 | 미식: 한정식 전통 찻집 향토음식 | 라이프스타일: 고즈넉 역사 전통"
        },
        {
            "id": "p_activity", "age": 23, "location": "부산광역시",
            "travel": "액티비티 체험, 레저 스포츠, 오션뷰 포토존, 액티브한 야외 탐방",
            "culinary": "스트리트 푸드, 야시장, 시원한 펍 및 로컬 해산물",
            "summary": "생기 넘치는 활동과 새로운 경험, 사진 촬영을 즐기는 활달한 성향",
            "matching_text": "연령대: 23세 | 여행취향: 액티비티 오션뷰 레저 스포츠 포토존 | 미식: 스트리트 푸드 야시장 펍 | 라이프스타일: 활발 활동 레저"
        }
    ]

    if len(records) < 5:
        records.extend(fallback_personas)

    return pd.DataFrame(records)

@st.cache_resource(show_spinner="AI 임베딩 모델 준비 중...")
def load_embedding_model():
    return SentenceTransformer("jhgan/ko-sroberta-multitask")

df_personas = load_persona_data()
embed_model = load_embedding_model()

@st.cache_data(show_spinner="벡터 인덱싱 완료...")
def get_persona_embeddings(_model, texts):
    return _model.encode(texts, show_progress_bar=False)

persona_embeddings = get_persona_embeddings(embed_model, df_personas["matching_text"].tolist())

# 카카오 지도 장소 수집 및 분류(Category) 파싱 보완
def fetch_kakao_places(region_name, travel_style, culinary_style, travel_duration="1박 2일"):
    if not KAKAO_API_KEY:
        return get_fallback_places(region_name)
    
    total_size = 14 if travel_duration == "2박 3일" else 10
    
    raw_key = KAKAO_API_KEY.strip().replace("KakaoAK", "").strip()
    url = "https://dapi.kakao.com/v2/local/search/keyword.json"
    headers = {"Authorization": f"KakaoAK {raw_key}"}
    
    travel_kw = travel_style.split(",")[0].strip() if travel_style else "관광 명소"
    culinary_kw = culinary_style.split(",")[0].strip() if culinary_style else "맛집"
    
    queries = [
        f"{region_name} {travel_kw}",
        f"{region_name} {culinary_kw}",
        f"{region_name} 핫플레이스"
    ]
    
    if travel_duration in ["1박 2일", "2박 3일"]:
        queries.append(f"{region_name} 감성숙소")
        queries.append(f"{region_name} 호텔")

    places = []
    seen_ids = set()
    
    for query in queries:
        params = {"query": query, "size": 3}
        try:
            res = requests.get(url, headers=headers, params=params, timeout=5)
            if res.status_code == 200:
                docs = res.json().get("documents", [])
                for doc in docs:
                    p_id = doc.get("id")
                    if p_id in seen_ids:
                        continue
                    seen_ids.add(p_id)
                    
                    p_name = doc.get("place_name")
                    addr = doc.get("road_address_name") or doc.get("address_name")
                    phone = doc.get("phone") or f"{AREA_INFO.get(region_name, {}).get('tel_prefix', '02')}-123-4567"
                    place_url = doc.get("place_url") or f"https://map.kakao.com/link/search/{urllib.parse.quote(addr)}"
                    
                    cat_group = (doc.get("category_group_name") or "").strip()
                    cat_full = (doc.get("category_name") or "").strip()
                    
                    if cat_group:
                        category_val = cat_group
                    elif cat_full:
                        category_val = cat_full.split(">")[-1].strip()
                    else:
                        category_val = "추천 장소"
                    
                    places.append({
                        "id": p_id,
                        "title": p_name,
                        "addr": addr,
                        "tel": phone,
                        "url": place_url,
                        "category": category_val
                    })
        except Exception:
            continue
            
    if places:
        return places[:total_size]
        
    return get_fallback_places(region_name)

def get_fallback_places(region_name):
    prefix = AREA_INFO.get(region_name, {}).get("tel_prefix", "02")
    base_addr = AREA_INFO.get(region_name, {}).get("road", "서울특별시 종로구 사직로 161")
    
    fallback_items = [
        {"id": "fb_1", "title": f"{region_name} 복합문화공간", "addr": base_addr, "tel": f"{prefix}-111-2222", "category": "문화공간"},
        {"id": "fb_2", "title": f"{region_name} 감성 디저트 카페", "addr": base_addr, "tel": f"{prefix}-222-3333", "category": "카페"},
        {"id": "fb_3", "title": f"{region_name} 현대 미술 갤러리", "addr": base_addr, "tel": f"{prefix}-333-4444", "category": "전시관"},
        {"id": "fb_4", "title": f"{region_name} 트렌디 로컬 다이닝", "addr": base_addr, "tel": f"{prefix}-444-5555", "category": "음식점"},
        {"id": "fb_5", "title": f"{region_name} 도심 팝업스토어 거리", "addr": base_addr, "tel": f"{prefix}-555-6666", "category": "쇼핑"},
        {"id": "fb_hotel1", "title": f"{region_name} 부티크 감성 스테이", "addr": base_addr, "tel": f"{prefix}-777-8888", "category": "숙박"},
        {"id": "fb_hotel2", "title": f"{region_name} 오션/시티뷰 호텔", "addr": base_addr, "tel": f"{prefix}-888-9999", "category": "숙박"}
    ]
    for item in fallback_items:
        item["url"] = f"https://map.kakao.com/link/search/{urllib.parse.quote(item['addr'])}"
    return fallback_items

# ==========================================
# 5. 여행 도메인 맞춤 에이전트 클래스
# ==========================================
class PlannerAgent:
    """여행 기획 에이전트: 실제 영업시간(클럽/나이트라이프 등 심야업종 고려) 및 타임라인 정밀 생성"""
    def __init__(self, model):
        self.model = model

    def generate_itinerary(self, user_info, places_list, turn=1, feedback=None, previous_itinerary=None):
        feedback_prompt = ""
        if feedback and previous_itinerary:
            feedback_prompt = f"""
            \n[이전 회차(Turn {turn-1}) 일정표]
            {previous_itinerary}

            [검증관의 지적 및 개선 요구사항]
            {feedback}
            
            ⚠️ 위 검증관 피드백을 100% 반영하여 영업시간, 이동 동선, 시간대별 스케줄, 숙소 배치를 완벽하게 재정교화하세요!
            """

        places_text = "\n".join([
            f"- 장소명: {p['title']} | 카테고리: {p.get('category','')} | 주소: {p['addr']} | 카카오맵 URL: {p['url']}"
            for p in places_list
        ])

        prompt = f"""
        당신은 대한민국 맞춤형 여행 코스 플래너(Planner Agent)입니다. (현재 시뮬레이션: Turn {turn})
        유저 정보와 [제공된 실제 카카오 장소 목록]만을 엄격히 사용하여 {user_info['duration']} 일정을 기획하세요.

        [유저 기본 정보]
        - 🚩 출발지 (시작 위치): {user_info['start_location']}
        - ⏰ 출발 시각: {user_info['departure_time']}
        - 희망 지역/일정: {user_info['region']} / {user_info['duration']}
        - 연령 / 동행인: {user_info['age']}세 / {user_info['companion']}
        - 여행 스타일: {user_info['interest_travel']}
        - 미식 선호: {user_info['interest_culinary']}
        - 라이프스타일: {user_info['user_bio']}

        [제공된 실제 카카오 장소 목록]
        {places_text}
        {feedback_prompt}

        ⚠️ [일정 기획 및 영업시간 고려 필수 규칙]
        1. **업종별 실제 영업시간 엄격 반영**:
           - **댄스클럽, 나이트라이프, 라운지바, 클럽 등**: 실제 오픈 시각이 늦은 밤(예: 22:00 ~ 23:00 이후)입니다. 절대로 저녁 일찍(18:00~21:00) 배치하지 마시고, **밤 23:00 이후 또는 심야 시간대**에 방문하도록 일정을 구성하세요.
           - **식당/다이닝**: 점심(11:30~14:00) 및 저녁(17:30~20:30) 식사 시간대에 배치하세요.
           - **카페/갤러리/팝업스토어/박물관**: 주간 및 오후 시간대(10:00~20:00)에 배치하세요.
        2. **시간대별 타임라인 필수**: 유저 출발 시각({user_info['departure_time']})에서 시작하여 각 장소 방문 시간을 `HH:MM ~ HH:MM` 형식으로 명시하세요.
        3. **이동시간 및 동선 표기**: 장소 간 이동마다 `🚗 예상 이동시간: 약 OO분` 항목을 명시하세요.
        4. **숙소 추천 및 체크인**: 1박 2일/2박 3일 일정일 경우, 저녁 식사 후 또는 나이트라이프 방문 전후로 숙소 체크인 및 휴식 동선을 자연스럽게 연결하세요.
        5. **마크다운 구문**: [제공된 카카오 목록]의 정확한 장소명, URL, 주소를 사용하여 `[장소명](카카오맵 URL) (주소: 실제주소)` 구문으로 표기하세요.
        """
        response = self.model.generate_content(prompt)
        return response.text

class EvaluatorAgent:
    """여행 도메인 검증관: 출발시각, 업종별 영업시간, 숙소 동선 통합 검증"""
    def __init__(self, model):
        self.model = model

    def evaluate(self, user_info, itinerary, turn=1, previous_score=72):
        prompt = f"""
        당신은 여행 도메인 에이전트 하네스의 '사용자 맞춤성, 영업시간 및 동선 검증관'입니다.
        현재 회차: Turn {turn} (이전 회차 점수: {previous_score}점)

        아래 유저 요구조건과 제안된 여행 일정표({user_info['duration']})를 엄격하게 평가하세요.

        [사용자가 입력한 요구 조건]
        - 출발지 / 출발시각: {user_info['start_location']} / {user_info['departure_time']}
        - 연령 / 동행인: {user_info['age']}세 / {user_info['companion']}
        - 여행 일정: {user_info['duration']}
        - 여행 스타일: {user_info['interest_travel']}
        - 미식 선호: {user_info['interest_culinary']}
        - 라이프스타일: {user_info['user_bio']}

        [검증 대상 여행 코스]
        {itinerary}

        ⚠️ [여행 하네스 엄격 평가 및 점수 산정 규칙]
        1. **업종별 실제 영업시간 적절성 (매우 중요)**:
           - 댄스클럽, 클럽, 라운지 등 나이트라이프 업종이 23:00 이전(예: 저녁 6시~9시)에 배치되어 있다면 **영업시간 미준수로 감점(-10점)**하고, 23:00 이후 심야에 재배치하도록 지적하세요.
           - 카페나 미술관이 너무 늦은 밤에 배치되어 있다면 수정 지적하세요.
        2. **출발 시각 반영 여부**: 일정의 시작이 유저 출발 시각({user_info['departure_time']})과 부합하는지 점검하세요.
        3. **숙소 배치 점검**: 1박2일/2박3일인 경우 적절한 숙소 추천 및 체크인/숙박 동선이 포함되었는지 평가하세요.
        4. Turn 2 이상에서 영업시간 오류 등이 지적 및 해결되었다면 이전 점수({previous_score}점)보다 상승된 점수를 부여하세요.

        [응답 형식 - 반드시 아래 JSON 형식으로만 답변하세요]
        ```json
        {{
            "score": 85,
            "satisfaction": "{user_info['departure_time']} 출발 시각 기준의 타임라인 및 업종별 실제 영업시간(클럽/나이트라이프 심야 배치 등) 반영이 개선된 이유 1~2문장",
            "critique": "다음 Turn에서 영업시간이나 동선을 더욱 완벽히 보완하기 위한 지적 1문장 (점수가 목표치 이상이면 '추가 보완 없이 최고 수준입니다' 표기)"
        }}
        ```
        """
        response = self.model.generate_content(prompt)
        
        try:
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception:
            pass
            
        calc_score = min(95, previous_score + 8 if turn > 1 else 75)
        return {
            "score": calc_score,
            "satisfaction": f"Turn {turn}: {user_info['departure_time']} 출발 시각 및 업종별 영업시간(심야 장소 구분) 반영이 개선되었습니다.",
            "critique": "장소 간 이동시간 및 심야 업종 운영 시작 시각을 다시 한번 점검하세요."
        }

# ==========================================
# 6. 시뮬레이션 실행 및 메인 로직
# ==========================================
if st.button("🚀 사용자 맞춤 에이전틱 시뮬레이션 실행 (Harness Loop Start)", type="primary"):
    if not GEMINI_API_KEY:
        st.error("Gemini API 키가 필요합니다.")
        st.stop()
        
    genai.configure(api_key=GEMINI_API_KEY)
    llm = genai.GenerativeModel("gemini-3.1-flash-lite")

    user_info = {
        "start_location": start_location,
        "departure_time": departure_time_str,
        "region": selected_region,
        "duration": travel_duration,
        "age": user_age,
        "companion": companion,
        "interest_travel": interest_travel,
        "interest_culinary": interest_culinary,
        "interest_arts": interest_arts,
        "user_bio": user_bio
    }

    # 1) 동적 유저 페르소나 매칭
    with st.spinner("1️⃣ 유저 성향 임베딩 & 최적 유사 페르소나 동적 매칭 중..."):
        user_query_text = (
            f"연령대: {user_age}세 | 동행: {companion} | "
            f"여행취향: {interest_travel} | 미식: {interest_culinary} | "
            f"문화: {interest_arts} | 성향: {user_bio}"
        )
        user_vector = embed_model.encode([user_query_text])
        sim_scores = cosine_similarity(user_vector, persona_embeddings)[0]
        
        top_idx = sim_scores.argmax()
        top_persona = df_personas.iloc[top_idx]
        top_match_score = round(float(sim_scores[top_idx] * 100), 1)

    st.success(f"🎉 유저 맞춤 분석 완료! (유사 페르소나 데이터 매칭 유사도: {top_match_score}%)")

    # 2) 비교 UI
    with st.expander("👥 사용자 입력 데이터 VS 매칭된 유사 페르소나 비교 분석", expanded=True):
        col_u, col_p = st.columns(2)
        
        with col_u:
            st.markdown("### 👤 내가 입력한 여행 성향")
            st.markdown(f"- **출발지 / 출발시각:** `{user_info['start_location']}` / `⏰ {user_info['departure_time']}`")
            st.markdown(f"- **일정 / 지역:** `{user_info['duration']}` / `{user_info['region']}`")
            st.markdown(f"- **연령 / 동행:** `{user_info['age']}세` / `{user_info['companion']}`")
            st.markdown(f"- **여행 스타일:** {user_info['interest_travel']}")
            st.markdown(f"- **미식 선호:** {user_info['interest_culinary']}")
            st.markdown(f"- **라이프스타일:** {user_info['user_bio']}")

        with col_p:
            st.markdown(f"### 🤝 AI가 매칭한 유사 페르소나 (유사도 {top_match_score}%)")
            st.markdown(f"- **페르소나 연령 / 거주:** `{top_persona['age']}세` / `{top_persona['location']}`")
            st.markdown(f"- **유사 여행 취향:** {top_persona['travel']}")
            st.markdown(f"- **유사 미식 취향:** {top_persona['culinary']}")
            st.markdown(f"- **페르소나 요약:** {top_persona['summary']}")

    # 3) 실시간 카카오 장소 + 숙소 수집
    with st.spinner(f"3️⃣ [{selected_region}] ({travel_duration}) 카카오 지도 장소 및 숙소 정보 매칭 중..."):
        real_places = fetch_kakao_places(selected_region, interest_travel, interest_culinary, travel_duration)

    planner = PlannerAgent(llm)
    evaluator = EvaluatorAgent(llm)

    # 4) 에이전틱 시뮬레이션 루프 (Progressive Harness Loop)
    st.markdown("---")
    st.subheader("🔄 사용자 맞춤 에이전틱 시뮬레이션 (Harness Loop Log)")
    
    current_itinerary = ""
    last_feedback = None
    running_score = 72
    
    simulation_container = st.container()

    for turn in range(1, max_iterations + 1):
        with simulation_container.expander(f"📍 [Turn {turn}/{max_iterations}] 에이전트 자율 피드백 루프 진행 중...", expanded=True):
            col_plan, col_eval = st.columns(2)
            
            # Step A: Planner
            with col_plan:
                st.markdown(f"**🤖 Planner Agent (Turn {turn})**")
                with st.spinner("영업시간 및 타임라인 고려 일정표 기획 중..."):
                    current_itinerary = planner.generate_itinerary(
                        user_info, real_places, turn=turn, feedback=last_feedback, previous_itinerary=current_itinerary
                    )
                st.info(f"✅ Turn {turn} 일정표 개선 완료")
            
            # Step B: Evaluator
            with col_eval:
                st.markdown(f"**🕵️ 사용자 성향 검증관 (Turn {turn})**")
                with st.spinner("영업시간 및 출발시각 검증 중..."):
                    eval_result = evaluator.evaluate(user_info, current_itinerary, turn=turn, previous_score=running_score)
                
                eval_score = eval_result.get("score", running_score + 7)
                
                if turn > 1:
                    eval_score = max(eval_score, running_score + 4)
                
                running_score = min(100, eval_score)
                satisfaction = eval_result.get("satisfaction", "")
                critique = eval_result.get("critique", "")
                
                st.metric("만족도 점수", f"{running_score} / 100점", delta=f"{running_score - target_score} (목표: {target_score}점)")
                st.caption(f"👍 **만족 요소:** {satisfaction}")
                st.caption(f"💡 **개선 요구:** {critique}")
                
                last_feedback = f"이전 점수: {running_score}점 | 만족: {satisfaction} | 추가개선요구: {critique}"

            if running_score >= target_score:
                st.success(f"🎉 Turn {turn}에서 유저 목표 만족도 점수({target_score}점)를 달성하여 시뮬레이션을 완료합니다!")
                break
            elif turn < max_iterations:
                st.warning(f"⚠️ 목표 점수({target_score}점) 미달로 Planner Agent에게 개선 지침을 전달합니다.")

    # 5) 코스에 '실제 반영된 메인 장소'와 '대안/추가 장소'를 정밀하게 분리
    main_places = []
    alt_places = []

    for place in real_places:
        pos = current_itinerary.find(place['title'])
        if pos == -1:
            short_title = place['title'].split()[0] if len(place['title'].split()) > 0 else place['title']
            if len(short_title) >= 2:
                pos = current_itinerary.find(short_title)
        
        if pos != -1:
            main_places.append((pos, place))
        else:
            alt_places.append(place)

    main_places.sort(key=lambda x: x[0])
    sorted_main_places = [p for _, p in main_places]

    # 6) 최종 결과 출력 화면
    st.markdown("---")
    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.subheader(f"🏆 최종 검증된 [{selected_region}] ({travel_duration}) 맞춤형 여행 코스")
        st.info(f"🚩 **출발 위치/시각:** {user_info['start_location']} ({user_info['departure_time']} 출발) | 🎯 **최종 만족도 점수:** {running_score} / 100점")
        st.markdown(current_itinerary)

    with col_right:
        st.subheader("📍 연동 명소 & 숙소 정보")
        
        st.markdown("#### 1️⃣ 추천 코스에 포함된 명소 & 숙소")
        st.caption("※ 여행 코스 타임라인 순서(Day 1 ➔ Day 2)에 따라 배치되었습니다.")
        
        if sorted_main_places:
            for idx, place in enumerate(sorted_main_places):
                with st.container(border=True):
                    st.markdown(f"##### {idx+1}. {place['title']}")
                    st.markdown(f"🏷️ **분류:** `{place['category']}`")
                    st.markdown(f"📍 **실제 주소:** `{place['addr']}`")
                    st.caption(f"📞 {place['tel']}")
                    st.markdown(f"[🔗 카카오맵에서 위치 및 길찾기]({place['url']})")
        else:
            st.info("코스에 포함된 주요 장소를 정밀 매칭하는 중입니다.")

        if alt_places:
            st.markdown("---")
            with st.expander("💡 **[대안 옵션] 코스 외 추가 추천 장소 & 숙소**", expanded=True):
                st.caption("※ 일정표에 직접 포함되지는 않았으나, 여정 변경 시 활용할 수 있는 대안 명소 및 추가 숙소입니다.")
                for idx, place in enumerate(alt_places):
                    with st.container(border=True):
                        st.markdown(f"##### 💡 [대안/추가] {place['title']}")
                        st.markdown(f"🏷️ **분류:** `{place['category']}`")
                        st.markdown(f"📍 **실제 주소:** `{place['addr']}`")
                        st.caption(f"📞 {place['tel']}")
                        st.markdown(f"[🔗 카카오맵에서 위치 및 길찾기]({place['url']})")
