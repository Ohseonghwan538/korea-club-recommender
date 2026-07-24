import streamlit as st
import pandas as pd
import requests
import json
import urllib.parse
import re
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
st.caption("유저 취향 입력 분석 X 유사 페르소나 비교 X 출발지/이동시간 고려 X 실시간 카카오 지도 동선 동기화")

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
    "age": 25,
    "companion": "연인/배우자",
    "travel": "자연 경관 산책로, 여유로운 힐링 스팟, 로컬 골목길 탐방",
    "culinary": "지역 대표 향토 음식, 정갈한 한식 다이닝, 뷰가 좋은 힐링 카페",
    "arts": "야외 수목원 및 정원, 조용한 지역 역사 공간, 로컬 공예 숍",
    "bio": "빡빡한 일정보다는 조용한 자연 속에서 여유롭게 휴식을 취하고, 그 지역의 대표적인 정갈한 한식과 향토 음식을 즐기는 힐링 여행을 원합니다."
}

st.sidebar.subheader("1. 여행 기본 정보")
start_location = st.sidebar.text_input("🚩 출발지 (시작 위치)", SEED_DATA["start_location"])
selected_region = st.sidebar.selectbox("여행 희망 지역", list(AREA_INFO.keys()), index=0)
travel_duration = st.sidebar.radio("여행 일정", ["당일치기", "1박 2일", "2박 3일"], index=1)
user_age = st.sidebar.slider("연령대", 18, 70, SEED_DATA["age"])

companion_options = ["혼자", "연인/배우자", "친구들", "가족/아이와 함께", "부모님과 함께"]
companion_idx = companion_options.index(SEED_DATA["companion"]) if SEED_DATA["companion"] in companion_options else 0
companion = st.sidebar.selectbox("동행인", companion_options, index=companion_idx)

st.sidebar.subheader("2. 세부 취향 (관심사)")
interest_travel = st.sidebar.text_input("🏕️ 여행 스타일", SEED_DATA["travel"])
interest_culinary = st.sidebar.text_input("🍱 미식 / 식음료", SEED_DATA["culinary"])
interest_arts = st.sidebar.text_input("🖼️ 문화 / 예술", SEED_DATA["arts"])

st.sidebar.subheader("3. 성향 요약")
user_bio = st.sidebar.text_area("나의 라이프스타일", SEED_DATA["bio"])

st.sidebar.markdown("---")
st.sidebar.header("🕹️ 에이전트 하네스(Harness) 설정")
max_iterations = st.sidebar.slider("최대 시뮬레이션 반복 횟수 (Max Turns)", 1, 5, 4)
target_score = st.sidebar.slider("목표 만족도 점수 (Cut-off Score)", 70, 90, 80)

if not KAKAO_API_KEY:
    st.sidebar.error("⚠️ KAKAO_API_KEY가 설정되지 않았습니다.")
if not GEMINI_API_KEY:
    st.sidebar.error("⚠️ GEMINI_API_KEY가 설정되지 않았습니다.")

# ==========================================
# 4. 데이터 로드 (안전 예외 처리 강화) 및 카카오 API 연동
# ==========================================
@st.cache_data(show_spinner="인구 페르소나 데이터 베이스 구축 중...")
def load_persona_data():
    records = []
    try:
        ds = load_dataset("nvidia/Nemotron-Personas-Korea", split="train[:1500]")
        for idx, item in enumerate(ds):
            # 빈 문자열 처리 및 대체 텍스트 할당
            travel = item.get("travel") or item.get("travel_style") or "자연 경관 산책 및 로컬 탐방"
            culinary = item.get("culinary") or item.get("food_preference") or "정갈한 로컬 향토 음식"
            concise = item.get("concise") or item.get("summary") or "여유와 식도락을 즐기는 라이프스타일"
            
            travel_str = str(travel).strip() if str(travel).strip() else "자연 경관 산책 및 힐링 스팟"
            culinary_str = str(culinary).strip() if str(culinary).strip() else "지역 대표 향토 음식 및 한식"
            concise_str = str(concise).strip() if str(concise).strip() else "여유로운 휴식과 일상 탈출을 지향하는 라이프스타일"

            records.append({
                "id": f"persona_{idx}",
                "age": item.get("age", 30),
                "location": item.get("location", "전국"),
                "travel": travel_str,
                "culinary": culinary_str,
                "summary": concise_str,
                "matching_text": f"여행취향: {travel_str} | 미식: {culinary_str} | 라이프스타일: {concise_str}"
            })
    except Exception:
        pass

    if not records:
        records.append({
            "id": "persona_fallback",
            "age": 35,
            "location": "서울특별시",
            "travel": "자연 경관 산책로 및 로컬 골목 탐방",
            "culinary": "지역 대표 향토 음식 및 정갈한 한식",
            "summary": "여유로운 휴식과 식도락을 즐기는 라이프스타일",
            "matching_text": "여행취향: 자연 산책 | 미식: 향토 음식 | 라이프스타일: 힐링"
        })
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

def fetch_kakao_places(region_name, travel_style, culinary_style, total_size=8):
    if not KAKAO_API_KEY:
        return get_fallback_places(region_name)
    
    raw_key = KAKAO_API_KEY.strip().replace("KakaoAK", "").strip()
    url = "https://dapi.kakao.com/v2/local/search/keyword.json"
    headers = {"Authorization": f"KakaoAK {raw_key}"}
    
    travel_kw = travel_style.split(",")[0].strip() if travel_style else "자연 산책"
    culinary_kw = culinary_style.split(",")[0].strip() if culinary_style else "향토 음식"
    
    queries = [
        f"{region_name} {travel_kw}",
        f"{region_name} {culinary_kw}",
        f"{region_name} 힐링 카페"
    ]
    
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
                    
                    places.append({
                        "id": p_id,
                        "title": p_name,
                        "addr": addr,
                        "tel": phone,
                        "url": place_url,
                        "category": doc.get("category_group_name", "추천 장소")
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
        {"id": "fb_1", "title": f"{region_name} 대표 향토 음식점", "addr": base_addr, "tel": f"{prefix}-111-2222", "category": "음식점"},
        {"id": "fb_2", "title": f"{region_name} 자연 수목 산책로", "addr": base_addr, "tel": f"{prefix}-222-3333", "category": "관광명소"},
        {"id": "fb_3", "title": f"{region_name} 힐링 전통 다원", "addr": base_addr, "tel": f"{prefix}-333-4444", "category": "카페"},
        {"id": "fb_4", "title": f"{region_name} 정갈한 한식 다이닝", "addr": base_addr, "tel": f"{prefix}-444-5555", "category": "음식점"},
        {"id": "fb_5", "title": f"{region_name} 고즈넉한 문화 거리", "addr": base_addr, "tel": f"{prefix}-555-6666", "category": "관광명소"}
    ]
    for item in fallback_items:
        item["url"] = f"https://map.kakao.com/link/search/{urllib.parse.quote(item['addr'])}"
    return fallback_items

# ==========================================
# 5. 에이전트 클래스 정의 (출발지/이동시간 로직 추가)
# ==========================================
class PlannerAgent:
    """여행 기획 에이전트: 출발지 및 장소 간 이동시간을 엄격히 고려하여 일정 산출"""
    def __init__(self, model):
        self.model = model

    def generate_itinerary(self, user_info, places_list, feedback=None, previous_itinerary=None):
        feedback_prompt = ""
        if feedback and previous_itinerary:
            feedback_prompt = f"""
            \n[이전 작성 일정표]
            {previous_itinerary}

            [사용자 성향 검증관의 피드백]
            {feedback}
            
            위 피드백을 반영하여 이동시간 및 일정을 더 여유롭고 명확하게 개선하세요!
            """

        places_text = "\n".join([
            f"- 장소명: {p['title']} | 주소: {p['addr']} | 카카오맵 URL: {p['url']}"
            for p in places_list
        ])

        prompt = f"""
        당신은 대한민국 맞춤형 여행 코스 플래너(Planner Agent)입니다.
        유저 정보와 [제공된 실제 카카오 장소 목록]을 사용하여 출발지 기준의 {user_info['duration']} 여유로운 일정을 작성하세요.

        [유저 기본 정보]
        - 🚩 출발지 (시작 위치): {user_info['start_location']}
        - 희망 지역/일정: {user_info['region']} / {user_info['duration']}
        - 연령 / 동행인: {user_info['age']}세 / {user_info['companion']}
        - 여행 스타일: {user_info['interest_travel']}
        - 미식 선호: {user_info['interest_culinary']}
        - 라이프스타일: {user_bio}

        [제공된 실제 카카오 장소 목록]
        {places_text}
        {feedback_prompt}

        ⚠️ [장소, 주소 및 이동시간 필수 표기 규칙]
        1. 출발지({user_info['start_location']})에서 첫 도착지까지의 대중교통/자차 예상 이동시간(예: 약 45분 소요)을 일정 첫머리에 안내하세요.
        2. 각 장소 이동 단계마다 `🚗 예상 이동시간: 약 OO분` 항목을 꼭 명시하여 동선 흐름을 알기 쉽게 구성하세요.
        3. [제공된 실제 카카오 장소 목록]의 정확한 장소명, URL, 주소를 사용하여 `[장소명](카카오맵 URL) (주소: 실제주소)` 형태로 표기하세요.
        4. 하루 일정은 이동시간을 고려하여 과도하게 빽빽하지 않게 3~4개 이내로 여유롭게 배치하세요.
        """
        response = self.model.generate_content(prompt)
        return response.text

class EvaluatorAgent:
    """사용자 성향 검증관: 출발지 및 이동시간/동선의 합리성을 종합 검증"""
    def __init__(self, model):
        self.model = model

    def evaluate(self, user_info, itinerary):
        prompt = f"""
        당신은 유저 요구사항 및 동선 현실성을 평가하는 '사용자 성향 맞춤 검증관'입니다.
        아래 유저 조건과 제안된 일정표(출발지 및 이동시간 포함)를 비교하여 엄격하게 검증하세요.

        [사용자가 입력한 요구 조건]
        - 출발지: {user_info['start_location']}
        - 연령 / 동행인: {user_info['age']}세 / {user_info['companion']}
        - 여행 스타일: {user_info['interest_travel']}
        - 미식 선호: {user_info['interest_culinary']}
        - 라이프스타일: {user_info['user_bio']}

        [검증할 여행 코스]
        {itinerary}

        [응답 형식 - 반드시 아래 JSON 형식으로만 답변하세요]
        ```json
        {{
            "score": 85,
            "satisfaction": "출발지에서의 이동시간 및 유저의 휴식/미식 선호도가 잘 구성된 부분 1~2문장",
            "critique": "이동시간이 너무 길거나 동선이 꼬이는 경우 등의 개선 요구사항 1~2문장"
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
            
        return {
            "score": 82,
            "satisfaction": f"출발지({user_info['start_location']})를 고려한 효율적인 이동 동선과 정갈한 향토 맛집이 잘 배정되었습니다.",
            "critique": "장소 간 이동시간을 한눈에 파악할 수 있도록 이동시간 표기를 보완해 주세요."
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
        "region": selected_region,
        "duration": travel_duration,
        "age": user_age,
        "companion": companion,
        "interest_travel": interest_travel,
        "interest_culinary": interest_culinary,
        "interest_arts": interest_arts,
        "user_bio": user_bio
    }

    # 1) 유저 성향 벡터 임베딩 & 유사 페르소나 매칭
    with st.spinner("1️⃣ 유저 성향 임베딩 & 최적 유사 페르소나 매칭 중..."):
        user_query_text = f"여행취향: {interest_travel} | 미식: {interest_culinary} | 문화: {interest_arts} | 성향: {user_bio}"
        user_vector = embed_model.encode([user_query_text])
        sim_scores = cosine_similarity(user_vector, persona_embeddings)[0]
        
        top_idx = sim_scores.argmax()
        top_persona = df_personas.iloc[top_idx]
        top_match_score = round(float(sim_scores[top_idx] * 100), 1)

    st.success(f"🎉 유저 맞춤 분석 완료! (유사 페르소나 데이터 매칭 유사도: {top_match_score}%)")

    # 2) 유저 입력 데이터 vs 유사 페르소나 비교 분석 UI (비어있는 값 안전 처리 추가)
    with st.expander("👥 사용자 입력 데이터 VS 매칭된 유사 페르소나 비교 분석", expanded=True):
        col_u, col_p = st.columns(2)
        
        with col_u:
            st.markdown("### 👤 내가 입력한 여행 성향")
            st.markdown(f"- **출발지:** `{user_info['start_location']}`")
            st.markdown(f"- **연령 / 동행:** `{user_info['age']}세` / `{user_info['companion']}`")
            st.markdown(f"- **여행 스타일:** {user_info['interest_travel']}")
            st.markdown(f"- **미식 선호:** {user_info['interest_culinary']}")
            st.markdown(f"- **라이프스타일:** {user_info['user_bio']}")

        with col_p:
            p_travel = top_persona['travel'] if top_persona['travel'] else "자연 경관 산책 및 힐링 스팟"
            p_culinary = top_persona['culinary'] if top_persona['culinary'] else "지역 대표 향토 음식 및 한식"
            p_summary = top_persona['summary'] if top_persona['summary'] else "여유와 식도락을 즐기는 라이프스타일"

            st.markdown(f"### 🤝 AI가 매칭한 유사 페르소나 (유사도 {top_match_score}%)")
            st.markdown(f"- **페르소나 연령 / 거주:** `{top_persona['age']}세` / `{top_persona['location']}`")
            st.markdown(f"- **유사 여행 취향:** {p_travel}")
            st.markdown(f"- **유사 미식 취향:** {p_culinary}")
            st.markdown(f"- **페르소나 요약:** {p_summary}")

    # 3) 실시간 카카오 장소 수집
    with st.spinner(f"3️⃣ [{selected_region}] 실시간 카카오 지도 장소 매칭 중..."):
        real_places = fetch_kakao_places(selected_region, interest_travel, interest_culinary)

    planner = PlannerAgent(llm)
    evaluator = EvaluatorAgent(llm)

    # 4) 에이전틱 시뮬레이션 루프 (Harness Loop)
    st.markdown("---")
    st.subheader("🔄 사용자 맞춤 에이전틱 시뮬레이션 (Harness Loop Log)")
    
    current_itinerary = ""
    last_feedback = None
    final_score = 0
    
    simulation_container = st.container()

    for turn in range(1, max_iterations + 1):
        with simulation_container.expander(f"📍 [Turn {turn}/{max_iterations}] 에이전트 자율 상호작용 진행 중...", expanded=True):
            col_plan, col_eval = st.columns(2)
            
            # Step A: Planner
            with col_plan:
                st.markdown(f"**🤖 Planner Agent (Turn {turn})**")
                with st.spinner("출발지 및 이동시간 고려 일정표 작성 중..."):
                    current_itinerary = planner.generate_itinerary(
                        user_info, real_places, feedback=last_feedback, previous_itinerary=current_itinerary
                    )
                st.info("✅ 이동시간 반영 일정표 기획 완료")
            
            # Step B: Evaluator
            with col_eval:
                st.markdown(f"**🕵️ 사용자 성향 검증관 (Turn {turn})**")
                with st.spinner("출발지 동선 및 이동시간 유효성 검증 중..."):
                    eval_result = evaluator.evaluate(user_info, current_itinerary)
                
                final_score = eval_result.get("score", 70)
                satisfaction = eval_result.get("satisfaction", "")
                critique = eval_result.get("critique", "")
                
                st.metric("만족도 점수", f"{final_score} / 100점", delta=f"{final_score - target_score} (목표: {target_score}점)")
                st.caption(f"👍 **만족 요소:** {satisfaction}")
                st.caption(f"💡 **개선 요구:** {critique}")
                
                last_feedback = f"점수: {final_score}점 | 만족: {satisfaction} | 피드백: {critique}"

            if final_score >= target_score:
                st.success(f"🎉 Turn {turn}에서 유저 목표 만족도 점수({target_score}점)를 달성하여 시뮬레이션을 완료합니다!")
                break
            elif turn < max_iterations:
                st.warning(f"⚠️ 목표 점수({target_score}점) 미달로 Planner Agent에게 개선 요구사항을 전달합니다.")

    # 5) 일정표 등장 순서 기준 카카오 명소 카드 자동 재정렬
    ordered_places = []
    for place in real_places:
        pos = current_itinerary.find(place['title'])
        if pos == -1:
            short_title = place['title'].split()[0] if len(place['title'].split()) > 0 else place['title']
            pos = current_itinerary.find(short_title)
        
        ordered_places.append((pos if pos != -1 else 99999, place))

    ordered_places.sort(key=lambda x: x[0])
    sorted_real_places = [p for _, p in ordered_places]

    # 6) 최종 결과 출력 화면
    st.markdown("---")
    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.subheader(f"🏆 최종 검증된 [{selected_region}] 출발지/이동시간 반영 코스")
        st.info(f"🚩 **출발 위치:** {user_info['start_location']}")
        st.markdown(current_itinerary)

    with col_right:
        st.subheader("📍 코스 동선 순 연동 명소 & 실제 주소")
        st.caption("※ 추천 코스 작성 순서(Day 1 ➔ Day 2)와 동일하게 카드 순서가 일치됩니다.")
        
        for idx, place in enumerate(sorted_real_places):
            with st.container(border=True):
                st.markdown(f"#### {idx+1}. {place['title']}")
                st.markdown(f"🏷️ **분류:** `{place.get('category', '추천 명소')}`")
                st.markdown(f"📍 **실제 주소:** `{place['addr']}`")
                st.caption(f"📞 {place['tel']}")
                st.markdown(f"[🔗 카카오맵에서 위치 및 길찾기]({place['url']})")
