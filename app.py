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
    page_title="에이전틱 시뮬레이션 기반 한국 여행 코스 추천",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 에이전틱 시뮬레이션 기반 맞춤 한국 여행 코스 플래너")
st.caption("Nemotron 페르소나 Evaluator Agent X Planner Agent X 시뮬레이션 Harness 제어 루프")

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
    "울산광역시": {"road": "울산광역시 남구 무거동 대학로 93", "tel_prefix": "052"},
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
# 3. 사이드바 - 유저 성향 및 하네스(Harness) 설정
# ==========================================
st.sidebar.header("⚙️ 유저 여행 성향 설정")

selected_region = st.sidebar.selectbox("여행 희망 지역", list(AREA_INFO.keys()), index=0)
travel_duration = st.sidebar.radio("여행 일정", ["당일치기", "1박 2일"], index=1)
user_age = st.sidebar.slider("연령대", 18, 70, 28)
companion = st.sidebar.selectbox("동행인", ["혼자", "연인/배우자", "친구들", "가족/아이와 함께", "부모님과 함께"], index=1)

interest_travel = st.sidebar.text_input("🏕️ 여행 스타일", "이 장소에서만 할 수 있는 특별한 경험, 핫플 탐방")
interest_culinary = st.sidebar.text_input("🍱 미식 / 식음료", "로컬 푸드, 감성 디저트")
interest_arts = st.sidebar.text_input("🖼️ 문화 / 예술", "트렌디한 문화 공간")
user_bio = st.sidebar.text_area("나의 라이프스타일", "흔한 대표 관광지보다는 트렌디한 문화와 독특한 핫플레이스를 탐방하고 싶습니다.")

st.sidebar.markdown("---")
st.sidebar.header("🕹️ 에이전트 하네스(Harness) 설정")
max_iterations = st.sidebar.slider("최대 시뮬레이션 반복 횟수 (Max Turns)", 1, 4, 3)
target_score = st.sidebar.slider("목표 만족도 점수 (Cut-off Score)", 70, 95, 85)

if not KAKAO_API_KEY:
    st.sidebar.error("⚠️ KAKAO_API_KEY가 설정되지 않았습니다.")
if not GEMINI_API_KEY:
    st.sidebar.error("⚠️ GEMINI_API_KEY가 설정되지 않았습니다.")

# ==========================================
# 4. 데이터 로드 및 카카오 API 연동
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

@st.cache_resource(show_spinner="AI 임베딩 모델 준비 중...")
def load_embedding_model():
    return SentenceTransformer("jhgan/ko-sroberta-multitask")

df_personas = load_persona_data()
embed_model = load_embedding_model()

@st.cache_data(show_spinner="페르소나 인덱싱 중...")
def get_persona_embeddings(_model, texts):
    return _model.encode(texts, show_progress_bar=False)

persona_embeddings = get_persona_embeddings(embed_model, df_personas["matching_text"].tolist())

def fetch_kakao_places(region_name, query_style, size=6):
    """카카오 로컬 API 검색 및 주소 기반 카카오맵 링크 생성"""
    if not KAKAO_API_KEY:
        return get_fallback_places(region_name)
    
    raw_key = KAKAO_API_KEY.strip().replace("KakaoAK", "").strip()
    url = "https://dapi.kakao.com/v2/local/search/keyword.json"
    headers = {"Authorization": f"KakaoAK {raw_key}"}
    params = {"query": f"{region_name} 이색 체험 핫플레이스", "size": size}
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        if response.status_code == 200:
            documents = response.json().get("documents", [])
            if documents:
                places = []
                for doc in documents:
                    place_name = doc.get("place_name", f"{region_name} 대표 명소")
                    addr = doc.get("road_address_name") or doc.get("address_name") or f"{region_name} 대표 주소"
                    tel = doc.get("phone") or f"{AREA_INFO.get(region_name, {}).get('tel_prefix', '02')}-123-4567"
                    
                    encoded_addr = urllib.parse.quote(addr)
                    place_url = f"https://map.kakao.com/link/search/{encoded_addr}"
                    
                    places.append({"title": place_name, "addr": addr, "tel": tel, "url": place_url})
                return places
    except Exception as e:
        st.error(f"카카오 API 연동 오류: {e}")
        
    return get_fallback_places(region_name)

def get_fallback_places(region_name):
    prefix = AREA_INFO.get(region_name, {}).get("tel_prefix", "02")
    fallback_items = [
        {"title": f"{region_name} 성수동 이색 팝업 스토어 & 문화공간", "addr": "서울특별시 성동구 연무장길 1", "tel": f"{prefix}-123-4567"},
        {"title": f"{region_name} 익선동 한옥 이색 감성 카페거리", "addr": "서울특별시 종로구 수표로28길 28", "tel": f"{prefix}-234-5678"},
        {"title": f"{region_name} DDP 동대문디자인플라자 전시", "addr": "서울특별시 중구 을지로 281", "tel": f"{prefix}-345-6789"},
        {"title": f"{region_name} 한강 달빛 야경 & 반포한강공원", "addr": "서울특별시 서초구 신반포로11길 40", "tel": f"{prefix}-456-7890"},
        {"title": f"{region_name} 용산 용리단길 로컬 맛집 거리", "addr": "서울특별시 용산구 한강대로38길 15", "tel": f"{prefix}-567-8901"}
    ]
    for item in fallback_items:
        item["url"] = f"https://map.kakao.com/link/search/{urllib.parse.quote(item['addr'])}"
    return fallback_items

# ==========================================
# 5. 에이전트 클래스 정의 (Agentic Simulation)
# ==========================================
class PlannerAgent:
    """여행 기획 에이전트: 코스 작성 및 피드백 기반 수정"""
    def __init__(self, model):
        self.model = model

    def generate_itinerary(self, user_info, places_text, feedback=None, previous_itinerary=None):
        feedback_prompt = ""
        if feedback and previous_itinerary:
            feedback_prompt = f"""
            \n[이전 일정표]
            {previous_itinerary}

            [페르소나 평가 에이전트의 피드백]
            {feedback}
            
            위 피드백과 지적 사항을 완벽하게 반영하여 코스를 개선/재작성해 주세요!
            """

        prompt = f"""
        당신은 대한민국 최고 맞춤형 여행 코스 플래너(Planner Agent)입니다.
        아래 유저 정보와 실시간 카카오 장소 목록을 기반으로 {user_info['duration']} 일정을 작성하세요.

        [유저 기본 정보]
        - 지역/일정: {user_info['region']} / {user_info['duration']}
        - 동행인: {user_info['companion']}
        - 관심사: {user_info['interest_travel']}, {user_info['interest_culinary']}, {user_info['interest_arts']}

        [카카오 API 실제 장소 정보]
        {places_text}
        {feedback_prompt}

        [작성 가이드]
        1. 감성적인 코스 타이틀
        2. 시간대별(오전/점심/오후/저녁) 명확한 일정
        3. 장소 언급 시 소괄호 안에 카카오맵 지도 링크 및 주소 필수 명시 (예: [장소명](카카오맵 주소 링크) (주소: 실제 주소))
        4. 이 장소를 추천하는 구체적인 이유 명시
        """
        response = self.model.generate_content(prompt)
        return response.text

class EvaluatorAgent:
    """페르소나 평가 에이전트: 페르소나 관점의 검토, 점수 산출 및 피드백 제공"""
    def __init__(self, model):
        self.model = model

    def evaluate(self, persona_profile, user_info, itinerary):
        prompt = f"""
        당신은 아래의 인구 페르소나(Persona) 본인입니다.
        제안된 여행 코스가 당신의 연령, 라이프스타일, 취향에 얼마나 맞는지 평가하세요.

        [당신의 페르소나 프로필]
        - 연령/지역: {persona_profile['age']}세 / {persona_profile['location']}
        - 라이프스타일 요약: {persona_profile['summary']}
        - 여행/미식 취향: {persona_profile['travel']} / {persona_profile['culinary']}

        [검토할 여행 코스]
        {itinerary}

        [응답 형식 - 반드시 아래 JSON 형식으로만 답변하세요]
        ```json
        {{
            "score": 80,
            "satisfaction": "만족스러운 부분 1~2문장",
            "critique": "아쉬운 부분 및 Planner Agent에게 요청할 개선점 1~2문장"
        }}
        ```
        - score는 0~100점 사이의 정수입니다.
        - 페르소나 성향에 부합하지 않거나 동선/취향이 아쉬우면 솔직하게 감점하고 critique를 작성하세요.
        """
        response = self.model.generate_content(prompt)
        text = response.text
        
        # JSON 추출
        try:
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception:
            pass
            
        return {"score": 75, "satisfaction": "전반적으로 양호합니다.", "critique": "시간대별 동선을 조금 더 효율적으로 조정해주세요."}

# ==========================================
# 6. 시뮬레이션 하네스(Harness) 실행
# ==========================================
if st.button("🚀 에이전틱 시뮬레이션 실행 (Harness Loop Start)", type="primary"):
    if not GEMINI_API_KEY:
        st.error("Gemini API 키가 필요합니다.")
        st.stop()
        
    genai.configure(api_key=GEMINI_API_KEY)
    llm = genai.GenerativeModel("gemini-3.1-flash-lite")

    # 1) 페르소나 매칭
    with st.spinner("1️⃣ Nemotron 인구 데이터셋 매칭 중..."):
        user_query_text = f"여행취향: {interest_travel} | 미식: {interest_culinary} | 문화: {interest_arts} | 성향: {user_bio}"
        user_vector = embed_model.encode([user_query_text])
        sim_scores = cosine_similarity(user_vector, persona_embeddings)[0]
        
        df_matched = df_personas.copy()
        df_matched["match_score"] = (sim_scores * 100).round(1)
        matched_persona = df_matched.sort_values(by="match_score", ascending=False).iloc[0]

    # 2) 카카오 장소 수집
    with st.spinner(f"2️⃣ [{selected_region}] 실시간 장소 데이터 수집 중..."):
        real_places = fetch_kakao_places(selected_region, interest_travel)
        places_text = "\n".join([f"- 장소명: {p['title']} | 실제주소: {p['addr']} | 카카오맵 주소 링크: {p['url']}" for p in real_places])

    user_info = {
        "region": selected_region,
        "duration": travel_duration,
        "companion": companion,
        "interest_travel": interest_travel,
        "interest_culinary": interest_culinary,
        "interest_arts": interest_arts
    }

    planner = PlannerAgent(llm)
    evaluator = EvaluatorAgent(llm)

    st.success(f"🎯 매칭된 페르소나: {matched_persona['age']}세 ({matched_persona['summary']}) | 유사도: {matched_persona['match_score']}%")

    # 3) 에이전틱 시뮬레이션 루프 (Harness)
    st.subheader("🔄 에이전틱 시뮬레이션 하네스 로그")
    
    current_itinerary = None
    last_feedback = None
    final_score = 0
    
    simulation_container = st.container()

    for turn in range(1, max_iterations + 1):
        with simulation_container.expander(f"📍 [Turn {turn}/{max_iterations}] 에이전트 상호작용 진행 중...", expanded=True):
            col_plan, col_eval = st.columns(2)
            
            # Step A: Planner Agent 생성/수정
            with col_plan:
                st.markdown(f"**🤖 Planner Agent (Turn {turn})**")
                with st.spinner("코스 기획/수정 중..."):
                    current_itinerary = planner.generate_itinerary(
                        user_info, places_text, feedback=last_feedback, previous_itinerary=current_itinerary
                    )
                st.info("✅ 여행 일정표 생성 완료")
            
            # Step B: Evaluator Agent 평가
            with col_eval:
                st.markdown(f"**🕵️ Persona Evaluator Agent (Turn {turn})**")
                with st.spinner("페르소나 관점 검토 중..."):
                    eval_result = evaluator.evaluate(matched_persona, user_info, current_itinerary)
                
                final_score = eval_result.get("score", 70)
                satisfaction = eval_result.get("satisfaction", "")
                critique = eval_result.get("critique", "")
                
                st.metric("만족도 점수", f"{final_score} / 100점", delta=f"{final_score - target_score} (목표: {target_score}점)")
                st.caption(f"👍 **만족 요소:** {satisfaction}")
                st.caption(f"💡 **개선 요구:** {critique}")
                
                last_feedback = f"점수: {final_score}점 | 만족: {satisfaction} | 피드백: {critique}"

            # Termination Condition Check (하네스 수락 기준)
            if final_score >= target_score:
                st.success(f"🎉 Turn {turn}에서 목표 만족도 점수({target_score}점)를 달성하여 하네스 루프를 조기 종료합니다!")
                break
            elif turn < max_iterations:
                st.warning(f"⚠️ 목표 점수({target_score}점) 미달로 Planner Agent에게 피드백을 전달하여 재기획을 수행합니다.")

    # 4) 최종 결과 화면 출력
    st.markdown("---")
    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.subheader(f"🏆 최종 검증된 [{selected_region}] {travel_duration} 추천 코스")
        st.markdown(current_itinerary)

    with col_right:
        st.subheader("📍 카카오 추천 명소 & 실제 주소")
        for idx, place in enumerate(real_places):
            with st.container(border=True):
                st.markdown(f"#### {idx+1}. {place['title']}")
                st.markdown(f"📍 **실제 주소:** `{place['addr']}`")
                st.caption(f"📞 {place['tel']}")
                
                encoded_addr = urllib.parse.quote(place['addr'])
                address_map_url = f"https://map.kakao.com/link/search/{encoded_addr}"
                st.markdown(f"[🔗 카카오맵에서 위치 및 경로 보기]({address_map_url})")
