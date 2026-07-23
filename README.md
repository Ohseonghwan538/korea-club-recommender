# ✈️ Nemotron-Personas X TourAPI 성향 맞춤 여행 코스 추천 시뮬레이터

본 프로젝트는 Hugging Face의 `nvidia/Nemotron-Personas-Korea` 페르소나 데이터셋과 **한국관광공사 TourAPI 4.0(국문 관광정보 서비스)** 실시간 OpenAPI 데이터를 결합하여 사용자의 성향과 취향에 맞춘 **여행 코스(Itinerary)**를 자동 생성하는 Streamlit 웹 애플리케이션입니다.

## 🌟 주요 기능
1. **페르소나 임베딩 매칭**: Nemotron 데이터셋에서 유저와 취향이 가장 유사한 한국인 페르소나 매칭.
2. **실시간 TourAPI 연동**: 한국관광공사 OpenAPI를 통해 전국 시/도 관광지, 문화시설, 맛집, 레포츠 실시간 수집.
3. **Gemini AI 여행 코스 생성**: 시간 순서별 당일치기/1박 2일 동선 및 개인화 추천 사유 자동 작성.
4. **Secrets 보안 지향**: API Key를 코드에 노출하지 않고 `st.secrets`로 안전하게 관리.
