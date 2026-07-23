# 🧩 Nemotron-Personas 기반 AI 동호회 추천 플랫폼

본 프로젝트는 Hugging Face의 `nvidia/Nemotron-Personas-Korea` 데이터셋과 **SentenceTransformers 기반 벡터 검색(RAG)**을 활용하여 사용자의 성향, 지역, 관심사에 최적화된 동호회를 매칭해주는 **Streamlit 웹 애플리케이션**입니다.

## 🌟 주요 기능
1. **유저 라이프스타일 입력**: 연령, 지역, 세부 관심사(운동, 예술, 여행, 맛집), 성향 서술
2. **벡터 유사도 매칭 (Vector Search)**: `jhgan/ko-sroberta-multitask` 한국어 임베딩 모델 기반 코사인 유사도 분석
3. **하이브리드 필터링**: 지역 하드 필터링 + 취향 의도 검색 결합
4. **LLM 기반 맞춤 추천서 생성**: Gemini API 연동 시 AI 동호회 초대장 및 추천 코스 자동 작성