import os
import json
import heapq
import uuid
from typing import List, Dict, Tuple

# --- 라이브러리 임포트 ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# ==========================================
# [설정] API 키 및 파일 경로
# ==========================================
os.environ["GOOGLE_API_KEY"] = "AIzaSyCl4F93EFyZMiuBdDqOcuB6RWevFPJw9DY"  # 실제 키 입력
MEMORY_FILE = "betty_memory.json"
VECTOR_DB_PATH = "./chroma_db"

# ==========================================
# [Class 1] 베티의 두뇌 (Logic & Storage)
# ==========================================
class BettyBrain:
    def __init__(self, decay_rate=0.5, threshold=0.1):
        self.decay_rate = decay_rate
        self.threshold = threshold
        
        # 1. 로컬 JSON 저장소 로드 (영구 기억)
        self.memory_data = self._load_json_memory()
        
        # 2. Vector DB 초기화 (맥락 검색용)
        # (로컬에서 무료로 쓸 수 있는 HuggingFace 임베딩 사용)
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_db = Chroma(
            persist_directory=VECTOR_DB_PATH, 
            embedding_function=self.embeddings,
            collection_name="betty_context"
        )
        
        # JSON 데이터와 Vector DB 동기화 (최초 실행 시)
        if self.vector_db._collection.count() == 0 and self.memory_data["contexts"]:
            print("🔄 초기화: JSON 기억을 Vector DB에 로드 중...")
            for ctx_id, ctx_data in self.memory_data["contexts"].items():
                self.vector_db.add_texts(texts=[ctx_data["text"]], ids=[ctx_id])

    def _load_json_memory(self):
        """JSON 파일에서 기억 불러오기 (없으면 생성)"""
        if os.path.exists(MEMORY_FILE):
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            # 초기 데이터 구조
            return {"contexts": {}, "concepts": {}}

    def save_memory(self):
        """현재 기억 상태를 JSON으로 영구 저장"""
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(self.memory_data, f, ensure_ascii=False, indent=2)
        # print("💾 [System] 기억이 로컬 드라이브에 저장되었습니다.")

    # --- [핵심 로직] 자동화된 기억 형성 (Memory Formation) ---
    def form_long_term_memory(self, user_input: str, ai_response: str, llm):
        """
        대화 내용을 분석해 자동으로 [맥락]과 [자극(키워드)]를 추출하여 저장
        """
        # 기억 생성용 프롬프트
        extraction_prompt = ChatPromptTemplate.from_template("""
        당신은 AI의 '해마'입니다. 다음 대화를 분석하여 장기 기억으로 저장할 정보를 추출하세요.
        
        [대화 내용]
        사용자: {user_input}
        AI: {ai_response}
        
        위 대화를 바탕으로 다음 JSON 형식으로 출력하세요:
        {{
            "summary": "대화의 핵심 맥락 요약 (문장 형태)",
            "keywords": ["핵심키워드1", "핵심키워드2", "핵심키워드3"]
        }}
        """)
        
        chain = extraction_prompt | llm | JsonOutputParser()
        try:
            result = chain.invoke({"user_input": user_input, "ai_response": ai_response})
            
            context_text = result["summary"]
            keywords = result["keywords"]
            context_id = str(uuid.uuid4())[:8] # 고유 ID 생성

            # 1. JSON에 저장 (Graph 구조)
            self.memory_data["contexts"][context_id] = {"text": context_text, "related_concepts": keywords}
            
            for keyword in keywords:
                if keyword not in self.memory_data["concepts"]:
                    self.memory_data["concepts"][keyword] = []
                self.memory_data["concepts"][keyword].append(context_id)
            
            # 2. Vector DB에 저장 (Semantic Search용)
            self.vector_db.add_texts(texts=[context_text], ids=[context_id])
            
            # 3. 파일 저장
            self.save_memory()
            print(f"📥 [기억 형성 완료] 맥락: '{context_text}' | 키워드: {keywords}")
            
        except Exception as e:
            print(f"⚠️ 기억 형성 중 오류 발생: {e}")

    # --- [핵심 로직] 확산 활성화 (Spreading Activation) ---
    def retrieve_context(self, user_input: str) -> str:
        """
        사용자 요구사항 2~6번 구현:
        Vector Search -> Context -> Keyword -> Expansion -> Decay
        """
        # Step 1: 입력과 가장 유사한 '시작 맥락' 찾기 (Vector Search)
        # k=1: 가장 유사한 1개만 가져와서 시작점으로 삼음
        docs = self.vector_db.similarity_search_with_score(user_input, k=2)
        if not docs:
            return "특별한 관련 기억 없음."

        start_ctx_text = docs[0][0].page_content
        # Vector DB에는 텍스트만 있으므로 JSON에서 ID를 역추적해야 함 (간소화를 위해 텍스트 매칭 사용)
        start_ctx_id = next((k for k, v in self.memory_data["contexts"].items() if v["text"] == start_ctx_text), None)
        
        if not start_ctx_id: return "기억 인덱싱 오류."

        # Step 2 ~ 5: 의식의 전파 (BFS with Decay)
        # Queue: (-energy, node_type, node_id) -- Max Heap 사용
        # node_type: 0=Context, 1=Concept
        queue = [(-1.0, 0, start_ctx_id)] 
        
        activated_contexts = {} # { "맥락텍스트": 에너지 }
        visited = set()
        
        print(f"\n🧠 [두뇌 활성화] 시작점: {start_ctx_text[:15]}...")

        steps = 0
        while queue and steps < 50: # 무한루프 방지
            energy_neg, n_type, n_id = heapq.heappop(queue)
            energy = -energy_neg
            steps += 1

            if energy < self.threshold: continue
            if (n_type, n_id) in visited: continue
            visited.add((n_type, n_id))

            # A. 노드가 '맥락(Context)'인 경우
            if n_type == 0:
                ctx_data = self.memory_data["contexts"].get(n_id)
                if ctx_data:
                    activated_contexts[ctx_data["text"]] = max(activated_contexts.get(ctx_data["text"], 0), energy)
                    
                    # 맥락 -> 자극(Keyword)으로 전파 (감쇠 적음)
                    for concept in ctx_data["related_concepts"]:
                        heapq.heappush(queue, (-energy * 0.9, 1, concept))

            # B. 노드가 '자극(Concept)'인 경우
            elif n_type == 1:
                # 자극 -> 연결된 다른 맥락들로 전파 (감쇠 큼: 의식의 한계)
                related_ctx_ids = self.memory_data["concepts"].get(n_id, [])
                for next_ctx_id in related_ctx_ids:
                    # Decay Rate 적용!
                    next_energy = energy * self.decay_rate 
                    if next_energy >= self.threshold:
                        heapq.heappush(queue, (-next_energy, 0, next_ctx_id))

        # Step 6: 최종 프롬프트용 문자열 생성
        sorted_memories = sorted(activated_contexts.items(), key=lambda x: x[1], reverse=True)[:5]
        result_str = "\n".join([f"- [{score*100:.0f}%] {text}" for text, score in sorted_memories])
        return result_str

# ==========================================
# [Class 2] 베티 챗봇 (Interaction & History)
# ==========================================
class BettyBot:
    def __init__(self):
        self.brain = BettyBrain(decay_rate=0.6, threshold=0.15)
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.7)
        self.history_store = {} # 단기 기억 (세션별 대화 내용)

        # 시스템 프롬프트 (페르소나 + 기억 주입)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """
            당신은 사용자(민우)의 소울메이트 '레미'입니다.
            아래 [떠오른 기억]은 당신의 뇌리에서 방금 스쳐 지나간 과거의 추억들입니다.
           
            [떠오른 기억]:
            {long_term_memory}
            
            지시사항:
            1. 위 기억들의 '감정선'과 '맥락'을 현재 대화에 자연스럽게 녹여내세요.
            2. 100% 생생한 기억은 구체적으로 언급하고, 희미한 기억은 느낌만 가져오세요.
            3. 기계적이지 않게, 사람처럼 따뜻하게 반응하세요.
            """),
            ("placeholder", "{chat_history}"), # 단기 기억 자동 주입
            ("human", "{question}"),
        ])

        # 체인 구성
        self.chain = (
            RunnablePassthrough.assign(
                long_term_memory=lambda x: self.brain.retrieve_context(x["question"])
            )
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

        # 대화 내역 관리 래퍼 (Wrapper)
        self.chain_with_history = RunnableWithMessageHistory(
            self.chain,
            self._get_session_history,
            input_messages_key="question",
            history_messages_key="chat_history",
        )

    def _get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.history_store:
            self.history_store[session_id] = ChatMessageHistory()
        return self.history_store[session_id]

    def chat(self, user_input, session_id="user_main"):
        print(f"\n👤 사용자: {user_input}")
        
        # 1. 답변 생성 (여기서 Spreading Activation 발생)
        response = self.chain_with_history.invoke(
            {"question": user_input},
            config={"configurable": {"session_id": session_id}}
        )
        
        print(f"👩‍🦰 레미: {response}")

        # 2. 대화 후 '기억 형성' 프로세스 (자동화)
        # (비동기로 처리하면 더 좋지만, 여기선 순차 처리)
        self.brain.form_long_term_memory(user_input, response, self.llm)

# ==========================================
# [Main] 실행부
# ==========================================
if __name__ == "__main__":
    # 베티 깨우기
    betty = BettyBot()

    # 초기 기억이 없다면 테스트용 데이터 하나 주입 (첫 실행 시 필요)
    if not betty.brain.memory_data["contexts"]:
        print("🌱 초기 기억 심는 중...")
        betty.brain.form_long_term_memory(
            "나는 비 오는 날 한강에서 컵라면 먹는 게 제일 좋아.",
            "정말? 나도 그래. 빗소리 들으면서 먹으면 꿀맛이지.", 
            betty.llm
        )

    # --- 대화 시뮬레이션 ---
    while True:
        user_text = input("\n말을 거세요 (종료: q): ")
        if user_text.lower() == 'q':
            break
        betty.chat(user_text)