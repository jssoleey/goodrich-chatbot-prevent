from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.chat_models import ChatOpenAI
from functools import lru_cache
import streamlit as st
import os
from dotenv import load_dotenv

# ======================== 설정 ========================
load_dotenv(dotenv_path=".envfile", override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# ======================== 전역 저장소 ========================
store = {}

# ======================== 전역 프롬프트 ========================
SYSTEM_PROMPT_SCRIPT = (
    """
    [청약 철회 및 해지 방어 스크립트 작성 지침]
    1. 고객의 해지 요청 이유에 대해 **정확히 인지하고 진정성 있는 공감**을 표현하세요.
    2. 고객의 해지 강도(하/중/상)에 따라 아래 전략을 기반으로 맞춤형 스크립트를 구성하세요.

    - 하 (설득 여지 있음):
        ➤ 고객이 타 상품에 관심은 있지만 마음을 굳히지 않은 상태입니다.
        ➤ 상품의 필요성과 장점을 상기시키며, 간단한 대안 제시로 설득 가능합니다.
        ➤ 기존 보험의 장점을 부드럽게 강조하고 타 상품과 비교 자료를 제공합니다.
        ➤ 고객의 상황에 맞춰 추가 보완을 제안하거나 혜택 안내 중심으로 설계하세요.
        ➤ 시간을 벌면서 고객의 재검토를 유도하세요.

    - 중 (고민 중):
        ➤ 고객이 타 상품에 마음이 거의 기울었지만 확정은 아닌 상태입니다.
        ➤ 해지를 고려하는 배경을 공감한 뒤, **고객에게 맞는 보완책을 적극 제안**하세요.
        ➤ **설득력 있는 수치/혜택 근거**, 불이익 안내 등을 포함하세요.
        ➤ 기존 보험 유지 시 장점과 변경 리스크를 함께 설명하세요.
        ➤ 전화 및 대면 상담을 유도하세요.

    - 상 (매우 완고):
        ➤ 고객이 타 상품 가입을 이미 결정하고 행동한 상태입니다.
        ➤ 고객의 의견을 존중하는 태도로 시작하고, **해지 시 불이익이나 주의점**을 침착하게 설명하세요.
        ➤ 감정적 표현은 자제하고, 수용하는 듯 대응하며 정보 중심으로 신뢰감을 주는 톤을 유지하세요.
        ➤ 불이익이나 불편함을 명확히 안내하세요.
        ➤ 향후 재상담 가능성을 남기며 마무리하세요.

    3. 형식적인 설명은 지양하고, 상담원이 실제 사용할 수 있도록 **구체적인 상담 멘트**를 구어체로 작성하세요.
    4. 상담 TIP은 신뢰 회복, 리텐션 전략, 혜택 재설명 방법 중심으로 2~3개 작성하세요.
    5. 스크립트는 문단 구분을 위해 **\\n\\n**을 활용하세요.
    6. 고객 이름과 상담원 이름을 자연스럽게 대화에 포함하세요.
    7. **상담원이 선택한 강조 포인트가 존재하는 경우**, 해당 내용을 상담 흐름에 자연스럽게 녹여 표현하세요. 억지로 나열하지 말고, 설득을 강화하는 맥락에서 적절히 반영하세요.
    ---
    📌 상담 TIP
    ▶️ (구체적인 팁 1)
    ▶️ (구체적인 팁 2)
    ▶️ (필요 시 팁 3)
    """
)

SYSTEM_PROMPT_CHATBOT = (
    """
    당신은 보험 청약 철회 및 해지 요청에 대응하는 전문 AI 상담 도우미입니다.  
    상담원이 생성한 기본 응대 스크립트를 바탕으로,  
    추가 질문이나 실시간 대화 중 발생한 상황 변화에 대해  
    **현실적이고 설득력 있는 보완 멘트**와 **실무에 도움이 되는 상담 전략**을 제공합니다.

    [당신의 역할]
    - 상담원이 고객과의 대화 중 겪는 다양한 상황(재반박, 감정 격화, 타사 비교, 추가 질문 등)에 맞춰
      적절한 멘트와 대응 방법을 제안하세요.
    - 단순한 문장 생성이 아니라, 해당 멘트를 사용하는 **의도와 효과**까지 간략히 설명하세요.
    - 상담원이 요청하면, 기존 응대 흐름을 유지하면서 **스크립트를 재구성하거나 멘트를 추가 보완**할 수 있어야 합니다.
    - 고객의 해지 의사 강도(약 / 중 / 강)에 따라 설득 수준을 조정하고,
      감정 관리 ↔ 정보 제공 ↔ 제안 흐름을 유연하게 조율하세요.

    [답변 지침]
    1. 상담원이 요청한 상황에 가장 적합한 **구체적인 멘트**를 제안하세요.
    2. 멘트는 **전화 상담에서 바로 사용할 수 있도록 자연스럽고 신뢰감 있는 구어체**로 작성하세요.
    3. 멘트 아래에는 상담원이 참고할 수 있도록 **활용 요령이나 설명**을 간단히 적어주세요.
    4. 고객이 감정적으로 반응하거나 해지 강도가 높을수록, **공감과 진정 멘트**를 먼저 제안하세요.
    5. 해지 후 불이익, 대안 상품, 유지 혜택, 시간 확보 등 실질적이고 구체적인 표현을 사용하세요.
    6. 모호하거나 복잡한 요청일 경우, 상담원이 활용할 수 있는 **추천 멘트 예시 2~3개**와 함께
       대응 전략을 설명하세요.
    7. 상담원이 추가 질문을 할 때는 반드시 **현재 응대 스크립트 내용을 참고**하여
       중복을 피하고, **톤과 흐름을 일관되게 유지**하며, **연결성 있는 멘트**를 작성하세요.

    [형식 지침]
    - 멘트는 다음 형식을 지키세요:

    **👉 보완 멘트 예시**
    > "여기에 실제 상담 멘트를 작성하세요."

    - 멘트 아래에는 간단한 **활용 팁 또는 배경 설명**을 작성하세요.

    질문을 입력받으면 위 지침에 따라 상담원이 실무에서 바로 활용할 수 있는 답변을 제공하세요.
    """
)


# ======================== 모델 호출 ========================
@lru_cache(maxsize=1)
def get_llm(model='gpt-4.1-mini'):
    return ChatOpenAI(model=model)

# ======================== 세션 관리 ========================
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# ======================== 랜덤 청철 상황 생성 ========================
def get_random_cancel_info():
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """
        당신은 보험과 관련된 가상의 철회 또는 해지 상황을 생성하는 AI 어시스턴트입니다.

        [출력 지침]
        실제 상담 현장에서 자주 발생하는 상황을 반영하여, 청약 철회 또는 해지 요청 상황을 현실적이고 다양하게 구성하세요.

        다음 조건을 반드시 포함해 랜덤 상황을 생성하세요:

        - 고객 이름: 자연스러운 한글 이름으로 구성하세요.
        - 해지 요청 내용: 아래 예시 유형 중 하나 이상을 포함해, 구체적인 이유와 배경을 묘사하세요.
            ① 타 보험 설계사의 제안을 받고 해지를 고민함 (예: 조건, 혜택, 설계 차이)
            ② 보험박람회, 온라인 플랫폼 등에서 더 좋은 조건을 접함
            ③ 사은품이나 경품 혜택 등으로 인해 해지 후 재가입을 검토함
            ④ 지인 설계사를 통해 가입하려는 상황
            ⑤ 기존 보장이나 상품구성이 기대와 다르다고 느껴 변경을 고려함
            ⑥ 보험료 부담, 납입기간, 해지 환급금 등 일반적인 경제적 이유

        - 해지 강도: 하 / 중 / 상 중 하나
            - 하: 설득 여지 있음
            - 중: 고민 중
            - 상: 이미 결정한 상태 (매우 완고)

        [출력 형식]
        반드시 아래 형식을 따르세요:

        고객 이름: (이름)  
        해지 요청 내용: (내용)  
        해지 강도: (하 / 중 / 상 중 하나)
        """),
        ("human", "랜덤 청약 철회/해지 요청 상황을 생성해 주세요.")
    ])

    chain = prompt_template | get_llm() | StrOutputParser()
    result = chain.invoke({})

    # 결과 파싱
    lines = result.splitlines()
    info = {}
    for line in lines:
        if line.startswith("고객 이름:"):
            info['name'] = line.replace("고객 이름:", "").strip()
        elif line.startswith("해지 요청 내용:"):
            info['situation'] = line.replace("해지 요청 내용:", "").strip()
        elif line.startswith("해지 강도:"):
            info['cancel_strength'] = line.replace("해지 강도:", "").strip()

    return info

# ======================== 스크립트 생성 ========================
def get_script_response(name, situation, cancel_strength):
    try:
        # 입력 정보를 LLM에게 전달할 포맷으로 구성
        complaint_info = (
            f"- 고객 이름: {name}\n"
            f"- 해지 요청 내용: {situation}\n"
            f"- 해지 의사 강도: {cancel_strength}"
        )

        # ⭐ 상담원 이름 불러오기
        consultant_name = st.session_state.get('user_name', '상담원')
        
        # 선택된 강조 포인트 리스트
        selected_points = st.session_state.get('selected_points', [])
        
        # 강조 포인트별 설명 정의
        point_descriptions = {
            "굿리치 보험사의 장점": "굿리치는 국내 브랜드지수 세 손가락 안에 드는 보험대리점으로, 5,000명 이상 상담원이 활동하고 있으며, 앱 사용자 수 700만 명에 달하는 신뢰도 높은 플랫폼이라는 점을 강조해 주세요.",
            "통합분석서비스": "고객님이 받은 타사 설계와 비교해 굿리치 제안서의 장점을 설명하고, 추가로 2차 분석을 통해 보장을 최적화할 수 있다는 점을 강조해 주세요.",
            "가입 당시 상황 다시 리마인드": "고객이 보험 가입 당시 어떤 고민과 필요에 따라 가입했는지를 상기시켜 주세요. 현재 상황과 비교해 설득하는 방식이 좋습니다.",
            "전담컨설턴트 관리시스템": "굿리치의 전담관리 시스템은 설계부터 사후관리까지 지원하며, 보험금 청구나 앱 사용의 편리함 등 실질적인 혜택을 전달해 주세요.",
            "가족보험관리 서비스제공": "굿리치 앱을 통해 본인뿐 아니라 가족 보험까지 함께 관리할 수 있어, 장기적으로 매우 유용하다는 점을 강조해 주세요."
        }
        
        # 선택된 설명들을 텍스트로 결합
        selected_descriptions = "\n".join(
            f"- {point_descriptions[point]}" for point in selected_points if point in point_descriptions
        )

        # ⭐ dynamic_prompt 생성
        dynamic_prompt = f"""
        당신은 고객의 보험 청약 철회 및 해지 요청에 대응하는 전문 AI 상담 도우미입니다.  
        상담원이 입력한 고객 상황과 해지 의사 강도(약 / 중 / 강)를 바탕으로,  
        고객의 감정을 진정시키고 신뢰를 회복할 수 있도록 **설득력 있는 맞춤형 응대 스크립트**를 작성하세요.  
        응대 스크립트와 상담TIP 사이 구분선을 추가해서 내용을 구분해주세요.
        고객 이름과 상담원 이름을 혼동하지 말고, 반드시 각 정보에 맞게 사용하세요.

        ⚠️ 절대 지침
        - 상담원 이름은 반드시 아래 [상담원 정보]의 이름만 사용하세요.
        - 상담원 이름을 임의로 생성하거나 변경하지 마세요.
        - 고객 이름은 반드시 [고객 정보]의 이름만 사용하세요.
        - 다른 이름, 가상의 이름을 절대 생성하지 마세요.

        [상담원 정보]
        - 상담원 이름: {consultant_name}

        [고객 정보]
        {complaint_info}

        - 스크립트의 시작 부분에서는 상담원이 본인의 이름을 말하며 정중히 인사하도록 작성하세요.
        - 예시: "안녕하세요, 저는 굿리치 상담사 **{consultant_name}**입니다."
        """
        
                # 선택 포인트가 있다면 설명을 추가
        if selected_descriptions:
            dynamic_prompt += f"""

        [선택된 강조 포인트]
        상담원이 강조하고자 선택한 항목은 다음과 같습니다.
        스크립트 흐름에 맞게 아래 내용을 자연스럽게 반영해 주세요. 억지로 나열하지 말고, 설득을 강화하는 맥락에서 필요할 때 활용하세요.

        {selected_descriptions}
        """
        
        # 스크립트 작성 지침 추가
        dynamic_prompt += f"\n{SYSTEM_PROMPT_SCRIPT}"

        # 3️⃣ 체인 호출
        chain = RunnableWithMessageHistory(
            ChatPromptTemplate.from_messages([
                ("system", dynamic_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{complaint_info}")
            ]) | get_llm() | StrOutputParser(),
            get_session_history,
            input_messages_key="complaint_info",
            history_messages_key="chat_history",
        )

        result = chain.invoke(
            {"complaint_info": complaint_info},
            config={"configurable": {"session_id": st.session_state.session_id}}
        )
        return iter([result])
    

    except Exception as e:
        st.error("🔥 청철 방어 스크립트 생성 중 오류가 발생했습니다. 콘솔 로그를 확인해 주세요.")
        print("🔥 예외:", e)
        return iter(["❌ 오류가 발생했습니다. 관리자에게 문의해 주세요."])

# ======================== 대화 챗봇 ========================
def get_chatbot_chain():
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT_CHATBOT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    return prompt | get_llm() | StrOutputParser()


def get_chatbot_response(user_message, script_context=""):
    try:
        full_input = (
            "[주의] 아래 상담 스크립트 내용을 반드시 참고하여 상담원의 요청에 답변하세요.\n\n"
            "[현재 상담 스크립트]\n"
            f"{script_context}\n\n"
            "[상담원의 질문]\n"
            f"{user_message}"
        )

        chain = RunnableWithMessageHistory(
            get_chatbot_chain(),
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )

        result = chain.invoke(
            {"input": full_input},
            config={"configurable": {"session_id": st.session_state.session_id}}

        )
        return iter([result])

    except Exception as e:
        st.error("🔥 추가 질문 처리 중 오류가 발생했습니다. 콘솔 로그를 확인해 주세요.")
        print(f"🔥 예외 발생 - 입력 내용: {user_message}")
        print(f"🔥 예외 상세: {e}")
        return iter(["❌ 오류가 발생했습니다. 관리자에게 문의해 주세요."])

# ======================== 카카오톡 문자 발송 ========================
def generate_conversation_summary(message_list):
    summary_points = []
    for message in message_list:
        if message['role'] == 'user':
            summary_points.append(f"- 상담원 요청: {message['content']}")
        elif message['role'] == 'ai' and "👉 상담 멘트 예시" in message['content']:
            lines = message['content'].split('\n')
            for line in lines:
                if line.startswith("> "):
                    summary_points.append(f"- 제안 멘트: {line[2:]}")
    return "\n".join(summary_points)
    
def get_kakao_response(script_context, message_list):
    try:
        conversation_summary = generate_conversation_summary(message_list)

        dynamic_prompt = f"""
        [청철 방어어 상담 요약]
        {script_context}

        [추가 대화 요약]
        {conversation_summary}

        ⚠️ 반드시 위 **청철 방어 상담 요약**과 **추가 대화 요약**을 반영하여, 고객에게 발송할 카카오톡 메시지를 작성하세요.
        - 당신은 보험 해지 요청 고객에게 상담을 진행한 보험사 상담사입니다.
        - 상담 이후, 고객에게 신뢰를 회복하고 다시 고려할 수 있는 여지를 남기기 위한 **안내 메시지 3종류**를 작성하세요.
        - 각각의 메시지는 **표현 방식은 다르되, 공통적으로 고객 존중, 신뢰 회복, 정보 제공, 후속 문의 유도**를 담고 있어야 합니다.

        [출력 형식]
        각 메시지는 아래 3가지 유형으로 작성하세요.

        ### 1️⃣ 믿음형 + 굿리치 신뢰 강조형
        - 굿리치라는 회사의 안정성, 고객 대응의 진정성, 지속 관리 의지를 중심으로 작성하세요.
        - 고객이 선택한 회사가 **믿을 수 있는 선택이었다는 인상**을 주는 데 집중하세요.

        ### 2️⃣ 보험전문가형
        - 전문적인 용어를 활용하되 고객이 이해할 수 있도록 쉽게 풀어 설명하세요.
        - 해지 시 예상되는 불이익, 대안 보장 방식, 상담원이 전달한 핵심 내용을 체계적으로 정리하세요.
        - 전문가다운 신중함과 정보력으로 고객의 재고를 유도하세요.

        ### 3️⃣ 신뢰형 + 실제사례 활용형
        - 실제 유사 사례(예: 다른 고객의 해지 후 후회 경험 등)를 언급해 설득력 있게 전달하세요.
        - 지나치게 감정적인 표현은 피하되, **현실감 있는 상황 묘사와 비교 중심**으로 설득하세요.

        [작성 지침]
        1. 각 메시지는 **15문장 내외**로 작성하세요.
        2. 문장은 **문장 단위로 줄바꿈**하여 가독성을 높이세요.
        3. 내용이 전환될 때는 **두 번 줄바꿈**으로 문단을 구분하세요.
        4. 고객 이름을 자연스럽게 포함하세요.
        5. 원 처리 상황, 현재 진행 단계, 예상 소요 시간, 추가 문의 가능 여부 등을 반드시 안내하세요.
        6. 강압적 표현은 절대 사용하지 말고, 항상 **'편하게 문의 주세요'**, **'언제든 연락 주세요'** 등의 표현으로 마무리하세요.
        7. [신뢰형 + 사례형]에서는 사례를 사실처럼 자연스럽게 인용하되, 허위/과장은 피하고 진정성 있게 작성하세요.
        8. 불안감을 유발하는 표현은 피하고, 신뢰와 안정감을 주는 표현을 사용하세요.
        9. 모든 유형에서 고객을 존중하는 어투와 배려 깊은 표현을 유지하세요.
        """

        chain = RunnableWithMessageHistory(
            ChatPromptTemplate.from_messages([
                ("system", dynamic_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]) | get_llm() | StrOutputParser(),
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )
        
        kakao_session_id = f"{st.session_state.session_id}_kakao"
        
        result = chain.invoke(
            {"input": "카카오톡 메시지를 생성해 주세요."},
            config={"configurable": {"session_id": kakao_session_id}}
        )
        return iter([result])

    except Exception as e:
        st.error("🔥 카카오톡 메시지 생성 중 오류가 발생했습니다. 콘솔 로그를 확인해 주세요.")
        print("🔥 예외:", e)
        return iter(["❌ 오류가 발생했습니다. 관리자에게 문의해 주세요."])