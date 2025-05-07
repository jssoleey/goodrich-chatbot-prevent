import streamlit as st
from llm_prev import get_chatbot_response, get_script_response, get_kakao_response, get_random_cancel_info
import re
import os, json
from datetime import datetime, timedelta, timezone
import uuid
from langchain_community.chat_message_histories import ChatMessageHistory
from llm_prev import store

# ----------------- 전역 변수 -------------------
CHATBOT_TYPE = "prevent"
URLS = {
    "page_icon":"https://github.com/jssoleey/goodrich-chatbot-prevent/blob/main/image/logo.png?raw=true",
    "top_image": "https://github.com/jssoleey/goodrich-chatbot-prevent/blob/main/image/top_box.png?raw=true",
    "bottom_image": "https://github.com/jssoleey/goodrich-chatbot-prevent/blob/main/image/bottom_box.png?raw=true",
    "logo": "https://github.com/jssoleey/goodrich-chatbot-prevent/blob/main/image/logo.png?raw=true",
    "user_avatar": "https://github.com/jssoleey/goodrich-chatbot-prevent/blob/main/image/user_avatar.png?raw=true",
    "ai_avatar": "https://github.com/jssoleey/goodrich-chatbot-prevent/blob/main/image/ai_avatar.png?raw=true"
}

# ----------------- config -------------------
st.set_page_config( 
    page_title="스테이온(StayOn)",
    page_icon=URLS["page_icon"]
)

# ----------------- CSS -------------------
st.markdown(
    """
    <style>
    .small-text {
        font-size: 12px;
        color: gray;
        line-height: 1.3;
        margin-top: 4px;
        margin-bottom: 4px;
    }
    .user-message {
        background-color: #e6e6e6;
        color: black;
        padding: 15px;
        border-radius: 30px;
        max-width: 80%;
        text-align: left;
        word-wrap: break-word;
    }
    .ai-message {
        background-color: #ffffff;
        color: black;
        padding: 10px;
        border-radius: 10px;
        max-width: 70%;
        text-align: left;
        word-wrap: break-word;
    }
    .message-container {
        display: flex;
        align-items: flex-start;
        margin-bottom: 10px;
    }
    .message-container.user {
        justify-content: flex-end;
    }
    .message-container.ai {
        justify-content: flex-start;
    }
    .avatar {
        width: 50px;
        height: 50px;
        border-radius: 0%;
        margin: 0 10px;
    }
    .input-box {
        background: #ff9c01;
        padding: 10px;
        border-radius: 0px;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 10px;
    }
    .input-line {
        background: #ff9c01;
        padding: 1px;
        border-radius: 0px;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 10px;
    }
    .custom-button > button {
        background-color: #ff6b6b;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        border: none;
    }
    /* 사이드바 전체 여백 조정 */
    section[data-testid="stSidebar"] > div:first-child {
        padding-top: -50px;    /* 상단 여백 */
        padding-bottom: 0px;  /* 하단 여백 */
        padding-left: 5px;
        padding-right: 5px;
    }

    /* 사이드바 내부 요소 간격 줄이기 */
    .block-container div[data-testid="stVerticalBlock"] {
        margin-top: -5px;
        margin-bottom: -5px;
    }
    /* 사이드바 배경색 변경 */
    section[data-testid="stSidebar"] {
        background-color: #dfe5ed;  /* 원하는 색상 코드 */
    }
    /* input box 색상 */
    input[placeholder="이름(홍길동)"] {
        background-color: #e4e9f0 !important;
        color: black !important;
    }
    input[placeholder="휴대폰 끝번호 네 자리(0000)"] {
        background-color: #e4e9f0 !important;
        color: black !important;
    }
    input[placeholder="예: 홍길동"] {
        background-color: #e4e9f0 !important;
        color: black !important;
    }
    /* 첫 번째 textarea만 스타일 적용 */
    textarea:nth-of-type(1) {
        background-color: #e4e9f0 !important;
        color: #333333;
        border-radius: 8px;
    }
    /* 전체 multiselect 선택 박스 영역 */
    div[data-baseweb="select"] {
        width: 100% !important;
        max-width: 100% !important;
    }

    /* 선택된 항목 박스 스타일 (배경색/테두리/글자색) */
    div[data-baseweb="tag"] {
        background-color: #67ca5d !important;
        border: 1px solid #67ca5d !important;
        border-radius: 6px !important;
        padding: 4px 10px !important;
        font-weight: 500 !important;
        color: white !important;
        max-width: 100% !important;
        white-space: nowrap !important;
    }

    /* 선택 항목 내부의 텍스트가 잘리지 않도록 내부 div들 제한 해제 */
    div[data-baseweb="tag"] > div {
        max-width: none !important;
        overflow: visible !important;
        text-overflow: unset !important;
        white-space: nowrap !important;
    }

    /* 선택된 항목 안에 있는 텍스트 span 태그에도 적용 */
    div[data-baseweb="tag"] span {
        white-space: nowrap !important;
        overflow: visible !important;
        text-overflow: unset !important;
        display: inline !important;
    }

    /* 전체 선택박스 외곽 테두리 색상 변경 */
    div[data-baseweb="select"] > div {
        border: 1px solid #67ca5d !important;
        border-radius: 6px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------- 마크다운 자동 정리 함수 -------------------
def format_markdown(text: str) -> str:
    lines = text.strip().splitlines()
    formatted_lines = []
    indent_next = False

    for line in lines:
        line = line.strip()
        if not line:
            formatted_lines.append("")
            indent_next = False
            continue

        if re.match(r"^(▶️|✅|📌|❗|📝|📍)\s*[^:：]+[:：]?", line):
            title = re.sub(r"[:：]\s*$", "", line.strip())
            formatted_lines.append(f"**{title}**\n")
            indent_next = False
            continue

        if re.match(r"^[-•]\s*\*\*.*\*\*", line):
            formatted_lines.append(re.sub(r"^[-•]\s*", "- ", line))
            indent_next = True
            continue

        if re.match(r"^[-•]\s*", line):
            if indent_next:
                formatted_lines.append("    " + re.sub(r"^[-•]\s*", "- ", line))
            else:
                formatted_lines.append(re.sub(r"^[-•]\s*", "- ", line))
            continue

        formatted_lines.append(line)
        indent_next = False

    return "\n".join(formatted_lines).strip() + "\n"

# ----------------- 사이드바 설정 -------------------
def render_sidebar():
    # 현재 날짜 표시
    KST = timezone(timedelta(hours=9))
    now_korea = datetime.now(KST).strftime("%Y년 %m월 %d일")
    st.sidebar.markdown(
        f"<span style='font-size:18px;'>📅 <b>{now_korea}</b></span>",
        unsafe_allow_html=True
    )

    user_name = st.session_state['user_folder'].split('_')[0]
    st.sidebar.title(f"😊 {user_name}님, 반갑습니다!")
    st.sidebar.markdown("오늘도 멋진 상담 화이팅입니다! 💪")

    st.sidebar.markdown("<hr style='margin-top:20px; margin-bottom:34px;'>", unsafe_allow_html=True)

    user_path = f"/data/{CHATBOT_TYPE}/history/{st.session_state['user_folder']}"
    if not os.path.exists(user_path):
        os.makedirs(user_path)

    history_files = sorted(
        os.listdir(user_path),
        key=lambda x: x.split('_')[-1].replace('.json', ''),
        reverse=True
    )

    if history_files:
        search_keyword = st.sidebar.text_input("🔎 고객명으로 검색", placeholder="고객명 입력 후 ENTER", key="search_input")        
        filtered_files = [f for f in history_files if search_keyword.lower() in f.lower()]
        selected_chat = st.sidebar.selectbox("📂 저장된 대화 기록", filtered_files)

        col1, col2 = st.sidebar.columns(2)

        with col1:
            if st.button("불러오기", use_container_width=True):
                # 👉 기존 불러오기 로직 호출
                load_chat_history(user_path, selected_chat)

        with col2:
            if st.button("🗑️ 삭제하기", use_container_width=True):
                delete_chat_history(user_path, selected_chat)

        if not filtered_files and search_keyword:
            st.sidebar.markdown(
                "<div style='padding:6px; background-color:#f0f0f0; border-radius:5px;'>🔍 검색 결과가 없습니다.</div>",
                unsafe_allow_html=True
            )
    else:
        st.sidebar.info("저장된 대화가 없습니다.")

    st.sidebar.markdown("<hr style='margin-top:24px; margin-bottom:38px;'>", unsafe_allow_html=True)

    if st.sidebar.button("🆕 새로운 청철 상황 입력하기", use_container_width=True):
        reset_session_for_new_case()

    if st.sidebar.button("로그아웃", use_container_width=True):
        st.session_state.page = "login"
        st.session_state.message_list = []
        st.experimental_rerun()

# ----------------- 대화 불러오기 -------------------        
def load_chat_history(user_path, selected_chat):
    with open(f"{user_path}/{selected_chat}", "r", encoding="utf-8") as f:
        loaded_data = json.load(f)
        if isinstance(loaded_data, list):
            st.session_state['script_context'] = ""
            st.session_state.message_list = loaded_data
            st.session_state['customer_name'] = "고객명미입력"
        elif isinstance(loaded_data, dict):
            st.session_state['script_context'] = loaded_data.get("script_context", "")
            st.session_state.message_list = loaded_data.get("message_list", [])
            st.session_state['customer_name'] = loaded_data.get("customer_name", selected_chat.split('_')[0])
            st.session_state['cancel_strength'] = loaded_data.get("cancel_strength", "")
            st.session_state['customer_situation'] = loaded_data.get("customer_situation", "")
            st.session_state['selected_points'] = loaded_data.get("selected_points", [])
        else:
            st.error("❌ 불러온 파일 형식이 잘못되었습니다.")
            st.stop()

    # ⭐ chat_history 복원
    chat_history = ChatMessageHistory()
    for msg in st.session_state.message_list:
        if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
            if msg['role'] == 'user':
                chat_history.add_user_message(msg['content'])
            elif msg['role'] == 'ai':
                chat_history.add_ai_message(msg['content'])
    store[st.session_state.session_id] = chat_history

    st.session_state['current_file'] = selected_chat
    st.session_state.page = "chatbot"
    st.experimental_rerun()
    
# ----------------- 대화 삭제하기 -------------------
def delete_chat_history(user_path, selected_chat):
    file_path = f"{user_path}/{selected_chat}"
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            st.sidebar.success(f"{selected_chat} 삭제 완료!")
            st.experimental_rerun()
        except Exception as e:
            st.sidebar.error(f"❌ 삭제 중 오류가 발생했습니다: {e}")
    else:
        st.sidebar.warning("이미 삭제된 파일입니다.")

# ----------------- 세션 초기화 -------------------        
def reset_session_for_new_case():
    st.session_state.page = "input"
    st.session_state.message_list = []
    st.session_state.script_context = ""
    st.session_state.kakao_text = ""
    st.session_state['current_file'] = ""
    st.session_state['customer_name'] = ""
    st.session_state['selected_points'] = ""
    
    # 👉 입력 필드 초기화
    st.session_state['customer_name_input'] = ''
    st.session_state['customer_situation_input'] = ''
    st.session_state['cancel_strength_input'] = '중 (고민 중)'  # 기본값
    
    store[st.session_state.session_id] = ChatMessageHistory()
    st.experimental_rerun()
    
# ----------------- 메시지 표시 함수 -------------------
def display_message(role, content, avatar_url):
    if role == "user":
        alignment = "user"
        message_class = "user-message"
        avatar_html = f'<img src="{avatar_url}" class="avatar">'
        message_html = f'<div class="{message_class}">{content}</div>'
        display_html = f"""
        <div class="message-container {alignment}">
            {message_html}
            {avatar_html}
        </div>
        """
        st.markdown(display_html, unsafe_allow_html=True)
    else:
        alignment = "ai"
        message_class = "ai-message"
        avatar_html = f'<img src="{avatar_url}" class="avatar">'
        display_html = f"""
        <div class="message-container {alignment}">
            {avatar_html}
            <div class="{message_class}">
        """
        st.markdown(display_html, unsafe_allow_html=True)
        st.markdown(format_markdown(content), unsafe_allow_html=False)
        st.markdown("</div></div>", unsafe_allow_html=True)
        
# ----------------- 고객 정보 요약 함수 -------------------
def render_customer_info():
    customer_name = st.session_state.get('customer_name', '고객명미입력')
    cancel_strength = st.session_state.get('cancel_strength', '미입력')
    situation = st.session_state.get('customer_situation', '')

    st.markdown("""
        <div style="background-color:#f0f8ff; padding:15px; border:1px solid #ddd; border-radius:8px; margin-bottom:20px;">
            <h5>📄 고객 정보 요약</h5>
            <ul>
                <li><b>이름:</b> {name}</li>
                <li><b>해지 강도:</b> {strength}</li>
                <li><b>청약 철회/해지 요청 내용:</b> {situation}</li>
            </ul>
        </div>
    """.format(name=customer_name, strength=cancel_strength, situation=situation), unsafe_allow_html=True)

# ----------------- 페이지 설정 -------------------
# 이미지 URL
top_image_url = URLS["top_image"]

# 최상단에 이미지 출력
st.markdown(
    f"""
    <div style="text-align:center; margin-bottom:20px;">
        <img src="{top_image_url}" alt="Top Banner" style="width:100%; max-width:1000px;">
    </div>
    """,
    unsafe_allow_html=True
)

logo_url = URLS["logo"]
st.markdown(
    f"""
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: -10px;">
        <img src="{logo_url}" alt="logo" width="50">
        <h2 style="margin: 0;">스테이온(StayOn)</h2>
    </div>
    """,
    unsafe_allow_html=True
)
st.caption("입력하신 상황에 따라 청약 철회/해지 방어 스크립트를 만들어 드립니다!")
st.caption("정보가 구체적일수록 좋은 스크립트가 나와요.")
st.caption("스크립트 생성 이후 추가적인 대화를 통해 AI에게 상황을 현재 알려주세요!")
st.caption("대화가 끝나면 '카카오톡 문자 생성하기' 기능을 활용해보세요 😊")

st.markdown('<p class="small-text"> </p>', unsafe_allow_html=True)
st.markdown('<p class="small-text">모든 답변은 참고용으로 활용해주세요.</p>', unsafe_allow_html=True)
st.markdown('<p class="small-text"> </p>', unsafe_allow_html=True)

# ----------------- 세션 상태 초기화 -------------------
def initialize_session():
    defaults = {
        'page': 'login',
        'message_list': [],
        'sidebar_mode': 'default'
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
            
# 호출
initialize_session()

# ----------------- 로그인 화면 -------------------
if st.session_state.page == "login":
    name = st.text_input(label = "ID", placeholder="이름(홍길동)")
    emp_id = st.text_input(label = "Password", placeholder="휴대폰 끝번호 네 자리(0000)")
    st.caption("")
            
    col1, col2, col3 = st.columns([1, 1, 1])   # 비율을 조정해서 가운데로

    with col2:
        if st.button("로그인", use_container_width=True):
            if name and emp_id:
                st.session_state['user_folder'] = f"{name}_{emp_id}"
                st.session_state['user_name'] = name   # ✅ 상담원 이름 따로 저장
                st.session_state.page = "input"
                st.session_state.session_id = f"{name}_{uuid.uuid4()}"
                st.experimental_rerun()
            else:
                st.warning("이름과 전화번호를 모두 입력해 주세요.")

# ----------------- 고객 정보 입력 화면 -------------------
if st.session_state.page == "input":
    
    # 사이드바 호출
    render_sidebar()
            
    st.markdown(
        "<h4 style='margin-bottom: 20px;'>👤 청약 철회/해지 상황을 입력해 주세요</h4>",
        unsafe_allow_html=True
    )

    # 1️⃣ 고객 이름 입력
    name = st.text_input("고객 이름", placeholder="예: 홍길동", value=st.session_state.get('customer_name_input', ''))

    # 2️⃣ 해지 요청 내용 입력
    situation = st.text_area(
        label="청약 철회 또는 해지 요청 내용",
        placeholder="고객의 구체적인 철회/해지 사유, 요청 배경, 대화 내용을 상세히 입력해 주세요.\n(예: 보험료 부담으로 해지를 원하며, 대안 제시에 일부 관심을 보임)",
        value=st.session_state.get('customer_situation_input', '')
    )

    # 3️⃣ 해지 강도 선택 (라디오 버튼 방식)
    cancel_strength = st.radio(
        "고객의 해지 강도",
        ["하 (설득 여지 있음)", "중 (고민 중)", "상 (매우 완고)"],
        index=["하 (설득 여지 있음)", "중 (고민 중)", "상 (매우 완고)"].index(
            st.session_state.get('cancel_strength_input', "중 (고민 중)")
        ) if 'cancel_strength_input' in st.session_state else 1,
        horizontal=True
    )
    
    # 4 강조 포인트(topping)
    selected_points = st.multiselect(
        "📌 강조할 포인트(선택한 내용이 스크립트에 반영됩니다)",
        options = [
            "굿리치의 신뢰도와 브랜드 공신력 강조",
            "타사 설계와의 비교 설명",
            "가입 당시 상황 다시 리마인드",
            "전담컨설턴트 관리시스템 강조",
            "가족보험관리 서비스 강조"
        ],
        default=[],
    )
    st.caption("")

    # 버튼
    col1, col2 = st.columns([1, 1])
    
    with col1 :
        if st.button("🎲 랜덤 청철 상황 생성하기", use_container_width=True):
            with st.spinner("랜덤 청철 상황 생성 중입니다..."):
                random_info = get_random_cancel_info()
                st.session_state['customer_name_input'] = random_info['name']
                st.session_state['customer_situation_input'] = random_info['situation']
                # 해지 강도는 세션에 "하 (설득 여지 있음)" 같은 형식으로 저장
                strength_map = {
                    "하": "하 (설득 여지 있음)",
                    "중": "중 (고민 중)",
                    "상": "상 (매우 완고)"
                }
                st.session_state['cancel_strength_input'] = strength_map.get(random_info['cancel_strength'], "중 (고민 중)")
                
            st.experimental_rerun()
                
    with col2:
        if st.button("🚀 방어 스크립트 생성하기", use_container_width=True):
            if name and situation:
                # 1️⃣ 세션 저장
                st.session_state['customer_name'] = name
                st.session_state['cancel_strength'] = cancel_strength
                st.session_state['customer_situation'] = situation
                st.session_state['selected_points'] = selected_points

                # 2️⃣ 세션 초기화
                st.session_state.kakao_text = ""
                st.session_state['current_file'] = ""

                # 3️⃣ 방어 스크립트 생성
                with st.spinner("청약 철회/해지 방어 스크립트를 생성 중입니다..."):
                    ai_response = get_script_response(name, situation, cancel_strength)
                    script_text = "".join(ai_response)

                    # 4️⃣ 생성된 스크립트를 세션에 저장
                    st.session_state['script_context'] = script_text
                    st.session_state.message_list = []
                    st.session_state.message_list.append({"role": "ai", "content": script_text})

                # 5️⃣ 페이지 전환: 챗봇 화면
                st.session_state.page = "chatbot"
                st.experimental_rerun()
            else:
                st.warning("고객 이름과 해지 요청 내용을 모두 입력해 주세요.")

# ----------------- 챗봇 화면 -------------------
elif st.session_state.page == "chatbot":
        
    # 사이드바 호출
    render_sidebar()
    
    # 고객정보 호출
    render_customer_info()
        
    user_avatar = URLS["user_avatar"]
    ai_avatar = URLS["ai_avatar"]
        
    messages = st.session_state.get("message_list", [])

    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict) and "role" in message and "content" in message:
                role = message["role"]
                content = message["content"]
                avatar = user_avatar if role == "user" else ai_avatar
                display_message(role, content, avatar)
            else:
                st.warning("⚠️ 불러온 메시지 형식이 잘못되었습니다.")
    else:
        st.error("❌ 메시지 리스트가 손상되었습니다. 다시 불러와 주세요.")

    if user_question := st.chat_input("청철 상담 관련 질문을 자유롭게 입력해 주세요."):
        st.session_state.message_list.append({"role": "user", "content": user_question})
        display_message("user", user_question, user_avatar)

        with st.spinner("답변을 준비 중입니다..."):
            ai_response = get_chatbot_response(user_question, st.session_state['script_context'])
            formatted_response = format_markdown("".join(ai_response))
            st.session_state.message_list.append({"role": "ai", "content": formatted_response})
            display_message("ai", formatted_response, ai_avatar)

    # 👉 버튼 영역: 두 개의 버튼을 나란히 배치
    col1, col2 = st.columns([1, 1])
    
    with col1:                
        if st.button("💬 카카오톡 발송용 문자 생성하기", use_container_width=True):
            if not st.session_state.get('script_context'):
                st.warning("⚠️ 상담 스크립트가 없습니다. 먼저 스크립트를 생성해 주세요.")
            else:
                with st.spinner("카카오톡 문자를 생성 중입니다..."):
                    kakao_message = get_kakao_response(
                        script_context = st.session_state['script_context'],
                        message_list = st.session_state['message_list']
                    )
                    st.session_state['kakao_text'] = "".join(kakao_message)
                    
            # ✅ 안내 문구 출력
            st.info("✅ 카카오톡 문자가 생성되었습니다! 계속해서 추가 질문을 이어가실 수 있습니다.")
                            
    with col2:
        if st.button("💾 대화 저장하기", use_container_width=True):
            user_path = f"/data/{CHATBOT_TYPE}/history/{st.session_state['user_folder']}"
            if not os.path.exists(user_path):
                os.makedirs(user_path)
            if st.session_state.message_list:
                # 1️⃣ 고객 이름 확보
                customer_name = st.session_state.get('customer_name', '고객명미입력')

                # 2️⃣ 기존 파일명 여부 확인
                if st.session_state.get('current_file'):
                    # 기존 파일명에서 고객 이름 유지, 시간만 갱신
                    KST = timezone(timedelta(hours=9))
                    new_filename = f"{customer_name}_{datetime.now(KST).strftime('%y%m%d-%H%M%S')}.json"
                    
                    # 기존 파일 삭제 (덮어쓰기 효과)
                    old_file = f"{user_path}/{st.session_state['current_file']}"
                    if os.path.exists(old_file):
                        os.remove(old_file)
                else:
                    # 새로운 저장이라면
                    KST = timezone(timedelta(hours=9))
                    new_filename = f"{customer_name}_{datetime.now(KST).strftime('%y%m%d-%H%M%S')}.json"

                # 3️⃣ 데이터 저장
                data_to_save = {
                    "customer_name": customer_name,
                    "cancel_strength": st.session_state.get('cancel_strength', ''),
                    "customer_situation": st.session_state.get('customer_situation', ''),
                    "script_context": st.session_state.get('script_context', ''),
                    "message_list": st.session_state.message_list
                }

                with open(f"{user_path}/{new_filename}", "w", encoding="utf-8") as f:
                    json.dump(data_to_save, f, ensure_ascii=False, indent=4)

                # 4️⃣ 파일명 업데이트
                st.session_state['current_file'] = new_filename

                st.success(f"대화가 저장되었습니다! ({new_filename})")
            else:
                st.warning("저장할 대화가 없습니다.")
    
    # 👉 생성된 카카오톡 문자 출력 (있을 때만 표시)
    if st.session_state.get('kakao_text'):
        st.markdown("### 📩 카카오톡 발송용 문자")
        st.text_area("아래 내용을 수정 또는 복사해 사용하세요.", value=st.session_state['kakao_text'], height=400)
        
# 이미지 URL
bottom_image_url = URLS["bottom_image"]

# 최하단에 이미지 출력
st.caption("")

st.markdown(
    f"""
    <div style="text-align:center; margin-bottom:20px;">
        <img src="{bottom_image_url}" alt="Top Banner" style="width:100%; max-width:1000px;">
    </div>
    """,
    unsafe_allow_html=True
)
