document.addEventListener('DOMContentLoaded', () => {
    // === 1. 로그인 로직 (login.html 전용) ===
    const loginForm = document.getElementById('login-form');
    if (loginForm) {
        loginForm.addEventListener('submit', async (e) => {
            e.preventDefault();

            const studentId = document.getElementById('student_id').value;
            const password = document.getElementById('password').value;
            const loginButton = document.getElementById('login-button');
            const statusMessage = document.getElementById('status-message');

            loginButton.disabled = true;
            loginButton.textContent = '로그인 중...';
            statusMessage.textContent = '';
            statusMessage.style.color = '#4f46e5';

            try {
                const response = await fetch('/login', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ student_id: studentId, password: password })
                });

                const data = await response.json();

                if (response.ok && data.success) {
                    // 로그인 성공: chat 페이지로 이동
                    statusMessage.textContent = '✅ 로그인 성공! 잠시 후 채팅 페이지로 이동합니다.';
                    window.location.href = data.redirect;
                } else {
                    // 로그인 실패 (401 또는 success: false)
                    const errorDetail = data.detail || data.message || '알 수 없는 오류 발생';
                    statusMessage.textContent = `❌ 로그인 실패: ${errorDetail}`;
                    statusMessage.style.color = '#ef4444';
                }

            } catch (error) {
                statusMessage.textContent = `❌ 서버 통신 오류: ${error.message}`;
                statusMessage.style.color = '#ef4444';
            } finally {
                loginButton.disabled = false;
                loginButton.textContent = '🚀 로그인';
            }
        });
        return; // 로그인 페이지에서는 채팅 로직 실행하지 않음
    }


    // === 2. 채팅 로직 (chat.html 전용) ===
    const chatInput = document.getElementById('chat-input');
    const sendButton = document.getElementById('send-btn');
    const chatWindow = document.getElementById('chat-window');

    // 메시지를 DOM에 추가하는 함수
    const addMessage = (role, content) => {
        const container = document.createElement('div');
        container.className = `${role}-msg-container`;

        const bubble = document.createElement('div');
        bubble.className = `${role}-msg-bubble`;

        // 줄바꿈 처리
        bubble.innerHTML = content.replace(/\n/g, '<br>');

        container.appendChild(bubble);
        chatWindow.appendChild(container);
        chatWindow.scrollTop = chatWindow.scrollHeight; // 스크롤 하단 이동
    };

    // 기존 메시지 로드 (새로고침 시 메시지 history 로드)
    // FastAPI /chat 엔드포인트에서 템플릿을 통해 기존 메시지를 로드하지 않기 때문에,
    // 세션 변경 시 (혹은 페이지 로드 후) AJAX로 기록을 다시 가져오는 함수가 필요합니다.
    const loadChatHistory = async (sessionId) => {
        chatWindow.innerHTML = ''; // 기존 메시지 삭제
        addMessage('ai', '안녕하세요! 무엇을 도와드릴까요?'); // 기본 시작 메시지

        // **********************************************
        // TODO: FastAPI에 세션 ID를 받아 메시지를 반환하는 API 엔드포인트 (/api/messages/{session_id})가 필요합니다.
        // 현재는 DB에 저장된 메시지를 템플릿 로드시 (chat.html) 직접 로드하지 않기 때문에, 
        // 페이지가 로드되거나 세션이 변경될 때 이 API를 호출해야 합니다.
        // 임시로 메시지가 없다고 가정하고 진행합니다.
        // **********************************************
    };

    // 페이지 로드 시 현재 세션 기록 로드 (현재는 TODO 상태)
    loadChatHistory(CURRENT_CHAT_ID);


    // 메시지 전송 로직
    const sendMessage = async () => {
        const prompt = chatInput.value.trim();
        if (!prompt) return;

        // 1. 사용자 메시지 화면에 표시
        addMessage('user', prompt);
        chatInput.value = '';
        sendButton.disabled = true;

        // 2. AI 응답을 기다리는 임시 메시지 표시
        const loadingMsg = document.createElement('div');
        loadingMsg.className = 'ai-msg-container';
        loadingMsg.innerHTML = '<div class="ai-msg-bubble">답변 생성 중...</div>';
        chatWindow.appendChild(loadingMsg);
        chatWindow.scrollTop = chatWindow.scrollHeight;

        try {
            // 3. FastAPI API 호출
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },

                // 🔴 [수정 전] body: JSON.stringify({ student_id: prompt }) 
                // 🟢 [수정 후] 서버가 원하는 'prompt' 키값으로 변경!
                body: JSON.stringify({ prompt: prompt })
            });

            const data = await response.json();

            // 4. 로딩 메시지 제거 후 AI 응답 표시
            chatWindow.removeChild(loadingMsg);

            if (response.ok && data.success) {
                addMessage('ai', data.response);
            } else {
                addMessage('ai', `❌ 오류 발생: ${data.detail || data.message || '응답 실패'}`);
            }

        } catch (error) {
            chatWindow.removeChild(loadingMsg);
            addMessage('ai', `❌ 서버 통신 중 오류가 발생했습니다: ${error.message}`);
        } finally {
            sendButton.disabled = false;
            chatWindow.scrollTop = chatWindow.scrollHeight;
        }
    };

    // 버튼 클릭 및 엔터 키 이벤트
    if (sendButton) {
        sendButton.addEventListener('click', sendMessage);
    }
    if (chatInput) {
        chatInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                sendMessage();
            }
        });
    }

    // === 3. 사이드바 세션 변경 로직 ===
    document.querySelectorAll('.session-item').forEach(item => {
        item.addEventListener('click', async () => {
            const sessionId = item.dataset.sessionId;

            // 1. FastAPI에 현재 세션 ID를 변경 요청
            const response = await fetch('/set_chat_session', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: sessionId })
            });

            if (response.ok) {
                // 2. 페이지를 새로고침하여 새로운 세션 정보 로드
                window.location.reload();
            } else {
                alert('세션 변경에 실패했습니다.');
            }
        });
    });

    // 새 대화 버튼
    document.getElementById('new-chat-btn')?.addEventListener('click', async () => {
        // 빈 세션 ID로 변경 요청 후 새로고침
        const response = await fetch('/set_chat_session', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: '' }) // 빈 문자열을 보내서 main.py에서 새 ID 생성 유도
        });

        if (response.ok) {
            window.location.reload();
        } else {
            alert('새 대화 시작에 실패했습니다.');
        }
    });

});