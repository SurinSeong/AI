import streamlit as st

## 제목 설정 ##
st.title('echo-bot')

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
    
for content in st.session_state.chat_history:
    with st.chat_message(content['role']):
        st.markdown(content['message'])
        
if prompt := st.chat_input('무엇이든 물어보세요.'):
    with st.chat_message('user'):
        st.markdown(prompt)
        st.session_state.chat_history.append({'role':'user', 'message':prompt})
        
    with st.chat_message('ai', avatar="🤖"):
        response = f'{prompt}..'
        st.markdown(response)
        st.session_state.chat_history.append({'role':'ai', 'message':response})
        
print(st.session_state.chat_history)