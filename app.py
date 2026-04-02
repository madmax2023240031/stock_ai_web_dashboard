import streamlit as st
import plotly.graph_objects as go
from datetime import timedelta

# ✨ 우리가 분리한 모듈(직원들)에서 기능 불러오기
from data_loader import get_ticker_from_name, load_data, fetch_and_analyze_news, get_fundamentals
from model_engine import run_ml_models, generate_llm_report

st.set_page_config(page_title="AI 주식 분석 대시보드", layout="wide", initial_sidebar_state="expanded")

if 'page' not in st.session_state: st.session_state.page = 'HOME'
if 'current_ticker' not in st.session_state: st.session_state.current_ticker = 'NVDA'
if 'current_name' not in st.session_state: st.session_state.current_name = '엔비디아'

def go_to_detail(ticker, name):
    st.session_state.current_ticker = ticker
    st.session_state.current_name = name
    st.session_state.page = 'DETAIL'

def go_home():
    st.session_state.page = 'HOME'

# --- 1. 사이드바 ---
st.sidebar.header("🔍 빠른 종목 검색")
with st.sidebar.form(key='search_form'):
    search_input = st.text_input("종목명 또는 코드 입력 (예: 카카오, AAPL)", help="한국/미국 주식 모두 지원합니다!")
    submit_btn = st.form_submit_button("🚀 AI 분석 시작", use_container_width=True)

if submit_btn and search_input:
    found_ticker = get_ticker_from_name(search_input)
    go_to_detail(found_ticker, search_input)
    st.rerun()

st.sidebar.markdown("---")
ai_choice = st.sidebar.radio("🧠 AI 예측 알고리즘 선택", ("XGBoost", "인공신경망 딥러닝", "앙상블 (추천)"))

# --- 2. 홈 화면 ---
if st.session_state.page == 'HOME':
    st.title("🌐 글로벌 증시 실시간 트렌드")
    st.write("관심 종목을 클릭해 **차트(정형) + 뉴스(비정형) 하이브리드 AI 분석**을 받아보세요.")
    st.markdown("---")

    home_stocks = {"🇺🇸 엔비디아": "NVDA", "🇺🇸 애플": "AAPL", "🇺🇸 테슬라": "TSLA", "🇰🇷 삼성전자": "005930.KS", "🇰🇷 현대차": "005380.KS", "🪙 비트코인": "BTC-USD"}
    cols = st.columns(3)
    
    for i, (name, tk) in enumerate(home_stocks.items()):
        with cols[i % 3]:
            with st.container(border=True):
                st.markdown(f"#### {name}")
                df_mini = load_data(tk)
                if not df_mini.empty and len(df_mini) >= 2:
                    curr_p = df_mini['Close'].iloc[-1]
                    prev_p = df_mini['Close'].iloc[-2]
                    pct_change = ((curr_p - prev_p) / prev_p) * 100
                    is_krw = tk.endswith('.KS') or tk.endswith('.KQ')
                    price_str = f"{curr_p:,.0f}원" if is_krw else f"${curr_p:,.2f}"
                    
                    st.metric(label="현재가", value=price_str, delta=f"{pct_change:+.2f}%")
                    
                    df_30 = df_mini.tail(30)
                    line_color = '#2ca02c' if df_30['Close'].iloc[-1] >= df_30['Close'].iloc[0] else '#d62728'
                    fig_mini = go.Figure(go.Scatter(x=df_30.index, y=df_30['Close'], line=dict(color=line_color, width=3)))
                    fig_mini.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=80, xaxis_visible=False, yaxis_visible=False, template='plotly_dark')
                    st.plotly_chart(fig_mini, use_container_width=True, config={'displayModeBar': False})
                
                if st.button(f"📊 {name.split(' ')[1] if ' ' in name else name} 분석하기", key=f"btn_{tk}", use_container_width=True):
                    go_to_detail(tk, name)
                    st.rerun()

# --- 3. 상세 화면 ---
elif st.session_state.page == 'DETAIL':
    tk = st.session_state.current_ticker
    nm = st.session_state.current_name
    
    if st.button("🔙 메인 홈으로 돌아가기"):
        go_home()
        st.rerun()
        
    st.title(f"🚀 {nm} ({tk}) 통합 AI 분석")
    df = load_data(tk)

    if df.empty or len(df) < 5:
        st.error("데이터를 불러오지 못했거나 데이터가 너무 적습니다.")
    else:
        curr_p = df['Close'].iloc[-1]
        prev_p = df['Close'].iloc[-2]
        diff = curr_p - prev_p
        pct_change = (diff / prev_p) * 100
        is_krw = tk.endswith('.KS') or tk.endswith('.KQ')
        
        with st.container():
            st.metric(label="현재 주가 (최신 장마감 기준)", value=f"{curr_p:,.0f} 원" if is_krw else f"$ {curr_p:,.2f}", delta=f"{diff:+,.0f} 원 ({pct_change:+.2f}%)" if is_krw else f"{diff:+.2f} 달러 ({pct_change:+.2f}%)")
        
        with st.spinner('📰 최신 뉴스와 펀더멘털 데이터를 분석 중입니다...'):
            news_score, news_data = fetch_and_analyze_news(f"{nm} 주가")
            fundamentals = get_fundamentals(tk) 

        st.markdown("---")
        st.markdown("### 🏢 기업 기초 체력 (Fundamentals)")
        f_cols = st.columns(4)
        f_cols[0].metric(label="💰 시가총액", value=fundamentals['시가총액'])
        f_cols[1].metric(label="📈 PER", value=fundamentals['PER'])
        f_cols[2].metric(label="📊 PBR", value=fundamentals['PBR'])
        f_cols[3].metric(label="💸 배당수익률", value=fundamentals['배당수익률'])
        st.markdown("---")

        target_date = (df.index.max() + timedelta(days=1)).strftime('%Y년 %m월 %d일')
        st.subheader("📊 실시간 주가 차트")
        fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])])
        fig.update_layout(xaxis_rangeslider_visible=False, template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)

        # ✨ 예측 모델 기능 호출 (단 1줄로 끝남!)
        accuracy, final_up_prob, final_down_prob, backtest_df, base_probs, news_impact = run_ml_models(df, ai_choice, news_score)

        st.markdown("---")
        st.markdown(f"### 🚀 내일 ({target_date}) 최종 주가 예측")
        if final_up_prob >= 50: st.success(f"🔥🔥 예측 모델은 {nm} 주가가 내일 **상승(UP)**할 것으로 판단했습니다! (확률: {final_up_prob:.1f}%)")
        else: st.error(f"❄️❄️ 예측 모델은 {nm} 주가가 내일 **하락(DOWN)**할 것으로 판단했습니다! (확률: {final_down_prob:.1f}%)")
            
        with st.spinner('🤖 생성형 AI가 오늘 수집된 퀀트 데이터를 종합하여 리포트를 작성하고 있습니다...'):
            # ✨ LLM 리포트 기능 호출 (단 3줄로 끝남!)
            llm_status, llm_msg = generate_llm_report(nm, tk, fundamentals, final_up_prob, news_data, news_score)
            if llm_status == "success": st.info(llm_msg)
            elif llm_status == "error_no_key": st.warning("🔑 LLM 애널리스트 리포트 기능을 활성화하려면 Streamlit Secrets에 API Key 설정이 필요합니다.")
            else: st.warning(f"⚠️ 에러의 진짜 원인: {llm_msg}")
        
        st.markdown("<br>", unsafe_allow_html=True) 
        colA, colB = st.columns(2)
        
        with colA:
            st.markdown(f"#### 💡 트레이딩 시그널")
            st.metric(label="🎯 모델 과거 테스트 정확도", value=f"{accuracy:.2f}%")
            if final_up_prob >= 70: st.success(f"📈 **[적극 매수]** 강력한 상승 신호")
            elif final_up_prob >= 55: st.info(f"🔼 **[매수 / 분할]** 완만한 상승 신호")
            elif final_down_prob >= 70: st.error(f"📉 **[적극 매도]** 강력한 하락 신호")
            elif final_down_prob >= 55: st.warning(f"🔽 **[매도 / 비중 축소]** 완만한 하락 신호")
            else: st.write(f"⏸️ **[관망 (Hold)]** 뚜렷한 추세 없음")
            st.caption(f"*(차트 확률 {base_probs[1]*100:.1f}% + 최신 뉴스 분석 보정치 {news_impact:+.1f}%p 결합)*")

        with colB:
            st.markdown("#### 📰 실시간 뉴스 감성 분석")
            if news_data:
                sentiment_text = "🔥 호재 우세" if news_score > 0.2 else "❄️ 악재 우세" if news_score < -0.2 else "➖ 사실 위주"
                st.write(f"**현재 시장 분위기:** {sentiment_text}")
                for item in news_data: st.markdown(f"{item['badge']} [{item['title']}]({item['link']})")
            else: st.write("최신 뉴스 데이터를 찾을 수 없습니다.")

        st.markdown("---")
        st.markdown("### 📈 모델 신뢰도 검증 (Backtesting Simulation)")
        
        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(x=backtest_df.index, y=backtest_df['Buy_Hold'], mode='lines', name='단순 보유 (Buy & Hold)', line=dict(color='gray', dash='dot')))
        fig_bt.add_trace(go.Scatter(x=backtest_df.index, y=backtest_df['AI_Strategy'], mode='lines', name='AI 시그널 트레이딩', line=dict(color='#00b4d8', width=3)))
        
        final_bh = backtest_df['Buy_Hold'].iloc[-1] - 100
        final_ai = backtest_df['AI_Strategy'].iloc[-1] - 100
        
        fig_bt.update_layout(title=f"가상 투자 수익률 비교 (단순 보유: {final_bh:+.1f}% vs AI 전략: {final_ai:+.1f}%)", yaxis_title="자산 가치 (초기=100)", height=350, template='plotly_dark', hovermode="x unified", margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_bt, use_container_width=True)
        st.markdown("---")
        st.warning("⚠️ **투자 유의사항**: 투자의 책임은 본인에게 있습니다.")