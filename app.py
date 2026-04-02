import streamlit as st
import yfinance as yf
import pandas as pd
import xgboost as xgb
import plotly.graph_objects as go
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import requests 
import numpy as np 
from sklearn.metrics import accuracy_score 
import FinanceDataReader as fdr 
from datetime import timedelta
from bs4 import BeautifulSoup
from transformers import pipeline

# --- 1. 웹사이트 기본 설정 ---
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

# --- 2. AI 모델 및 데이터 로드 함수 ---
@st.cache_resource 
def load_finbert():
    return pipeline("text-classification", model="snunlp/KR-FinBert-SC")

@st.cache_data 
def load_korean_stock_dict():
    krx_df = fdr.StockListing('KRX') 
    stock_dict = {str(row['Name']).replace(" ", "").upper(): f"{row['Code']}.KS" if row['Market'] == 'KOSPI' else f"{row['Code']}.KQ" for idx, row in krx_df.iterrows()}
    stock_dict.update({"애플": "AAPL", "테슬라": "TSLA", "엔비디아": "NVDA", "비트코인": "BTC-USD", "마이크로소프트": "MSFT", "구글": "GOOGL"})
    return stock_dict

krx_dict = load_korean_stock_dict()

def get_ticker_from_name(search_term):
    clean_term = search_term.replace(" ", "").upper()
    if clean_term in krx_dict: return krx_dict[clean_term]
    try:
        data = requests.get(f"https://query2.finance.yahoo.com/v1/finance/search?q={search_term}&lang=ko-KR&region=KR", headers={'User-Agent': 'Mozilla/5.0'}).json()
        if 'quotes' in data and len(data['quotes']) > 0: return data['quotes'][0]['symbol']
    except: pass
    return search_term

@st.cache_data 
def load_data(ticker):
    try:
        df = yf.download(ticker, start='2022-01-01', progress=False)
        if df.empty: return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        return df
    except: return pd.DataFrame()

@st.cache_data(ttl=3600) 
def fetch_and_analyze_news(keyword):
    url = f"https://news.google.com/rss/search?q={keyword}&hl=ko&gl=KR&ceid=KR:ko"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'xml') 
    items = soup.find_all('item')[:5] 
    
    if not items: return 0, []
    
    analyzer = load_finbert()
    score_sum, news_list = 0, []
    
    for item in items:
        headline, link = item.title.get_text(), item.link.get_text()
        res = analyzer(headline)[0]
        
        if res['label'] == 'positive': score, badge = res['score'], "🔥 [호재]"
        elif res['label'] == 'negative': score, badge = -res['score'], "❄️ [악재]"
        else: score, badge = 0, "➖ [중립]"
            
        score_sum += score
        news_list.append({"title": headline, "link": link, "badge": badge})
        
    return score_sum / len(items) if items else 0, news_list

# --- 3. 사이드바 ---
st.sidebar.header("🔍 빠른 종목 검색")
search_input = st.sidebar.text_input("종목명 또는 코드 입력 (예: 카카오, AAPL)")
if st.sidebar.button("🚀 AI 분석 시작", use_container_width=True):
    if search_input:
        found_ticker = get_ticker_from_name(search_input)
        go_to_detail(found_ticker, search_input)
        st.rerun()

st.sidebar.markdown("---")
ai_choice = st.sidebar.radio("🧠 차트 분석 모델", ("XGBoost", "인공신경망 딥러닝", "앙상블 (추천)"))

# ==========================================
# 🏠 화면 A: 홈 화면
# ==========================================
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
                if not df_mini.empty:
                    df_30 = df_mini.tail(30)
                    line_color = '#2ca02c' if df_30['Close'].iloc[-1] >= df_30['Close'].iloc[0] else '#d62728'
                    fig_mini = go.Figure(go.Scatter(x=df_30.index, y=df_30['Close'], line=dict(color=line_color, width=3)))
                    fig_mini.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=100, xaxis_visible=False, yaxis_visible=False, template='plotly_dark')
                    st.plotly_chart(fig_mini, use_container_width=True, config={'displayModeBar': False})
                if st.button(f"📊 {name.split(' ')[1] if ' ' in name else name} 분석하기", key=f"btn_{tk}", use_container_width=True):
                    go_to_detail(tk, name)
                    st.rerun()

# ==========================================
# 🔍 화면 B: 상세 분석 화면 (하이브리드 AI)
# ==========================================
elif st.session_state.page == 'DETAIL':
    tk = st.session_state.current_ticker
    nm = st.session_state.current_name
    
    if st.button("🔙 메인 홈으로 돌아가기"):
        go_home()
        st.rerun()
        
    st.title(f"🚀 {nm} ({tk}) 통합 AI 분석")
    df = load_data(tk)

    if df.empty:
        st.error("데이터를 불러오지 못했습니다.")
    else:
        with st.spinner('📰 최신 뉴스를 읽고 감성을 분석 중입니다...'):
            news_score, news_data = fetch_and_analyze_news(f"{nm} 주가")

        # ✨ 내일 날짜 계산 복구
        target_date = (df.index.max() + timedelta(days=1)).strftime('%Y년 %m월 %d일')

        st.subheader("📊 실시간 주가 차트")
        fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])])
        fig.update_layout(xaxis_rangeslider_visible=False, template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)

        # 1. 모델 학습 데이터 준비
        df['Tomorrow'], df['Return'], df['Vol_Change'], df['MA_5'] = df['Close'].shift(-1), df['Close'].pct_change(), df['Volume'].pct_change(), df['Close'].rolling(window=5).mean()
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        X, y = df[['Return', 'Vol_Change', 'MA_5']], (df['Tomorrow'] > df['Close']).astype(int)

        if "XGBoost" in ai_choice: model = xgb.XGBClassifier(n_estimators=100, max_depth=3, random_state=42)
        elif "딥러닝" in ai_choice: model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        else: model = VotingClassifier(estimators=[('xgb', xgb.XGBClassifier(max_depth=3)), ('mlp', MLPClassifier(hidden_layer_sizes=(100, 50))), ('rf', RandomForestClassifier(n_estimators=100))], voting='soft')

        # 학습 및 테스트 데이터 분리 (최근 20% 기간을 백테스팅에 사용)
        train_size = int(len(X[:-1]) * 0.8)
        X_train, y_train = X.iloc[:train_size], y.iloc[:train_size]
        X_test, y_test = X.iloc[train_size:-1], y.iloc[train_size:-1]
        
        # 모델 학습 및 정확도 측정
        model.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, model.predict(X_test)) * 100
        
        # 백테스팅 계산 로직
        test_dates = df.index[train_size:-1]
        ai_predictions = model.predict(X_test)
        backtest_df = pd.DataFrame(index=test_dates)
        backtest_df['Daily_Return'] = df['Return'].iloc[train_size:-1].values
        backtest_df['Buy_Hold'] = (1 + backtest_df['Daily_Return']).cumprod() * 100
        ai_signals_shifted = np.roll(ai_predictions, shift=1)
        ai_signals_shifted[0] = 1 
        backtest_df['AI_Strategy_Return'] = backtest_df['Daily_Return'] * ai_signals_shifted
        backtest_df['AI_Strategy'] = (1 + backtest_df['AI_Strategy_Return']).cumprod() * 100

        # 최종 예측 산출
        model.fit(X[:-1], y[:-1]) 
        base_probs = model.predict_proba(X.iloc[[-1]])[0]
        news_impact = news_score * 15.0 
        final_up_prob = min(max(base_probs[1] * 100 + news_impact, 0), 100)
        final_down_prob = 100 - final_up_prob

        # --- 출력 1: 예측 시그널 및 뉴스 ---
        st.markdown("---")
        
        # ✨ 내일 주가 예측 직관적 결과 복구
        st.markdown(f"### 🚀 내일 ({target_date}) 최종 주가 예측")
        if final_up_prob >= 50:
            st.success(f"🔥🔥 AI는 {nm} 주가가 내일 **상승(UP)**할 것으로 예측했습니다! (확률: {final_up_prob:.1f}%)")
        else:
            st.error(f"❄️❄️ AI는 {nm} 주가가 내일 **하락(DOWN)**할 것으로 예측했습니다! (확률: {final_down_prob:.1f}%)")
        
        st.markdown("<br>", unsafe_allow_html=True) # 약간의 여백
        
        colA, colB = st.columns(2)
        
        with colA:
            st.markdown(f"#### 💡 트레이딩 시그널")
            st.metric(label="🎯 모델 과거 테스트 정확도", value=f"{accuracy:.2f}%", help="최근 20%의 데이터를 가리고 모델이 맞춘 비율입니다.")
            if final_up_prob >= 70: st.success(f"📈 **[적극 매수]** 강력한 상승 신호")
            elif final_up_prob >= 55: st.info(f"🔼 **[매수 / 분할]** 완만한 상승 신호")
            elif final_down_prob >= 70: st.error(f"📉 **[적극 매도]** 강력한 하락 신호")
            elif final_down_prob >= 55: st.warning(f"🔽 **[매도 / 비중 축소]** 완만한 하락 신호")
            else: st.write(f"⏸️ **[관망 (Hold)]** 뚜렷한 추세 없음")
            st.caption(f"*(차트 확률 {base_probs[1]*100:.1f}% + 뉴스 보정치 {news_impact:+.1f}%p)*")

        with colB:
            st.markdown("#### 📰 실시간 뉴스 감성 분석")
            if news_data:
                sentiment_text = "🔥 호재 우세" if news_score > 0.2 else "❄️ 악재 우세" if news_score < -0.2 else "➖ 사실 위주"
                st.write(f"**현재 시장 분위기:** {sentiment_text}")
                for item in news_data:
                    st.markdown(f"{item['badge']} [{item['title']}]({item['link']})")
            else:
                st.write("최신 뉴스 데이터를 찾을 수 없습니다.")

        # --- 출력 2: 백테스팅 시각화 파트 ---
        st.markdown("---")
        st.markdown("### 📈 모델 신뢰도 검증 (Backtesting Simulation)")
        st.info("💡 **가상 투자 테스트:** 최근 기간 동안 주식을 계속 보유했을 때(회색 선)와, AI의 매수/매도 시그널을 따랐을 때(파란색 선)의 자산 변화 차이입니다. (초기 자산 100 기준)")
        
        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(x=backtest_df.index, y=backtest_df['Buy_Hold'], mode='lines', name='단순 보유 (Buy & Hold)', line=dict(color='gray', dash='dot')))
        fig_bt.add_trace(go.Scatter(x=backtest_df.index, y=backtest_df['AI_Strategy'], mode='lines', name='AI 시그널 트레이딩', line=dict(color='#00b4d8', width=3)))
        
        final_bh = backtest_df['Buy_Hold'].iloc[-1] - 100
        final_ai = backtest_df['AI_Strategy'].iloc[-1] - 100
        
        fig_bt.update_layout(
            title=f"가상 투자 수익률 비교 (단순 보유: {final_bh:+.1f}% vs AI 전략: {final_ai:+.1f}%)",
            yaxis_title="자산 가치 (초기=100)",
            height=350, template='plotly_dark',
            hovermode="x unified"
        )
        st.plotly_chart(fig_bt, use_container_width=True)

        st.markdown("---")
        st.warning("⚠️ **투자 유의사항 (Disclaimer)**: 본 하이브리드 모델은 차트와 텍스트 데이터를 기반으로 한 통계적 확률일 뿐이며, 과거의 수익이 미래의 수익을 보장하지 않습니다. 투자의 책임은 본인에게 있습니다.")
