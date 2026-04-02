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
        return df.dropna()
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

@st.cache_data(ttl=3600)
def get_fundamentals(ticker):
    try:
        info = yf.Ticker(ticker).info
        mc = info.get('marketCap')
        pe = info.get('trailingPE')
        pb = info.get('priceToBook')
        div = info.get('dividendYield')
        currency = info.get('currency', 'KRW')

        if mc:
            if mc >= 1e12: mc_str = f"{mc/1e12:.2f}조 {currency}"
            elif mc >= 1e8: mc_str = f"{mc/1e8:.0f}억 {currency}"
            else: mc_str = f"{mc:,} {currency}"
        else: mc_str = "N/A"

        pe_str = f"{pe:.2f}배" if pe else "N/A"
        pb_str = f"{pb:.2f}배" if pb else "N/A"
        div_str = f"{div*100:.2f}%" if div else "N/A"
        
        return {"시가총액": mc_str, "PER": pe_str, "PBR": pb_str, "배당수익률": div_str}
    except:
        return {"시가총액": "N/A", "PER": "N/A", "PBR": "N/A", "배당수익률": "N/A"}

# --- 3. 사이드바 ---
st.sidebar.header("🔍 빠른 종목 검색")

with st.sidebar.form(key='search_form'):
    search_input = st.text_input("종목명 또는 코드 입력 (예: 카카오, AAPL)", help="한국 주식은 한글로, 미국 주식은 영어 티커(예: TSLA)로 검색해도 모두 찾아냅니다!")
    submit_btn = st.form_submit_button("🚀 AI 분석 시작", use_container_width=True)

if submit_btn and search_input:
    found_ticker = get_ticker_from_name(search_input)
    go_to_detail(found_ticker, search_input)
    st.rerun()

st.sidebar.markdown("---")
# ✨ 전문 용어를 쉽게 풀어쓴 모델 선택 툴팁 추가 ✨
ai_choice = st.sidebar.radio(
    "🧠 AI 예측 알고리즘 선택", 
    ("XGBoost", "인공신경망 딥러닝", "앙상블 (추천)"),
    help="• XGBoost: 빠르고 정확하게 데이터를 분류하는 실전형 모델입니다.\n• 딥러닝: 사람의 뇌 구조를 모방해 복잡한 패턴을 찾아냅니다.\n• 앙상블: 여러 AI 모델의 의견을 다수결로 종합해 가장 오류가 적고 안정적인 결론을 도출합니다."
)

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

    if df.empty or len(df) < 5:
        st.error("데이터를 불러오지 못했거나 데이터가 너무 적습니다.")
    else:
        curr_p = df['Close'].iloc[-1]
        prev_p = df['Close'].iloc[-2]
        diff = curr_p - prev_p
        pct_change = (diff / prev_p) * 100
        
        is_krw = tk.endswith('.KS') or tk.endswith('.KQ')
        price_str = f"{curr_p:,.0f} 원" if is_krw else f"$ {curr_p:,.2f}"
        delta_str = f"{diff:+,.0f} 원 ({pct_change:+.2f}%)" if is_krw else f"{diff:+.2f} 달러 ({pct_change:+.2f}%)"
        
        with st.container():
            st.metric(label="현재 주가 (최신 장마감 기준)", value=price_str, delta=delta_str)
        
        with st.spinner('📰 최신 뉴스와 펀더멘털 데이터를 분석 중입니다...'):
            news_score, news_data = fetch_and_analyze_news(f"{nm} 주가")
            fundamentals = get_fundamentals(tk) 

        st.markdown("---")
        st.markdown("### 🏢 기업 기초 체력 (Fundamentals)")
        # 모바일 환경을 고려하여 재무 지표도 깔끔하게 툴팁 적용
        f_cols = st.columns(4)
        f_cols[0].metric(label="💰 시가총액", value=fundamentals['시가총액'], help="회사의 전체 가치이자 몸집의 크기를 나타냅니다.")
        f_cols[1].metric(label="📈 PER (주가수익비율)", value=fundamentals['PER'], help="회사가 1년에 버는 돈에 비해 주가가 몇 배로 평가받는지 보여줍니다. 낮을수록 저평가되어 있을 확률이 높습니다.")
        f_cols[2].metric(label="📊 PBR (주가순자산비율)", value=fundamentals['PBR'], help="회사가 가진 순수 재산 대비 주가 수준입니다. 1보다 낮으면 가진 재산보다도 주가가 낮다는 뜻입니다.")
        f_cols[3].metric(label="💸 배당수익률", value=fundamentals['배당수익률'], help="지금 주식 1주를 사면 1년 동안 받을 수 있는 배당금의 비율입니다.")
        st.markdown("---")

        target_date = (df.index.max() + timedelta(days=1)).strftime('%Y년 %m월 %d일')

        st.subheader("📊 실시간 주가 차트")
        fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])])
        fig.update_layout(xaxis_rangeslider_visible=False, template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)

        df['Tomorrow'] = df['Close'].shift(-1)
        df['Return'] = df['Close'].pct_change()
        df['Vol_Change'] = df['Volume'].pct_change()
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        
        X = df[['Return', 'Vol_Change', 'MA_5']]
        y = (df['Tomorrow'] > df['Close']).astype(int)

        if "XGBoost" in ai_choice: model = xgb.XGBClassifier(n_estimators=100, max_depth=3, random_state=42)
        elif "딥러닝" in ai_choice: model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        else: model = VotingClassifier(estimators=[('xgb', xgb.XGBClassifier(max_depth=3)), ('mlp', MLPClassifier(hidden_layer_sizes=(100, 50))), ('rf', RandomForestClassifier(n_estimators=100))], voting='soft')

        train_size = int(len(X[:-1]) * 0.8)
        X_train, y_train = X.iloc[:train_size], y.iloc[:train_size]
        X_test, y_test = X.iloc[train_size:-1], y.iloc[train_size:-1]
        
        model.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, model.predict(X_test)) * 100
        
        test_dates = df.index[train_size:-1]
        backtest_df = pd.DataFrame(index=test_dates)
        backtest_df['Daily_Return'] = df['Return'].iloc[train_size:-1].values
        backtest_df['Buy_Hold'] = (1 + backtest_df['Daily_Return']).cumprod() * 100
        ai_signals_shifted = np.roll(model.predict(X_test), shift=1)
        ai_signals_shifted[0] = 1 
        backtest_df['AI_Strategy'] = (1 + backtest_df['Daily_Return'] * ai_signals_shifted).cumprod() * 100

        model.fit(X[:-1], y[:-1]) 
        base_probs = model.predict_proba(X.iloc[[-1]])[0]
        news_impact = news_score * 15.0 
        final_up_prob = min(max(base_probs[1] * 100 + news_impact, 0), 100)
        final_down_prob = 100 - final_up_prob

        st.markdown("---")
        st.markdown(f"### 🚀 내일 ({target_date}) 최종 주가 예측")
        if final_up_prob >= 50:
            st.success(f"🔥🔥 AI는 {nm} 주가가 내일 **상승(UP)**할 것으로 예측했습니다! (확률: {final_up_prob:.1f}%)")
        else:
            st.error(f"❄️❄️ AI는 {nm} 주가가 내일 **하락(DOWN)**할 것으로 예측했습니다! (확률: {final_down_prob:.1f}%)")
        
        st.markdown("<br>", unsafe_allow_html=True) 
        colA, colB = st.columns(2)
        
        with colA:
            st.markdown(f"#### 💡 트레이딩 시그널")
            # ✨ 정확도 설명에 친절한 툴팁 추가 ✨
            st.metric(label="🎯 모델 과거 테스트 정확도", value=f"{accuracy:.2f}%", help="최근 20%의 기간 동안 AI가 정답(실제 상승/하락)을 맞춘 비율입니다.")
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
                for item in news_data:
                    st.markdown(f"{item['badge']} [{item['title']}]({item['link']})")
            else:
                st.write("최신 뉴스 데이터를 찾을 수 없습니다.")

        st.markdown("---")
        # ✨ 백테스팅 설명 보강 ✨
        st.markdown("### 📈 모델 신뢰도 검증 (Backtesting Simulation)")
        st.info("💡 **가상 투자 테스트란?** 과거 특정 시점에 100만 원을 투자했다고 가정했을 때, 주식을 그냥 들고 있던 사람(회색 선)과 AI의 매수/매도 지시를 매일 따른 사람(파란색 선)의 최종 자산 차이를 보여주는 검증 그래프입니다.")
        
        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(x=backtest_df.index, y=backtest_df['Buy_Hold'], mode='lines', name='단순 보유 (Buy & Hold)', line=dict(color='gray', dash='dot')))
        fig_bt.add_trace(go.Scatter(x=backtest_df.index, y=backtest_df['AI_Strategy'], mode='lines', name='AI 시그널 트레이딩', line=dict(color='#00b4d8', width=3)))
        
        final_bh = backtest_df['Buy_Hold'].iloc[-1] - 100
        final_ai = backtest_df['AI_Strategy'].iloc[-1] - 100
        
        fig_bt.update_layout(
            title=f"가상 투자 수익률 비교 (단순 보유: {final_bh:+.1f}% vs AI 전략: {final_ai:+.1f}%)",
            yaxis_title="자산 가치 (초기=100)",
            height=350, template='plotly_dark',
            hovermode="x unified",
            margin=dict(l=20, r=20, t=40, b=20) # 스마트폰에서 그래프가 잘리지 않도록 마진 조정
        )
        st.plotly_chart(fig_bt, use_container_width=True)

        st.markdown("---")
        st.warning("⚠️ **투자 유의사항 (Disclaimer)**: 본 하이브리드 모델은 차트와 텍스트 데이터를 기반으로 한 통계적 확률일 뿐이며, 과거의 수익이 미래의 수익을 보장하지 않습니다. 투자의 책임은 본인에게 있습니다.")
