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

# --- 1. 웹사이트 기본 설정 ---
st.set_page_config(page_title="나만의 AI 주식 분석", layout="wide")
st.title("🚀 통합 AI 주식 예측 대시보드")
st.write("원하는 주식을 검색하고, 세 가지 AI 모델 중 하나를 선택해 내일의 주가를 예측해 보세요.")

# --- 코스피/코스닥 2500개 종목 자동 사전 로드 ---
@st.cache_data 
def load_korean_stock_dict():
    krx_df = fdr.StockListing('KRX') 
    stock_dict = {}
    for idx, row in krx_df.iterrows():
        name = str(row['Name']).replace(" ", "").upper()
        code = row['Code']
        market = row['Market']
        if market == 'KOSPI':
            stock_dict[name] = f"{code}.KS"
        elif market == 'KOSDAQ':
            stock_dict[name] = f"{code}.KQ"
        else:
            stock_dict[name] = f"{code}.KS"
            
    global_stocks = {
        "애플": "AAPL", "테슬라": "TSLA", "엔비디아": "NVDA", 
        "마이크로소프트": "MSFT", "구글": "GOOGL", "아마존": "AMZN",
        "알파벳": "GOOGL", "메타": "META", "넷플릭스": "NFLX"
    }
    stock_dict.update(global_stocks)
    return stock_dict

krx_dict = load_korean_stock_dict()

def get_ticker_from_name(search_term):
    clean_term = search_term.replace(" ", "").upper()
    if clean_term in krx_dict:
        return krx_dict[clean_term]
    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={search_term}&lang=ko-KR&region=KR"
    headers = {'User-Agent': 'Mozilla/5.0'} 
    try:
        response = requests.get(url, headers=headers)
        data = response.json()
        if 'quotes' in data and len(data['quotes']) > 0:
            return data['quotes'][0]['symbol']
    except:
        pass
    return search_term

# --- 2. 사이드바 설정 ---
st.sidebar.header("⚙️ 분석 설정")

popular_stocks = {
    "🟩 엔비디아 (NVDA)": "NVDA",
    "🟦 삼성전자 (005930.KS)": "005930.KS",
    "🍎 애플 (AAPL)": "AAPL",
    "⚡️ 테슬라 (TSLA)": "TSLA",
    "🏎️ BMW (BMW.DE)": "BMW.DE",
    "🪟 마이크로소프트 (MSFT)": "MSFT",
    "🔍 직접 종목 이름/코드 입력하기...": "CUSTOM"
}

selected_name = st.sidebar.selectbox("🎯 빠른 종목 선택", list(popular_stocks.keys()))

if popular_stocks[selected_name] == "CUSTOM":
    user_input = st.sidebar.text_input("🔍 종목명 또는 코드 (예: 카카오, Amazon, GOOGL)", "애플")
    ticker = get_ticker_from_name(user_input)
    st.sidebar.caption(f"💡 인식된 종목 코드: **{ticker}**")
else:
    ticker = popular_stocks[selected_name]

st.sidebar.markdown("---")
ai_choice = st.sidebar.radio(
    "🧠 AI 모델 선택",
    ("XGBoost (캐글 우승 알고리즘)", "인공신경망 딥러닝 (패턴 분석형)", "앙상블 (집단지성 최적화 🏆)")
)

# --- 3. 데이터 로드 및 날짜 계산 ---
@st.cache_data 
def load_data(ticker):
    try:
        df = yf.download(ticker, start='2023-01-01', progress=False)
        if df.empty: return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except: return pd.DataFrame()

df = load_data(ticker)

if df.empty:
    st.error("데이터를 불러오지 못했습니다. 종목 이름이나 코드를 다시 확인해 주세요!")
else:
    # ✨ 🌟 날짜 정보 추출 🌟 ✨
    start_date = df.index.min().strftime('%Y년 %m월 %d일')
    end_date = df.index.max().strftime('%Y년 %m월 %d일')
    # 마지막 데이터 날짜의 다음날을 예측 타겟 날짜로 설정
    target_date = (df.index.max() + timedelta(days=1)).strftime('%Y년 %m월 %d일')

    # --- 4. 차트 및 분석 기간 표시 ---
    st.subheader(f"📊 {ticker} 실시간 주가 차트")
    st.info(f"📅 **AI 분석 데이터 기간:** {start_date} ~ {end_date} (최신 장마감 데이터 기준)")
    
    fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name=ticker)])
    fig.update_layout(xaxis_rangeslider_visible=False, template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True)

    # --- 5. AI 예측 (날짜 추가) ---
    st.subheader(f"🤖 {ai_choice}의 [{target_date}] 주가 예측 결과")
    
    df['Tomorrow'] = df['Close'].shift(-1)
    df['Return'] = df['Close'].pct_change()
    df['Vol_Change'] = df['Volume'].pct_change()
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    X = df[['Return', 'Vol_Change', 'MA_5']] 
    y = (df['Tomorrow'] > df['Close']).astype(int)

    # --- 6. 모델 학습 및 확률 계산 ---
    if ai_choice == "XGBoost (캐글 우승 알고리즘)":
        model = xgb.XGBClassifier(n_estimators=100, max_depth=3, random_state=42)
    elif ai_choice == "인공신경망 딥러닝 (패턴 분석형)":
        model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    else:
        clf1 = xgb.XGBClassifier(n_estimators=100, max_depth=3, random_state=42)
        clf2 = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        clf3 = RandomForestClassifier(n_estimators=100, random_state=42)
        model = VotingClassifier(estimators=[('xgb', clf1), ('mlp', clf2), ('rf', clf3)], voting='soft')

    X_history, y_history = X[:-1], y[:-1]
    train_size = int(len(X_history) * 0.8)
    X_train, X_test = X_history.iloc[:train_size], X_history.iloc[train_size:]
    y_train, y_test = y_history.iloc[:train_size], y_history.iloc[train_size:]

    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test)) * 100
    model.fit(X_history, y_history) 

    today_data = X.iloc[[-1]]
    probabilities = model.predict_proba(today_data)[0]
    down_prob, up_prob = probabilities[0] * 100, probabilities[1] * 100

    # --- 7. 결과 출력 ---
    st.markdown("---")
    st.metric(label=f"🎯 {ai_choice} 과거 데이터 예측 정확도", value=f"{accuracy:.2f}%")
    st.markdown(f"### 💡 {target_date} AI 트레이딩 시그널")
    
    if up_prob >= 70:
        st.success(f"📈 **[적극 매수 (Strong Buy)]** 상승 확률이 **{up_prob:.1f}%**로 매우 높습니다.")
    elif up_prob >= 55:
        st.info(f"🔼 **[매수 (Buy) / 분할 접근]** 상승 확률이 **{up_prob:.1f}%**로 약간 우세합니다.")
    elif down_prob >= 70:
        st.error(f"📉 **[적극 매도 (Strong Sell)]** 하락 확률이 **{down_prob:.1f}%**로 매우 높습니다.")
    elif down_prob >= 55:
        st.warning(f"🔽 **[매도 (Sell) / 비중 축소]** 하락 확률이 **{down_prob:.1f}%**로 우세합니다.")
    else:
        st.write(f"⏸️ **[관망 (Hold)]** 상승({up_prob:.1f}%)과 하락({down_prob:.1f}%) 확률이 비슷합니다.")

    # --- 8. 예측 근거 설명 ---
    st.markdown("### 🔍 AI 모델의 예측 근거 분석")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 📝 분석 코멘트")
        st.info(f"분석 대상일: **{end_date}** 종가 기준")
        if ai_choice == "XGBoost (캐글 우승 알고리즘)":
            importances = model.feature_importances_
            st.write(f"🌳 **트리 모델 분석:** 과거 지표 흐름을 쪼개어 **{target_date}**의 방향성을 도출했습니다.")
        elif ai_choice == "인공신경망 딥러닝 (패턴 분석형)":
            st.write(f"🧠 **딥러닝 분석:** 비선형 패턴을 계산하여 **{target_date}**의 결론을 도출했습니다.")
        else: 
            st.write(f"🤝 **앙상블 분석:** 3개 모델의 다수결로 **{target_date}**의 최종 결론을 내렸습니다.")

    with col2:
        st.markdown("#### 📊 데이터 시각화")
        recent_df = df.tail(30)
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(x=recent_df.index, y=recent_df['Close'], mode='lines+markers', name='종가'))
        fig_trend.add_trace(go.Scatter(x=recent_df.index, y=recent_df['MA_5'], mode='lines', name='5일 이평선', line=dict(dash='dot')))
        fig_trend.update_layout(title="최근 30일 가격 추세", height=250, template='plotly_dark')
        st.plotly_chart(fig_trend, use_container_width=True)

    # --- 9. 면책 조항 ---
    st.markdown("---")
    st.warning("⚠️ **투자 유의사항 (Disclaimer)**: 본 예측 결과는 참고용이며, 모든 투자의 책임은 투자자 본인에게 있습니다.")
