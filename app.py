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
import FinanceDataReader as fdr # 👈 한국 주식 전체 목록을 가져올 마법의 도구 추가!

# --- 1. 웹사이트 기본 설정 ---
st.set_page_config(page_title="나만의 AI 주식 분석", layout="wide")
st.title("🚀 통합 AI 주식 예측 대시보드")
st.write("원하는 주식을 검색하고, 세 가지 AI 모델 중 하나를 선택해 내일의 주가를 예측해 보세요.")

# --- 🌟 코스피/코스닥 2500개 종목을 자동으로 사전에 넣는 마법 ---
@st.cache_data # 매번 다운받으면 느려지니까, 한 번만 다운받아서 기억(캐시)해 둡니다.
def load_korean_stock_dict():
    """한국 거래소(KRX)의 모든 종목 코드를 가져와서 야후 파이낸스용 사전으로 만듭니다."""
    krx_df = fdr.StockListing('KRX') # 한국 주식 전체 목록 다운로드!
    
    stock_dict = {}
    for idx, row in krx_df.iterrows():
        # 이름의 띄어쓰기를 없애고 대문자로 통일 (예: '삼성 전자' -> '삼성전자')
        name = str(row['Name']).replace(" ", "").upper()
        code = row['Code']
        market = row['Market']
        
        # 야후 파이낸스는 코스피는 .KS, 코스닥은 .KQ를 뒤에 붙여야 알아듣습니다.
        if market == 'KOSPI':
            stock_dict[name] = f"{code}.KS"
        elif market == 'KOSDAQ':
            stock_dict[name] = f"{code}.KQ"
        else:
            stock_dict[name] = f"{code}.KS" # 기본값
            
    # 기왕 만드는 김에 유명한 해외 주식들도 수동으로 슬쩍 추가해 줍니다.
    global_stocks = {
        "애플": "AAPL", "테슬라": "TSLA", "엔비디아": "NVDA", 
        "마이크로소프트": "MSFT", "구글": "GOOGL", "아마존": "AMZN",
        "알파벳": "GOOGL", "메타": "META", "넷플릭스": "NFLX"
    }
    stock_dict.update(global_stocks)
    
    return stock_dict

# 앱이 켜질 때 만능 사전을 로드합니다.
krx_dict = load_korean_stock_dict()

# --- 🌟 마법의 함수: 완벽해진 번역기 ---
def get_ticker_from_name(search_term):
    clean_term = search_term.replace(" ", "").upper()
    
    # 1. 2500개짜리 만능 사전에 그 이름이 있는지 확인! (있으면 0.1초 만에 코드 반환)
    if clean_term in krx_dict:
        return krx_dict[clean_term]
        
    # 2. 한국 주식도 아니고 유명 해외 주식도 아니면, 그때서야 야후 API에 물어봅니다.
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


# --- 2. 사이드바: 빠른 종목 선택 및 AI 두뇌 설정 ---
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

# --- 🌟 기능 업그레이드: 앙상블 모델 선택지 추가 ---
st.sidebar.markdown("---")
ai_choice = st.sidebar.radio(
    "🧠 AI 모델 선택",
    ("XGBoost (캐글 우승 알고리즘)", "인공신경망 딥러닝 (패턴 분석형)", "앙상블 (집단지성 최적화 🏆)")
)

st.sidebar.markdown("---")
st.sidebar.info("데이터를 불러오고 학습하는 데 몇 초 정도 걸릴 수 있습니다.")

# --- 3. 주식 데이터 불러오기 ---
@st.cache_data 
def load_data(ticker):
    df = yf.download(ticker, start='2023-01-01', end='2026-04-01', progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    return df

df = load_data(ticker)

if df.empty:
    st.error("데이터를 불러오지 못했습니다. 종목 이름이나 코드를 다시 확인해 주세요!")
else:
    # --- 4. 전문가용 캔들스틱 차트 그리기 ---
    st.subheader(f"📊 {ticker} 실시간 주가 차트")
    
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    name=ticker)])
    
    fig.update_layout(xaxis_rangeslider_visible=False, template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True)

    # --- 5. AI 예측을 위한 데이터 전처리 ---
    st.subheader(f"🤖 {ai_choice}의 내일 주가 예측 결과")
    
    df['Tomorrow'] = df['Close'].shift(-1)
    df['Return'] = df['Close'].pct_change()
    df['Vol_Change'] = df['Volume'].pct_change()
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    
    df = df.replace([np.inf, -np.inf], np.nan) 
    df = df.dropna()

    X = df[['Return', 'Vol_Change', 'MA_5']] 
    y = (df['Tomorrow'] > df['Close']).astype(int)

    # --- 6. 선택한 AI 두뇌로 학습 및 예측 ---
    if ai_choice == "XGBoost (캐글 우승 알고리즘)":
        model = xgb.XGBClassifier(n_estimators=100, max_depth=3, random_state=42)
    elif ai_choice == "인공신경망 딥러닝 (패턴 분석형)":
        model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    else:
        # ✨ 궁극의 앙상블 모델: 3개의 AI를 하나로 합칩니다.
        clf1 = xgb.XGBClassifier(n_estimators=100, max_depth=3, random_state=42)
        clf2 = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        clf3 = RandomForestClassifier(n_estimators=100, random_state=42)
        model = VotingClassifier(estimators=[('xgb', clf1), ('mlp', clf2), ('rf', clf3)], voting='hard')

    X_history = X[:-1]
    y_history = y[:-1]

    train_size = int(len(X_history) * 0.8)
    X_train, X_test = X_history.iloc[:train_size], X_history.iloc[train_size:]
    y_train, y_test = y_history.iloc[:train_size], y_history.iloc[train_size:]

    model.fit(X_train, y_train)
    test_predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, test_predictions) * 100

    model.fit(X_history, y_history) 

    today_data = X.iloc[[-1]]
    prediction = model.predict(today_data)[0]

    # --- 7. 최종 결과 및 정확도 출력 ---
    st.markdown("---")
    st.metric(label=f"🎯 {ai_choice} 과거 예측 정확도", value=f"{accuracy:.2f}%")
    
    if prediction == 1:
        st.success(f"🔥🔥 **{ticker}** 주식은 내일 **상승(UP)** 할 것으로 예측되었습니다! 🔥🔥")
    else:
        st.error(f"❄️❄️ **{ticker}** 주식은 내일 **하락(DOWN)** 할 것으로 예측되었습니다! ❄️❄️")

    # --- 8. AI 모델의 예측 근거 분석 (XAI) ---
    st.markdown("### 🔍 AI 모델의 예측 근거 분석")
    
    today_return = today_data['Return'].values[0] * 100
    today_vol = today_data['Vol_Change'].values[0] * 100
    today_ma5 = today_data['MA_5'].values[0]
    current_price = df['Close'].iloc[-1]
    ma_status = "위에서" if current_price > today_ma5 else "아래에서"

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📝 분석 코멘트")
        if ai_choice == "XGBoost (캐글 우승 알고리즘)":
            importances = model.feature_importances_
            feature_names = ['수익률(Return)', '거래량 변화율', '5일 이평선(MA_5)']
            max_idx = np.argmax(importances)
            key_feature = feature_names[max_idx]
            st.info(f"🌳 **트리 모델 분석:** 이번 예측에서 가장 핵심적으로 작용했다고 판단한 지표는 **'{key_feature}'** 입니다.")
            st.write(f"- 오늘 주가는 전일 대비 **{today_return:.2f}%** 변동했습니다.")
            st.write("- 모델은 과거 비슷한 변동성과 지표 흐름이 있었을 때의 주가 패턴을 쪼개어 분석한 뒤 위와 같은 결론을 내렸습니다.")

        elif ai_choice == "인공신경망 딥러닝 (패턴 분석형)":
            st.info(f"🧠 **딥러닝 분석:** 인공신경망 모델은 수많은 가상 뉴런을 거치며 복합적인 비선형 패턴을 계산했습니다.")
            st.write(f"- 현재 주가가 5일 이동평균선 **{ma_status}** 형성되어 있는 추세 패턴을 과거 데이터의 신경망 가중치와 연산하여 결론을 도출했습니다.")
            
        else: # 앙상블 XAI 추가!
            st.info("🤝 **앙상블(집단지성) 분석:** 세 가지 서로 다른 최고급 AI 모델이 독립적으로 데이터를 분석한 뒤, 다수결 투표(Voting)를 통해 최종 결론을 내렸습니다.")
            st.write("- 단일 알고리즘이 가질 수 있는 편향(Bias)과 과적합(Overfitting)의 위험을 상호 보완하여 가장 통계적으로 안정적인 결론을 도출해냅니다.")

    with col2:
        st.markdown("#### 📊 데이터 시각화")
        if ai_choice == "XGBoost (캐글 우승 알고리즘)":
            fig_imp = go.Figure(go.Bar(
                x=importances, y=feature_names, orientation='h', marker_color=['#1f77b4', '#ff7f0e', '#2ca02c']
            ))
            fig_imp.update_layout(title="어떤 힌트가 가장 중요했을까?", xaxis_title="중요도", height=250, template='plotly_dark')
            st.plotly_chart(fig_imp, use_container_width=True)
            
        else:
            recent_df = df.tail(30)
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(x=recent_df.index, y=recent_df['Close'], mode='lines+markers', name='종가', line=dict(color='#00b4d8')))
            fig_trend.add_trace(go.Scatter(x=recent_df.index, y=recent_df['MA_5'], mode='lines', name='5일 이평선', line=dict(dash='dot', color='#ffb703')))
            fig_trend.update_layout(title="최근 30일 가격 및 이평선 추세", yaxis_title="가격", height=250, template='plotly_dark')
            st.plotly_chart(fig_trend, use_container_width=True)

    # --- (이전 코드: 그래프 그리는 부분) ---
    # st.plotly_chart(fig_trend, use_container_width=True)

    # --- 9. 면책 조항 (Disclaimer) 추가 ---
    st.markdown("---")
    st.warning("""
    ⚠️ **투자 유의사항 (Disclaimer)**
    
    본 대시보드의 AI 예측 결과는 과거의 가격과 거래량 데이터를 기반으로 한 통계적·수학적 확률일 뿐이며, **미래의 실제 주가를 절대 보장하지 않습니다.** 주식 시장은 예측 불가능한 수많은 외부 변수(뉴스, 실적 발표, 거시경제 등)의 영향을 받으므로, **모든 투자의 최종 결정과 그에 따른 책임은 전적으로 투자자 본인에게 있습니다.** 본 서비스는 가벼운 참고 및 학습용으로만 활용해 주시기 바랍니다.
    """)