import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
import google.generativeai as genai
import streamlit as st

def run_ml_models(df, ai_choice, news_score):
    df = df.copy()
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

    return accuracy, final_up_prob, final_down_prob, backtest_df, base_probs, news_impact

def generate_llm_report(nm, tk, fundamentals, final_up_prob, news_data, news_score):
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    except:
        return "error_no_key", None

    try:
        sentiment_text = "호재 우세" if news_score > 0.2 else "악재 우세" if news_score < -0.2 else "중립/사실 위주"
        news_titles = ", ".join([item['title'] for item in news_data]) if news_data else "관련 뉴스 없음"
        
        prompt = f"""
        당신은 여의도의 탑 티어 퀀트 애널리스트입니다. 아래 데이터를 바탕으로 {nm}({tk})에 대한 3줄 요약 브리핑을 작성해주세요.
        - 시가총액: {fundamentals['시가총액']}
        - PER (주가수익비율): {fundamentals['PER']}
        - 퀀트 예측 모델의 내일 상승 확률: {final_up_prob:.1f}%
        - 오늘의 주요 뉴스 헤드라인: {news_titles}
        - 현재 시장 분위기: {sentiment_text}

        조건:
        1. 개인 투자자가 폰으로 읽기 쉽게 친절한 말투로 작성할 것.
        2. 명확하게 3줄로 요약할 것 (각 줄은 불릿 포인트 형태 사용).
        3. 마지막 줄에는 반드시 투자 권유가 아닌 참고용 브리핑이라는 면책 조항을 부드럽게 포함할 것.
        """
        model_gen = genai.GenerativeModel('gemini-2.5-flash')
        response = model_gen.generate_content(prompt)
        return "success", response.text
    except Exception as e:
        return "error_api", str(e)