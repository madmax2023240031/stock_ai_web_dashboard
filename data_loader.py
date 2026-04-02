import streamlit as st
import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
import FinanceDataReader as fdr

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