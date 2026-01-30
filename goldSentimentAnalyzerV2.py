import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import time

class GoldSentimentAnalyzerV2:
    def __init__(self):
        print("正在初始化 FinBERT 模型...")
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        self.headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}

    # 1. 直接從 yfinance 獲取新聞列表
    def get_news_list(self):
        print("正在從 Yahoo Finance 獲取黃金新聞列表...")
        gold = yf.Ticker("GC=F")
        raw_news = gold.news
        
        if not raw_news:
            print("警告：未抓取到任何新聞")
            return pd.DataFrame()

        processed_news = []
        
        for item in raw_news:
            # 取得 nested 的 content 部分 (yfinance 新格式)
            content = item.get('content', {})
            
            # 如果 item 本身就是攤平的，則直接取值；否則從 content 取值
            title = content.get('title', item.get('title', 'No Title'))
            summary = content.get('summary', item.get('summary', ''))
            
            # 取得正確的連結 (通常在 canonicalUrl 裡面)
            link_info = content.get('canonicalUrl', {})
            link = link_info.get('url', item.get('link', ''))
            
            # 取得時間
            pub_date = content.get('pubDate', item.get('providerPublishTime', ''))
            
            processed_news.append({
                "Title": title,
                "Summary": summary,
                "Link": link,
                "Date": pub_date
            })
        
        df = pd.DataFrame(processed_news)
        
        # 轉換日期格式
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            
        return df

    # 2. 抓取 Yahoo 新聞內文
    def fetch_content(self, url):
        if not url or "video" in url: return "Video Content (Skipped)"
        try:
            time.sleep(1.5) # 稍微增加延遲，避免被 Yahoo 偵測
            res = requests.get(url, headers=self.headers, timeout=10)
            if res.status_code != 200: return f"HTTP {res.status_code}"
            
            soup = BeautifulSoup(res.text, 'html.parser')
            
            # --- 強化版：嘗試多種可能的內文容器 ---
            content_selectors = [
                '.caas-body',           # Yahoo 原生
                '.article-body',        # Reuters 轉載
                '.body-copy',           # 一般新聞
                'article',              # 通用標籤
                '.main-content'         # 通用標籤
            ]
            
            body = None
            for selector in content_selectors:
                body = soup.select_one(selector)
                if body: break
            
            if body:
                # 抓取 p 標籤，並過濾掉太短的句子（通常是廣告或選單）
                paragraphs = body.find_all('p')
                text = " ".join([p.get_text() for p in paragraphs if len(p.get_text()) > 30])
                return text if len(text) > 50 else "Content too short"
            
            return "Selector Failed"
        except Exception as e:
            return f"Error: {str(e)}"

    # 3. FinBERT 評分
    def get_sentiment(self, text):
        if not text or len(text) < 10 or "Error" in text: return 0.0
        inputs = self.tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).numpy()[0]
        # FinBERT: [Positive, Negative, Neutral]
        return probs[0] - probs[1]

    def run(self):
        df = self.get_news_list()
        print(f"正在抓取並分析 {len(df)} 則新聞...")
        
        # 1. 抓取內文
        df['Full_Content'] = df['Link'].apply(self.fetch_content)
        
        # 2. 核心邏輯：如果 Full_Content 失敗，就合併 Summary 進去
        # 這能確保模型分析的是最完整的文字
        def merge_text(row):
            if len(row['Full_Content']) < 100:
                return f"{row['Title']}. {row['Summary']}"
            return row['Full_Content']
        
        df['Final_Text_For_AI'] = df.apply(merge_text, axis=1)
        
        # 3. 執行 FinBERT
        print("正在透過 FinBERT 計算情緒分數...")
        df['Sentiment_Score'] = df['Final_Text_For_AI'].apply(self.get_sentiment)
        
        return df

if __name__ == "__main__":
    analyzer = GoldSentimentAnalyzerV2()
    final_df = analyzer.run()
    
    # 儲存結果
    final_df.to_csv("Gold_YahooFinance_SentimentV2.csv", index=False, encoding="utf-8-sig")
    print("\n--- 分析完成 ---")
    print(final_df[['Title', 'Date', 'Title_Score', 'Content_Score']].head())