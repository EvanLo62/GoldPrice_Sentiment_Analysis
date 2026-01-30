import pandas as pd
import time
import torch
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

class GoldSentimentAnalyzer:
    def __init__(self):
        print("正在初始化 FinBERT 模型...")
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        
        # 初始化 Selenium (只啟動一次)
        print("正在啟動瀏覽器...")
        chrome_options = Options()
        # chrome_options.add_argument("--headless") # 無頭模式
        chrome_options.add_argument("--disable-blink-features=AutomationControlled") # 隱藏自動化特徵
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        # Selenium 4.10+ 會自動管理驅動程序，直接指定 service 即可
        try:
            self.driver = webdriver.Chrome(options=chrome_options)
        except Exception as e:
            print(f"啟動失敗，嘗試手動指定 Service: {e}")
            self.driver = webdriver.Chrome(service=Service(), options=chrome_options)

    # 1. 抓取列表
    def fetch_news_list(self, pages=1):
        import requests
        base_url = "https://www.investing.com/commodities/gold-news"
        news_data = []
        headers = {"User-Agent": "Mozilla/5.0"}
        
        for page in range(1, pages + 1):
            url = base_url if page == 1 else f"{base_url}/{page}"
            res = requests.get(url, headers=headers)
            soup = BeautifulSoup(res.text, 'html.parser')
            articles = soup.select('article[data-test="article-item"]')
            for article in articles:
                node = article.select_one('[data-test="article-title-link"]')
                if node:
                    href = node['href']
                    full_link = href if href.startswith('http') else "https://www.investing.com" + href
                    news_data.append({"Title": node.get_text(strip=True), "Link": full_link})
        return pd.DataFrame(news_data)

    # 2. 核心修正：使用 Selenium 抓內文
    def fetch_content_with_selenium(self, url):
        print(f"正在抓取內文: {url[:50]}...")
        try:
            self.driver.get(url)
            # 等待 10 秒直到 .WYSIWYG 元素出現
            wait = WebDriverWait(self.driver, 10)
            element = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".WYSIWYG, .articlePage")))
            
            # 抓取所有段落
            paragraphs = self.driver.find_elements(By.CSS_SELECTOR, ".WYSIWYG p, .articlePage p")
            content = " ".join([p.text for p in paragraphs if len(p.text) > 20])
            return content if len(content) > 50 else "Content too short"
        except Exception as e:
            return f"Error: {str(e)}"

    # 3. FinBERT 評分
    def get_sentiment(self, text):
        if not text or "Error" in text or len(text) < 10: return 0.0
        inputs = self.tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).numpy()[0]
        return probs[0] - probs[1] # Positive - Negative

    def run(self, page_limit=1):
        df = self.fetch_news_list(page_limit)
        # 逐筆抓取
        df['Content'] = df['Link'].apply(self.fetch_content_with_selenium)
        print("計算情緒分數中...")
        df['Title_Score'] = df['Title'].apply(self.get_sentiment)
        df['Content_Score'] = df['Content'].apply(self.get_sentiment)
        
        # 關閉瀏覽器
        self.driver.quit()
        return df

if __name__ == "__main__":
    analyzer = GoldSentimentAnalyzer()
    final_df = analyzer.run(page_limit=1)
    final_df.to_csv("gold_sentiment_final.csv", index=False, encoding="utf-8-sig")
    print("完成！請檢查 gold_sentiment_final.csv")
