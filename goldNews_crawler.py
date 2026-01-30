import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

def fetch_gold_news_v2(pages=3):
    # 新聞網址
    base_url = "https://www.investing.com/commodities/gold-news"
    
    # 強化 Headers，模擬真實瀏覽器行為
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/"
    }
    
    news_list = []
    
    for page in range(1, pages + 1):
        # 第一頁跟後續頁面的網址格式通常不同
        url = base_url if page == 1 else f"{base_url}/{page}"
        print(f"正在爬取: {url}")
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code != 200:
                print(f"無法存取網頁，狀態碼: {response.status_code}")
                # 如果被擋，可以嘗試印出 response.text 的前 100 字來診斷
                break
                
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Investing.com 的新聞通常放在 'article' 標籤中
            # 注意：如果 class 名稱變動，這裡需要微調
            articles = soup.select('article[data-test="article-item"]')
            
            for article in articles:
                title_node = article.select_one('[data-test="article-title-link"]')
                time_node = article.select_one('[data-test="article-publish-date"]')
                
                if title_node:
                    title = title_node.get_text(strip=True)
                    link = "https://www.investing.com" + title_node['href']
                    date = time_node.get_text(strip=True) if time_node else "Unknown"
                    
                    news_list.append({
                        "Date": date,
                        "Title": title,
                        "Link": link
                    })
            
            print(f"第 {page} 頁爬取成功，抓到 {len(articles)} 則新聞")
            time.sleep(3) # 稍微延長延遲，降低被封鎖風險
            
        except Exception as e:
            print(f"發生錯誤: {e}")
            break
            
    return pd.DataFrame(news_list)

# 執行
df = fetch_gold_news_v2(pages=2)
if not df.empty:
    print("\n--- 抓取結果預覽 ---")
    print(df.head())
    # 存成 CSV 方便後續分析
    df.to_csv("gold_news_2026.csv", index=False, encoding="utf-8-sig")
else:
    print("DataFrame 依舊為空，請檢查 CSS Selector 是否正確。")