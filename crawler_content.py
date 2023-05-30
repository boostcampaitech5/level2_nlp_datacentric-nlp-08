import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm.auto import tqdm

if __name__ == "__main__":
    train_data = pd.read_csv("new_train.csv")
    content = []
    headers = {
        "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/xxx.xx (KHTML, like Gecko) Chrome/xxx.0.0.0 Safari/xxx.xx"
    }
    for i in tqdm(range(len(train_data))):
        url = train_data.iloc[i]["url"]
        url = requests.get(url, headers=headers)
        soup = BeautifulSoup(url.text, "html.parser")
        # print(soup)

        article = soup.find("div", {"class": "go_trans _article_content"})
        article_text = ""
        if article != None:
            article_text = article.text.strip()

        pattern = "@yna.co.kr"
        cut_idx = article_text.find(pattern)
        if cut_idx != -1:
            article_text = article_text[:cut_idx]
            article_text = " ".join(article_text.split("    ")[:-1])
        content.append(str(article_text))
    train_data = train_data[:2]
    train_data["content"] = content
    train_data.to_csv("./train_data_with_content.csv", index=False)
