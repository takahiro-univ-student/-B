import pandas as pd
import re
import itertools
import matplotlib.pyplot as plt
import networkx as nx

plt.rcParams["font.family"] = "Hiragino Sans"
plt.rcParams["axes.unicode_minus"] = False

from janome.tokenizer import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

INPUT_CSV = r"/Users/masudamasahiro/Desktop/高度プログラミングB課題/SBJ-広島DB_clean.csv"
tokenizer = Tokenizer()

STOP_WORDS = {
    "する", "ある", "いる", "なる", "れる", "られる", "てる", "くれる", "行く", "思う", "できる", "ない",
    "こと", "もの", "さん", "よう", "ため", "これ", "それ", "ここ", "そこ", "あそこ", "どこ",
    "スタバ", "スターバックス", "店", "店舗", "利用", "の", "なに", "なん",
    "私", "あなた", "彼", "彼女", "人", "人々", "誰", "皆", "皆さん", "方", "方々",
    "時", "時間", "日", "年", "月", "週", "今", "今日", "明日", "昨日", "先週", "来週", "先月", "来月", "去年", "来年",
    "午前", "午後", "夜", "朝", "昼"
}

def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\n", " ", text)
    #除去
    text = re.sub(r"[^\w\sぁ-んァ-ヶ一-龥]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def tokenize_japanese(text):
    words = []
    for token in tokenizer.tokenize(str(text)):
        part = token.part_of_speech.split(",")[0]
        base = token.base_form
        if part in ["名詞", "動詞", "形容詞"]:
            if base not in STOP_WORDS and len(base) > 1:
                words.append(base)
    return words

def create_label(rating):
    rating = float(rating)
    if rating >= 4:
        return "positive"
    elif rating == 3:
        return "neutral"
    else:
        return "negative"

def analyze_tfidf(df, store_name):
    vectorizer = TfidfVectorizer(
        tokenizer=tokenize_japanese,
        token_pattern=None,
        max_df=0.9,
        min_df=2,
        max_features=300
    )
    X = vectorizer.fit_transform(df["review_text"])
    words = vectorizer.get_feature_names_out()
    scores = X.sum(axis=0).A1

    tfidf_df = pd.DataFrame({"word": words, "score": scores}).sort_values(by="score", ascending=False)

    print(f"\n===== {store_name} TF-IDFの上位30単語 =====")
    print(tfidf_df.head(30))
    return X, tfidf_df

def svm_sentiment_analysis(df, X, store_name):
    y = df["sentiment"]
    print(f"\n===== {store_name} SVM分類preチェック =====")
    print(y.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LinearSVC(class_weight="balanced")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\n===== {store_name} SVM分類結果 =====")
    print(classification_report(y_test, y_pred, zero_division=0))

def make_cooccurrence_network(df, store_name, output_file):
    pair_count = {}
    for text in df["review_text"]:
        words = list(set(tokenize_japanese(text)))[:20]
        for w1, w2 in itertools.combinations(words, 2):
            pair = tuple(sorted([w1, w2]))
            pair_count[pair] = pair_count.get(pair, 0) + 1

    top_pairs = sorted(pair_count.items(), key=lambda x: x[1], reverse=True)[:30]

    G = nx.Graph()
    for (w1, w2), count in top_pairs:
        G.add_edge(w1, w2, weight=count)

    plt.figure(figsize=(12, 9))
    pos = nx.spring_layout(G, k=0.7, seed=42)
    widths = [G[u][v]["weight"] * 0.3 for u, v in G.edges()]

    nx.draw_networkx_nodes(G, pos, node_size=900)
    nx.draw_networkx_edges(G, pos, width=widths, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_family="Hiragino Sans", font_size=10)

    plt.title(f"{store_name} 共起ネットワーク")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.show()
    print(f"\n共起ネットワークを保存しました: {output_file}")

def print_sentiment_summary(df, store_name):
    summary = df["sentiment"].value_counts(normalize=True) * 100
    print(f"\n===== {store_name} 感情割合 =====")
    print(summary.round(2))

def analyze_store(df_all, store_name):
    df = df_all[df_all["store_name"] == store_name].copy()

    print(f"\n==============================")
    print(f"分析対象店舗: {store_name}")
    print(f"レビュー数: {len(df)}")
    print(f"==============================")

    df["review_text"] = df["review_text"].apply(clean_text)
    df = df[df["review_text"] != ""]
    df["sentiment"] = df["rating"].apply(create_label)

    print_sentiment_summary(df, store_name)
    X, tfidf_df = analyze_tfidf(df, store_name)
    svm_sentiment_analysis(df, X, store_name)

    output_file = f"cooccurrence_{store_name}.png"
    make_cooccurrence_network(df, store_name, output_file)

    return tfidf_df

def main():
    df_all = pd.read_csv(INPUT_CSV)
    print("\n===== 店舗一覧 =====")
    stores = df_all["store_name"].dropna().unique()
    for i, store in enumerate(stores):
        print(f"{i}: {store}")

    store1 = input("\n比較したい店舗名1を入力してください: ")
    store2 = input("比較したい店舗名2を入力してください: ")

    tfidf1 = analyze_store(df_all, store1)
    tfidf2 = analyze_store(df_all, store2)

    print("\n===== 2店舗の特徴語比較 =====")
    print(f"\n--- {store1} の特徴語 ---")
    print(tfidf1.head(15))
    print(f"\n--- {store2} の特徴語 ---")
    print(tfidf2.head(15))

if __name__ == "__main__":
    main()