import pandas as pd
import re

from janome.tokenizer import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report


INPUT_CSV = r"/Users/masudamasahiro/Desktop/高度プログラミングB課題/SBJ-広島DB_clean.csv"

tokenizer = Tokenizer()

STOP_WORDS = {
    "する", "ある", "いる", "なる", "れる", "られる",
    "こと", "もの", "さん", "よう", "ため", "これ", "それ",
    "スタバ", "スターバックス", "店", "店舗", "利用", "の",
    "てる", "くれる", "行く", "思う", "できる", "ない",
    "ここ", "そこ", "あそこ", "どこ", "なに", "なん",
    "私", "あなた", "彼", "彼女", "人", "人々", "誰",
    "皆", "皆さん", "方", "方々", "時", "時間", "日",
    "年", "月", "週", "今", "今日", "明日", "昨日",
    "先週", "来週", "先月", "来月", "去年", "来年",
    "午前", "午後", "夜", "朝", "昼"
}


def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"[^\w\sぁ-んァ-ヶ一-龥]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


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


def svm_for_store(df_store, store_name):
    print("\n==============================")
    print(f"分析対象店舗: {store_name}")
    print(f"レビュー数: {len(df_store)}")
    print("==============================")

    df_store = df_store.copy()

    df_store["review_text"] = df_store["review_text"].apply(clean_text)
    df_store = df_store[df_store["review_text"] != ""]

    if len(df_store) < 10:
        print("レビュー数が少ないためSVMをスキップします。")
        return None

    df_store["sentiment"] = df_store["rating"].apply(create_label)

    print("\n===== 感情ラベル数 =====")
    print(df_store["sentiment"].value_counts())

    if df_store["sentiment"].nunique() < 2:
        print("感情ラベルが1種類しかないためSVMをスキップします。")
        return None

    vectorizer = TfidfVectorizer(
        tokenizer=tokenize_japanese,
        token_pattern=None,
        max_df=0.9,
        min_df=2,
        max_features=300
    )

    try:
        X = vectorizer.fit_transform(df_store["review_text"])
    except ValueError as e:
        print("TF-IDF変換に失敗したためスキップします。")
        print(e)
        return None

    y = df_store["sentiment"]

    min_class_count = y.value_counts().min()

    try:
        if min_class_count >= 2:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42
            )

        model = LinearSVC(class_weight="balanced")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        print("\n===== SVM分類結果 =====")
        print(
            classification_report(
                y_test,
                y_pred,
                zero_division=0
            )
        )

        report = classification_report(
            y_test,
            y_pred,
            zero_division=0,
            output_dict=True
        )

        return {
            "store_name": store_name,
            "review_count": len(df_store),
            "accuracy": report["accuracy"],
            "macro_f1": report["macro avg"]["f1-score"],
            "weighted_f1": report["weighted avg"]["f1-score"],
            "positive_count": int((y == "positive").sum()),
            "neutral_count": int((y == "neutral").sum()),
            "negative_count": int((y == "negative").sum())
        }

    except ValueError as e:
        print("SVM分析をスキップしました。")
        print(e)
        return None


def main():
    df_all = pd.read_csv(INPUT_CSV)

    df_all = df_all.dropna(subset=["store_name", "review_text", "rating"])

    results = []

    stores = df_all["store_name"].dropna().unique()

    for store_name in stores:
        df_store = df_all[df_all["store_name"] == store_name]

        result = svm_for_store(df_store, store_name)

        if result is not None:
            results.append(result)

    if len(results) > 0:
        result_df = pd.DataFrame(results)

        print("\n==============================")
        print("全店舗SVM結果まとめ")
        print("==============================")
        print(result_df)

        output_path = r"/Users/masudamasahiro/Desktop/高度プログラミングB課題/svm_all_store_results.csv"

        result_df.to_csv(
            output_path,
            index=False,
            encoding="utf-8-sig"
        )

        print("\n保存完了:")
        print(output_path)


if __name__ == "__main__":
    main()