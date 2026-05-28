import pandas as pd
import re

# 元CSV
INPUT_CSV = r"/Users/masudamasahiro/Desktop/高度プログラミングB課題/SBJ-広島DB.csv"

# 出力CSV
OUTPUT_CSV = r"/Users/masudamasahiro/Desktop/高度プログラミングB課題/SBJ-広島DB_clean.csv"

def clean_text(text):

    text = str(text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(
        r"[^\w\sぁ-んァ-ヶ一-龥]",
        " ",
        text
    )
    text = re.sub(r"\s+", " ", text)

    return text.strip()

def main():

    df = pd.read_csv(INPUT_CSV)

    print("読み込み完了")
    print("元データ数:", len(df))

    df = df[
        [
            "Name",
            "Rating",
            "Reviewer",
            "Review_time",
            "Review",
            "Likes"
        ]
    ]

    df.columns = [
        "store_name",
        "rating",
        "reviewer_name",
        "review_time",
        "review_text",
        "likes"
    ]
    df = df.dropna(subset=["store_name"])

    df = df[df["store_name"].astype(str).str.strip() != ""]

    df["rating"] = df["rating"].astype(str)

    df["rating"] = df["rating"].str.extract(r"(\d)")

    df["rating"] = pd.to_numeric(
        df["rating"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["review_text"]
    )

    df["review_text"] = df["review_text"].apply(
        clean_text
    )

    df = df[
        df["review_text"].str.strip() != ""
    ]

    before = len(df)

    df = df.drop_duplicates(
        subset=[
            "reviewer_name",
            "review_text"
        ]
    )

    df = df.dropna(
        subset=["store_name"]
    )

    df["review_text"] = df["review_text"].apply(
        clean_text
    )

    df = df[
        df["review_text"].str.strip() != ""
    ]

    df = df[
        df["store_name"].astype(str).str.strip() != ""
    ]

    before = len(df)

    df = df.drop_duplicates(
        subset=[
            "reviewer_name",
            "review_text"
        ]
    )

    after = len(df)

    print("重複削除数:", before - after)

    df["likes"] = df["likes"].fillna(0)

    df.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    print("保存完了")
    print("保存先:", OUTPUT_CSV)

    print("\n整形後データ数:", len(df))

    print("\n先頭5件")
    print(df.head())

if __name__ == "__main__":

    main()