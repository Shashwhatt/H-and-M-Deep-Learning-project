def summarize_item_features(df):
    print("Item Feature Summary")
    print("--------------------")
    print(f"Total items: {len(df)}")

    for column in df.columns:
        print(f"{column}: {df[column].nunique()} unique values")