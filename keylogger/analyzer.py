import os
import pandas as pd

def analyze_key_events(df):
    down_events = df[df["Event"] == "down"].copy()
    up_events = df[df["Event"] == "up"].copy()

    analyzed_df = pd.DataFrame()
    analyzed_df["User"] = down_events["User"].values
    analyzed_df["Key"] = down_events["Key"].values
    analyzed_df["TimeDown"] = down_events["TimeInMillis"].values
    analyzed_df["TimeUp"] = up_events["TimeInMillis"].values
    analyzed_df["H"] = up_events["TimeInMillis"].values - down_events["TimeInMillis"].values
    analyzed_df["DD"] = down_events["TimeInMillis"].diff().values
    analyzed_df["UU"] = up_events["TimeInMillis"].diff().values
    analyzed_df["UD"] = analyzed_df["UU"].values - analyzed_df["H"].values

    analyzed_df.to_csv("analyzed_results.csv", index=False)

    return analyzed_df

def extract_features(analyzed_df):
    processed_df = pd.DataFrame(columns=["H", "DD", "UD", "key_stroke_average", "back_space_count", "shift_left_favored", "used_caps", "label"])

    username = analyzed_df.iloc[0]["User"]
    start_time = analyzed_df["TimeDown"].iloc[0]
    end_time = analyzed_df["TimeUp"].iloc[-1]
    total_time = end_time - start_time

    processed_df.loc[0, "H"] = analyzed_df["H"].mean()
    processed_df.loc[0, "DD"] = analyzed_df["DD"].mean()
    processed_df.loc[0, "UD"] = analyzed_df["UD"].mean()
    processed_df.loc[0, "key_stroke_average"] = (len(analyzed_df)/total_time) * 500  # keys pressed per 500 ms
    processed_df.loc[0, "back_space_count"] = (analyzed_df["Key"].value_counts().get("BACK_SPACE", 0)/total_time) * 500

    shift_left_count = analyzed_df["Key"].value_counts().get("SHIFT_LEFT", 0)
    shift_right_count = analyzed_df["Key"].value_counts().get("SHIFT_RIGHT", 0)
    processed_df.loc[0, "shift_left_favored"] = 1 if shift_left_count >= shift_right_count else 0

    processed_df.loc[0, "used_caps"] = 1 if "CAPS_LOCK" in analyzed_df["Key"].values else 0
    processed_df.loc[0, "label"] = username

    return processed_df

def save_to_dataset(processed_df, file_path="keystrokeDataset.csv"):
    include_header = not os.path.exists(file_path)
    processed_df.to_csv(file_path, header=include_header, index=False, mode='a')

def main():
    csv_file = input("Enter file name with .csv extension to analyze (example: filename.csv): ")
    df = pd.read_csv(csv_file)

    analyzed_df = analyze_key_events(df)
    processed_df = extract_features(analyzed_df)
    save_to_dataset(processed_df)

if __name__ == "__main__":
    main()