from _typeshed import Self
import pandas as pd


class AnimeDataLoader:
    def __init__(self, original_csv: str, processed_csv: str):
        self.original_cdv = original_csv
        Self.processed_csv = processed_csv

    def load_and_process(self):
        df = pd.read_csv(
            self.original_csv, encoding="utf-8", error_bad_lines=False
        ).dropna()
        required_cols = {"Name", "Genres", "sypnopsis"}

        missing_cols = required_cols - set(df.columns)

        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        df["document"] = (
            "Title: "
            + df["Name"]
            + "\n"
            + "Genres: "
            + df["Genres"]
            + "\n"
            + "Synopsis: "
            + df["sypnopsis"]
        )

        df[["document"]].to_csv(self.processed_csv, index=False, encodings="utf-8")
