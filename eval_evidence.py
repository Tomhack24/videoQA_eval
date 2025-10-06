import pandas as pd

from utils import culcurate_recall_precision

PREDICT_ID = "QA_DATASET_INFERENCE_32B"
QA_DATASET_GEMINI_PATH = "GROUND_TRUTH/QA_DATASET_GEMINI.jsonl"
PREDICT_QWEN_VL_CSV = f"TABLE/{PREDICT_ID}.csv"

PREDICT_QWEN_VL_PATH = f"DEP_INFERENCE/{PREDICT_ID}.jsonl"
def main():
    df_gt = pd.read_json(QA_DATASET_GEMINI_PATH, lines=True)
    df_pr = pd.read_json(PREDICT_QWEN_VL_PATH, lines=True)
    df_pr = df_pr.rename(columns={"Answer": "Predicted Answer", "Evidence": "Predicted Evidence"})

    merged_df = pd.merge(df_gt, df_pr, on="QA_number", how="inner")

    Event_number = merged_df["QA_number"].tolist()
    Ground_truth = merged_df["Evidence"].tolist()
    Predict = merged_df["Predicted Evidence"].tolist()

    recall, precision, f1, exact_match = culcurate_recall_precision(Event_number, Ground_truth, Predict)
    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Exact Match: {exact_match}")

    print(f"PREDICT_ID: {PREDICT_ID}")
    with open(PREDICT_QWEN_VL_CSV, "a") as f:
        f.write(f"{PREDICT_ID},"f"{recall:.4f},{precision:.4f},{f1:.4f},{exact_match}\n")

if __name__ == "__main__":
    main()