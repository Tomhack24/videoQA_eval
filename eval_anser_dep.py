import pandas as pd
from utils import calculate_answer_scores 



QA_DATASET_GEMINI_PATH = "GROUND_TRUTH/QA_DATASET_GEMINI.jsonl"
PREDICT_QWEN_VL_PATH_LIST = ["DEP_INFERENCE/QA_DATASET_INFERENCE_3B.jsonl",
                            "DEP_INFERENCE/QA_DATASET_INFERENCE_7B.jsonl",
                            "DEP_INFERENCE/QA_DATASET_INFERENCE_32B.jsonl"]
OUTPUT_CSV = "ANSWER/QA_DATASET_ANSWER_EVAL.csv"

def main():
    with open(OUTPUT_CSV, "w") as f:
        f.write("Model, BLEU, ROUGE1, ROUGE2, ROUGEL, BERTScore_F1\n")

    for PREDICT_QWEN_VL_PATH in PREDICT_QWEN_VL_PATH_LIST:
        df_gt = pd.read_json(QA_DATASET_GEMINI_PATH, lines=True)
        df_pr = pd.read_json(PREDICT_QWEN_VL_PATH, lines=True)
        df_pr = df_pr.rename(columns={"Answer": "Predicted_Answer"})

        merged_df = pd.merge(df_gt, df_pr, on="QA_number", how="inner")
        reference_answers = merged_df["Answer"].tolist()
        predicted_answers = merged_df["Predicted_Answer"].tolist()
        results = calculate_answer_scores(predicted_answers, reference_answers)
        with open(OUTPUT_CSV, "a") as f:
            PREDICT_ID = PREDICT_QWEN_VL_PATH.split("/")[1].replace("QA_DATASET_INFERENCE_", "").replace(".jsonl", "")
            f.write(f"{PREDICT_ID},{results['bleu']:.4f},{results['rouge1']:.4f},{results['rouge2']:.4f},{results['rougeL']:.4f},{results['bertscore_f1']:.4f}\n")
        




if __name__ == "__main__":
    main()