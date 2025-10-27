import json
import pandas as pd

# JSON data paths
data_path = {
    "qwen2-1.5b-rag": "./micro_accuracy/qwen2-1.5b_rag_testset_rag_4_metrics.json",
    "qwen2-1.5b-cot": "./micro_accuracy/qwen2-1.5b_few_shot_cot_testset_3_shot_cot_metrics.json",
    "qwen2-1.5b-options-with-desc":"./micro_accuracy/qwen2-1.5b_options_desc_case_testset_4_options_desc_case_metrics.json",
    "gpt-4o-rag": "./micro_accuracy/gpt-4o-2024-08-06_rag_testset_rag_4_metrics.json",
    "gpt-4o-cot": "./micro_accuracy/gpt-4o-2024-08-06_few_shot_cot_testset_3_shot_cot_metrics.json",
    "gpt-4o-options-with-desc":"./micro_accuracy/gpt-4o-2024-08-06_options_desc_case_testset_4_options_desc_case_metrics.json"
}

frequency_path = './micro_accuracy/charge_frequencies.json'  # Path to the frequency JSON file

# Load frequency JSON data
def load_frequency_json():
    with open(frequency_path, 'r', encoding='utf-8') as f:
        frequency_data = json.load(f)
    return frequency_data

# Load model data
def load_json(model):
    file_path = data_path[model]
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
       for line in f.readlines():
           data.append(json.loads(line))
    return data

# Process and save to Excel
def excel():
    combined_df = pd.DataFrame()  # Initialize an empty DataFrame for combining all models' data

    # Load frequency data and sort by frequency in descending order
    frequency_data = load_frequency_json()
    sorted_frequencies = sorted(frequency_data.items(), key=lambda x: x[1], reverse=True)
    sorted_charges = [charge for charge, freq in sorted_frequencies]  # Extract charges in sorted order

    for model in ["qwen2-1.5b-rag", "qwen2-1.5b-cot", "gpt-4o-rag", "gpt-4o-cot","qwen2-1.5b-options-with-desc","gpt-4o-options-with-desc"]:
        data = load_json(model)

        # Create a DataFrame for the current model
        df = pd.DataFrame(data)

        # Create a transformed DataFrame with 'Charge' and model-specific F1 score column
        df_transformed_with_headers = pd.DataFrame({
            'Charge': df['charge'],         # Extract 'charge' column
            f'{model}': df['accuracy']      # Extract 'f1_score' column
        })

        # Merge the current model's DataFrame with the combined DataFrame
        if combined_df.empty:
            combined_df = df_transformed_with_headers
        else:
            # Merge on 'Charge' column
            combined_df = pd.merge(combined_df, df_transformed_with_headers, on='Charge', how='outer')

    # Add frequency column
    combined_df['Frequency'] = combined_df['Charge'].map(frequency_data)  # Add frequency column

    # Reorder the DataFrame based on sorted charges, and reindex to skip missing charges
    combined_df = combined_df.set_index('Charge').reindex(sorted_charges).dropna(how='all').reset_index()

    # Save the combined DataFrame to Excel
    excel_path_with_headers = './charges_f1_scores_with_frequencies_sorted.xlsx'
    combined_df.to_excel(excel_path_with_headers, index=False)

if __name__ == '__main__':
    excel()