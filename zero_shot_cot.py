import json
import os, time
import utils
from settings import model_path, oneline_models, loacl_models, batch_file_templet
from tqdm.auto import tqdm


def zero_shot_cot(model_name):
    temp_file = "temp.json"
    input_path = "dataset/zero_cot/0_cot.json"
    output_path = f"outputs/{model_name}/testset_zero_shot_cot.json"
    # call local models
    if model_name in oneline_models:
        print(f"construct batch files")
        with open(temp_file, "w", encoding="utf-8") as fo_batch:
            with open(input_path, "r", encoding="utf-8") as fi:
                for line in fi.readlines():
                    case = json.loads(line)
                    batch_file_templet["custom_id"] = str(case["id"])
                    batch_file_templet["body"]["model"] = model_name
                    batch_file_templet["body"]['messages'] = [{"role": "user", "content": case["prompt"]}]
                    fo_batch.write(json.dumps(batch_file_templet, ensure_ascii=False)+"\n")
        utils.invoke_gpt(bf_path=temp_file,
                         ans_file_path=input_path, 
                         output_path=output_path,
                         response_key="zero_cot_response")
    # call GPTs
    elif model_name in loacl_models:
        fo = open(output_path, "w", encoding="utf-8")
        print(f"loading {model_name} for zero cot")
        model, tokenizer = utils.load_model(model_name)
        with open(input_path, "r", encoding="utf-8") as fi:
            data = fi.readlines()
            process_bar = tqdm(range(len(data)))
            for line in data:
                case = json.loads(line)
                response = utils.llm_infer(model_name, model, tokenizer, case["prompt"])
                case["zero_cot_response"] = response
                fo.write(json.dumps(case, ensure_ascii=False)+"\n")
                process_bar.update(1)
        fo.close()

if __name__=="__main__":
    for model_name in ["gpt-4o-2024-08-06"]: # , "gpt-4o-2024-08-06"
        zero_shot_cot(model_name)

            
    
            
    
    