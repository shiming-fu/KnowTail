import json
import os
import utils
from settings import model_path
from tqdm.auto import tqdm

def reason_with_lt_options(model_name, demo_num):
    input_path = f"/home/kai/workspace/longtail/dataset/with_options_random/{model_name}_{demo_num}_options_random.json"
    output_path = f"/home/kai/workspace/longtail/outputs/{model_name}/testset_{demo_num}_options_random.json"
    print(f"loading {model_name} for {demo_num}-options-random")
    model, tokenizer = utils.load_model(model_name)
    fo = open(output_path, "w", encoding="utf-8")
    with open(input_path, "r", encoding="utf-8") as fi:
        data = fi.readlines()
        process_bar = tqdm(range(len(data)))
        for line in data:
            case = json.loads(line)
            response = utils.llm_infer(model_name, model, tokenizer, case["prompt"])
            case["options_random_response"] = response
            fo.write(json.dumps(case, ensure_ascii=False)+"\n")
            process_bar.update(1)
    fo.close()

if __name__=="__main__":
    # Node-1
    for model_name, path in model_path.items():
        if path == "":
            continue
        for demo_num in [4,5,6,7,8]:
            reason_with_lt_options(model_name, demo_num)
    pass

            
    
            
    
    