import json

def recall_of_bm25IR(path="dataset/origin/testset_for_llm_with_similar_cases.json", top_k=10):
    total_instance = 0
    recalled_label = 0
    with open(path, "r", encoding="utf-8") as fi:
        for line in fi.readlines():
            item = json.loads(line)
            ground_truth = item["charge"]
            retrieved_charges = [case['meta']['accusation'][0] for case in item['similar_cases'][:top_k]]
            total_instance+=1
            if ground_truth in retrieved_charges:
                recalled_label+=1
    print(f"recall: {round(recalled_label/total_instance, 4)}")
            
            
if __name__=="__main__":
    recall_of_bm25IR()