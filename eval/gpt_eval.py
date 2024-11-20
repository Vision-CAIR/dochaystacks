from openai import OpenAI
import json
from tqdm import tqdm
import argparse

class GPTEVAL:
    
    def __init__(self, outputs_path = None, api_key = "gpt-3.5-turbo"):
        
        self.api_key = api_key
        self.client = OpenAI()
        self.outputs_path = outputs_path
        
        with open(outputs_path, "r") as fp:
            self.responses = json.load(fp)

        self.template = "Task: You are an evaluator. Compare the Predicted Answer with the True Answer and determine if the Predicted Answer is Correct or Incorrect.\n\Instructions:\n1. If the Predicted Answer provides the same information or a reasonable interpretation of the True Answer, respond with 'Correct.'\n2. If the Predicted Answer does not match or does not reasonably interpret the True Answer, respond with 'Incorrect.'\n\nImportant: Answer only with 'Correct' or 'Incorrect' - no explanations.\n\nInput:\nQuestion: {}\nTrue Answer: {}\nPredicted Answer: {}\n"

    def eval(self):
        
        eval_results = [{"Accuracy": 0}]
        accuracy = 0
        summary = 0
        for response in tqdm(self.responses):
            
            imageId = response["imageId"]
            question = response["question"]
            ground_truth = response["ground_truth"]
            prediction = response["response"].strip()
            
            if prediction == '':
                eval_result = "Incorrect"
            else:
                completion = self.client.chat.completions.create(
                    model = self.api_key,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": self.template.format(question, ground_truth, prediction)
                                }
                            ]
                        }
                    ],
                )
                eval_result = completion.choices[0].message.content
            
            if eval_result.lower() == "correct":
                accuracy += 1
            summary += 1
                
            eval_results.append({
                "imageId": imageId,
                "question": question,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "eval": eval_result
            })
            
        eval_results[0]["Accuracy"] = accuracy / summary
        
        with open(self.outputs_path.replace(".json", "_eval.json"), "w") as fp:
            json.dump(eval_results, fp, indent = 2)
            
if __name__ == "__main__":
    
    args = argparse.ArgumentParser()
    args.add_argument("--api_key", type = str, default = "gpt-3.5-turbo")
    args.add_argument("--outpath", type = str, default = "./output")
    args = args.parse_args()

    eval = GPTEVAL(outputs_path = args.outpath)
    eval.eval()