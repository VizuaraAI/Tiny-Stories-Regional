import json
import os

# Specify the directory containing the JSON files
directory = "eval/hindi"
startswith = "eval"

# Read and display
print("")
for filename in sorted(os.listdir(directory)):
    # Check for files that match the correct naming pattern
    if filename.endswith(".json") and filename.startswith(startswith):
        file_path = os.path.join(directory, filename)
    
        # Initialze 
        with open(file_path, mode="r") as f: ratings = json.load(f)
        sum_completeness = 0
        sum_context_awr = 0
        sum_creativity = 0
        sum_fluency = 0
        sum_grammar = 0
        sum_overall = 0
        
        # Sum of ratings
        for rating_dict in ratings: 
            sum_completeness += float(rating_dict["completeness"])
            sum_context_awr += float(rating_dict["context awareness"])
            sum_creativity += float(rating_dict["creativity"])
            sum_fluency += float(rating_dict["fluency"])
            sum_grammar += float(rating_dict["grammar"])
            sum_overall += float(rating_dict["overall"])
            
        # Average dict
        avg = {
            "context awareness": sum_context_awr/len(ratings),
            "completeness": sum_completeness/len(ratings),
            "creativity": sum_creativity/len(ratings),
            "fluency": sum_fluency/len(ratings),
            "grammar": sum_grammar/len(ratings),
            "overall": sum_overall/len(ratings),
        }
        
        # Result
        print(f"\nFor: {file_path}\n")
        print(f"{avg}\n")
        true_overall = (avg["completeness"]+avg["context awareness"]+avg["creativity"]+avg["fluency"]+avg["grammar"])/5
        print(f"True overall: {true_overall:.3f}\n")