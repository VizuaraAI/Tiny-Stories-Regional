# General
import os
import time
# API
from requests_helper import sync_api_calls_multithreading
from requests_helper import generate_story
from requests_helper import model_name
from g4f.client import Client


# Files
print(f"\nModel: {model_name}")

start = time.time()

# Making API calls
full_scale = True
if full_scale:
    
    # API Parameter
    mode = "sync"
    evaluate = False
    
    # Story Generation
    if not evaluate: 
        input_name = "prompts_complex-2+-marathi-3M"
        output_name = f"stories_complete/marathi/stories({model_name})-{input_name}"
        
        # Sync API Calls
        if mode == "sync":
            print("\nSYNC GENERATION CALLS")
            print(f"Prompts: {input_name}\n")
            sync_api_calls_multithreading(
                input_prompts_file  = f"split1/{input_name}.json",
                output_stories_file = f"{output_name}.json",
                threads=16,
                truncate=0,    # Resume progress after first x% of input prompts  
                dry_run=False  # Single threaded dry run to check if read write occurs correctly
            )
            
    # Evaluation
    elif evaluate: 
        # Place all <.json> files containing stories to evaluate in <samples/directory>
        for filename in os.listdir("samples"):  
            print("\n")
            
            # File Name 
            eval_lang = "Marathi"                     # Hindi, Marathi, Bangla 
            input_name = filename[:-5]
            output_name = f"eval/eval-{input_name}" 
            
            # API Calls
            print("\nSYNC EVAL CALLS")
            print(f"Evaluating: {input_name}\n")
            sync_api_calls_multithreading(
                input_prompts_file  = f"samples/{input_name}.json",
                output_stories_file = f"{output_name}.json",
                language = eval_lang,
                threads=16,
                dry_run=False,
                evaluate=True
            )
            
            # Sleep
            time.sleep(2)

end = time.time()
print(f"\nTime taken: {(end-start)/60} mins\n")
    
    
# Single API Call
single = False
if single:
    client = Client()
    response = client.chat.completions.create(
        model="grok-beta",
        messages=[{"role": "user", "content": "Hello"}]
    )
    print(response.choices[0].message.content)

