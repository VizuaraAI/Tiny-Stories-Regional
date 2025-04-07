# General
from tqdm.auto import tqdm
import time
import json
import re
# API 
from g4f.client import Client
from g4f.Provider import PollinationsAI
from g4f.Provider import Copilot
from g4f.Provider import OIVSCode
# Multi-threading
from concurrent.futures import ThreadPoolExecutor
import threading
# Decoding
import unicodedata

model_name = "gpt-4o"
#model_name = "gpt-4o-mini"
#model_name = "llama-3.3-70b"
#model_name = "claude-3.5-sonnet"
#model_name = "grok-beta"


def generate_story(prompt, client, p1=False, debug_errors=False):
    
    # Initialize to avoid UnboundLocalError
    response = None
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{
                "role": "user",
                "content": prompt
            }],
            # provider=Copilot # Uncomment if you desire to use a particular provider on G4F. This can boost speeds but lead to reaching Rate Limit Error faster. 
        )
        
        # Get raw API response
        if response is None: return None
        raw_content = response.choices[0].message.content.strip()
        if raw_content is None: return None
        if p1: 
            print("\nPrompt: ", prompt)
            print("Raw API Response: ", raw_content)
        
        # Attempt to extract JSON using regex
        match = re.search(r'\{\s*"story"\s*:\s*".*?"\s*\}', raw_content)  
        if match:
            json_content = match.group(0)
            if p1: print("Extracted JSON:", json_content)
            # Attempt to parse the JSON
            story_dict = json.loads(json_content)
            # Removing non Hindi/Marathi content
            devanagari_regex = r'[^\u0900-\u097F\u0966-\u096F0-9\s.,!?\"\'।ऽ॰॥\-—…]'
            filtered_story = re.sub(devanagari_regex, '', story_dict['story'])
            filtered_story = unicodedata.normalize('NFC', filtered_story)
            # Returning filtered story
            return filtered_story
        else:
            if debug_errors: print("ERROR: No valid JSON found in the response.")
            return None

    except json.JSONDecodeError as e:
        if debug_errors: print(f"ERROR: JSON parsing error: {e}")
        return None
    except Exception as e:
        if debug_errors: print(f"ERROR: {e}")
        return None  
    

def evaluate_story(story, client, language="Hindi", p1=False, p2=False, p3=False):
    
    # Initialize to avoid UnboundLocalError
    response = None
    
    try:
        prompt = f'''{story}\n\nThe given {language} short story is for 5-7-year-old children. Keeping in mind the target demographic, rate the story on a scale of 1-10 for context awareness, completeness, grammar, fluency, and creativity. Evaluate context awareness by strictly assessing how well the story's middle and end align with the prompt "{prompt1}". Also, provide an overall rating on a scale of 1-10. Only return a  JSON dictionary in the following format: \n{{\n"context awareness": "your_context-awareness_score",\n"completeness": "your_completeness_score",\n"grammar": "your_grammar_score",\n"fluency": "your_fluency_score",\n"creativity": "your_creativity_score",\n"overall": "your_overall_rating"\n}}'''
        if p3: print('\n' + prompt + '\n')
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )
        
        # Get raw API response
        if response is None: return None
        rating = response.choices[0].message.content.strip()
        if rating is None: return None
        if p2: 
            print("\nStory: ", story)
            print("Raw API Response:", rating)
        
        # Return rating for current story
        rating = json.loads(rating)
        return rating
        
    except Exception as e:
        if p1: print(f"ERROR: {e}.\nPrompt:{prompt}\nRaw response: {response}")
        return None  
    

def process_chunk(output_file, chunk_prompts_dict, language="Hindi", lock=None, client=None, evaluate=False):
    
    # Current thread
    worker_name = (threading.current_thread().name)[-4:]
    
    # When generating stories
    if not evaluate:
        stories_dict_list = []
        count_errors = 0
        
        # Make API Calls
        for idx, prompt_dict in tqdm(enumerate(chunk_prompts_dict), desc="Processing chunk", unit="prompt", dynamic_ncols=True, total=len(chunk_prompts_dict)):
            # Extracting prompt and ID
            prompt = prompt_dict["prompt"]
            ID = prompt_dict["ID"]
            story = None
            
            # Generating story (API Call)
            while True: 
                story = generate_story(prompt, client)
                if story is not None: break
                count_errors += 1    

            # Appending results 
            story_dict = {
                "story": story,
                "ID": ID
            }
            stories_dict_list.append(story_dict)
            
            # Write 
            div = len(chunk_prompts_dict)//10
            if ((idx%div==0) or (idx==len(chunk_prompts_dict)-1)) and (idx!=0):
                
                with lock:
                    try:
                        # Read the current content of the file
                        with open(output_file, 'r', encoding='utf-8') as file:
                            current_data = json.load(file)
                    except (FileNotFoundError, json.JSONDecodeError):
                        current_data = []
                        
                    # Append new data
                    current_data.extend(stories_dict_list)
                    # Write updated content back to the file
                    with open(output_file, 'w', encoding='utf-8') as file:
                        json.dump(current_data, file, ensure_ascii=False, indent=4)
                    # Log
                    base_name = output_file[:-5]
                    with open(f"{base_name}-[LOG].txt", 'a') as f:
                        f.write(f"worker ({worker_name}) | written ({len(stories_dict_list)}) stories ({idx/div})th time | current idx: {idx+1} | errors {count_errors}\n")
                    # Reset 
                    stories_dict_list = []
    
    # When evaluating stories
    else:
        rating_dict_list = []
        chunk_stories_dict = chunk_prompts_dict
        
        # Make API Calls
        for idx, story_dict in tqdm(enumerate(chunk_stories_dict), desc="Evaluating chunk", unit="story",dynamic_ncols=True, total=len(chunk_stories_dict)):
            # Extracting story and ID
            story = story_dict["story"]
            ID = story_dict["ID"]
            rating_dict = None
            
            # Rate story
            while rating_dict is None:
                rating_dict = evaluate_story(story, client, language)
            # if rating_dict is None: continue
            
            # Appending 
            id_dict = { "ID": ID }
            rating_dict.update(id_dict)
            rating_dict_list.append(rating_dict)
        
        # Flattended and combined in sync_api_calls
        return rating_dict_list


def sync_api_calls_multithreading(input_prompts_file, output_stories_file, language="Hindi", threads = 16, evaluate = False, truncate = 0, dry_run = False):
    
    # Load prompts from the input file
    with open(input_prompts_file, "r", encoding="utf-8") as f:
        prompts_dict = json.load(f)
        if dry_run: prompts_dict = prompts_dict[:10]

    # Number of threads to use
    num_threads = threads
    if dry_run: num_threads = 1

    # Split prompts into chunks for multithreading
    chunk_size = len(prompts_dict) // num_threads
    chunks = [prompts_dict[i:i + chunk_size] for i in range(0, len(prompts_dict), chunk_size)]
    # Truncate
    if truncate > 0:
        percentage = truncate / 100  # After first x %
        chunks = [chunk[int(len(chunk) * percentage):] for chunk in chunks]

    # Define client 
    client = Client()
    
    # GlobalLock 
    lock = threading.Lock()
    
    # Multithreading
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = executor.map(lambda chunk: process_chunk(output_stories_file, chunk, language, lock, client, evaluate), chunks)
        # Flatten
        if evaluate: 
            all_ratings = [rating_dict for result in results for rating_dict in result]
            with open(output_stories_file, "w", encoding="utf-8") as out_file:
                json.dump(all_ratings, out_file, ensure_ascii=False, indent=4)
    
    # Sanity check
    time.sleep(2)
    with open(output_stories_file, 'r', encoding='utf-8') as file: 
        data = json.load(file)
    print(f"\n\nGenerated {len(data)} data saved to {output_stories_file}")
    