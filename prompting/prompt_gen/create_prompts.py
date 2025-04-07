from pathlib import Path
from tqdm import tqdm
import random
import json
import os


def random_unique_prompts(prompt_elements_json, num_prompts, lang = "hin", feature_lang = "english", prompt_detail = 1, output_json = "prompts.json", p1=True, p2=False):
    """
    Generates and prints `num` unique random triplets (noun, verb, adjective) 
    from a JSON file, ensuring no duplicates. Optionally prints details of 
    each triplet and the list of generated IDs.

    Args:
    - json_file (str): Path to the JSON file with "nouns", "verbs", and "adjectives" lists.
    - num (int): Number of unique triplets to generate.
    - p1 (bool, optional): If `True`, prints the selected noun, verb, and adjective. Defaults to `True`.
    - p2 (bool, optional): If `True`, prints the triplet ID and current IDs list. Defaults to `False`.

    Returns:
    - None: Prints the generated prompts and IDs.

    Example:
    >>> random_unique_prompts("hindi_words.json", 2, p1=True, p2=True)
    nouns[1]: पढ़ाई
    verbs[2]: खेलना
    adjectives[0]: सुंदर
    121
    nouns[0]: सूरज
    verbs[1]: पढ़ना
    adjectives[2]: शांत
    102
    Current ids: ['121', '102']
    2 random unique prompts created
    """

    # Unique prompt ID and counter 
    ids = set()
    id2s = set()
    prompts = 0
    count_repeats = 0
    
    with open(prompt_elements_json, "r", encoding="utf-8") as file:
        # nouns, verbs adjectives, endings
        data = json.load(file)
        nouns = data["nouns"]
        verbs = data["verbs"]
        adjectives = data["adjectives"]
        if feature_lang == "english":
            features = data["english_features"]
        else:
            features = data["regional_features"]
        
        # Iterating
        print("")
        prompts_dict_list = []
        with tqdm(total=num_prompts, desc="Generating Prompts", unit="prompt") as pbar:
            while prompts < num_prompts:
                # Get a quad
                while True:
                    # Randomly sample a triplet
                    idx_noun = random.randint(0, len(nouns)-1)
                    idx_verb = random.randint(0, len(verbs)-1)
                    idx_feature1 = random.randint(0, len(features)-1)
                    idx_adjective = random.randint(0, len(adjectives)-1)
                    # Get quad and triplet 
                    id1 = f"{idx_noun}{idx_verb}{idx_adjective}{idx_feature1}"
                    id2 = f"{idx_noun}{idx_verb}{idx_adjective}"
                    # Check if unique
                    if id1 not in ids and id2 not in id2s:
                        ids.add(id1)
                        id2s.add(id2)
                        break  
                    # Repeats
                    count_repeats += 1
                    
                # Current values
                noun = nouns[idx_noun]
                verb = verbs[idx_verb]
                adjective = adjectives[idx_adjective]
                feature1 = features[idx_feature1]
                
                # Prompt 
                if lang == "hin": language = "Hindi"
                elif lang == "mar": language = "Marathi"
                elif lang == "beng": language = "Bengali"
                else: raise NotImplementedError

                if prompt_detail == 1:
                    # TinyStories
                    prompt = f'''Write a short story (3-5 paragraphs), in {language} Devanagari script, using elementary words that 5-7-year-old children would understand. The story should utilize the verb "{verb}", the noun "{noun}", and the adjective "{adjective}". The story should also naturally integrate the following features: "{feature1}". Remember to only use simple words and keep the story short!\n\nReturn the output as a JSON dictionary in the following format:\n{{\n    "story": "your_generated_story"\n}}'''
                    
                elif prompt_detail == 2:
                    # Story structure + tone
                    prompt = f'''Write a short story in {language} (in Devanagari script) suitable for 5-to-7-year-old children. Use simple, easy-to-understand words and limit the story to 3-4 short paragraphs (around 200-300 words). The story should feature a clear beginning, middle, and end. Incorporate the verb "{verb}", the noun "{noun}", and the adjective "{adjective}" naturally into the story. The story should also integrate the conclusion/tone "{feature1}" through actions and outcomes, without directly stating the tone (e.g., do not use "खुश" or similar words explicitly). Remember to only use simple words and keep the story short!\n\nReturn the output as a JSON dictionary in the following format:\n{{\n    "story": "your_generated_story"\n}}'''
                
                elif prompt_detail == "2+":
                    # Increased word limit
                    prompt = f'''Write a short story in {language} suitable for 5-to-7-year-old children. Use simple, easy-to-understand words and limit the story to 3-4 short paragraphs (around 350-500 words). The story should feature a clear beginning, middle, and end. Incorporate the verb "{verb}", the noun "{noun}", and the adjective "{adjective}" naturally into the story. The story should also integrate the conclusion/tone "{feature1}" through actions and outcomes, without directly stating the tone (e.g., do not use "आनंदी" or similar words explicitly). Remember to only use simple words and keep the story short!\n\nReturn the output as a JSON dictionary in the following format:\n{{\n    "story": "your_generated_story"\n}}'''
                    
                elif prompt_detail == 3:
                    # Dialogues added + themes 
                    prompt = f'''Write a short story in {language} (in Devanagari script) suitable for 5-to-7-year-old children. Use simple, easy-to-understand words and limit the story to 3-4 short paragraphs (around 200-300 words). The story should feature a clear beginning, middle, and end. Incorporate the verb "{verb}", the noun "{noun}", and the adjective "{adjective}" naturally into the story. Include at most three dialogues between characters, ensuring that sentence structure is straightforward and age-appropriate. Use themes like friendship, adventure, or learning a life lesson, and ensure the moral of the story is implicit through the resolution, without labeling it as a "moral". The story should also integrate the conclusion/tone "{feature1}" through actions and outcomes, without directly stating the tone (e.g., do not use "खुश" or similar words explicitly). Avoid overcomplicating sentences or introducing unfamiliar words, focusing on a gentle and engaging narrative suitable for young readers. Remember limit the story to 3-4 short paragraphs (around 200-300 words).\n\nReturn the output as a JSON dictionary in the following format:\n{{\n    "story": "your_generated_story"\n}}'''
                    
                elif prompt_detail == 4:
                    # Famous stories added
                    prompt = f'''Write a short story in {language} (in Devanagari script) story suitable for 5-to-7-year-old children. Use simple, easy-to-understand words and limit the story to 3-4 short paragraphs (around 200-300 words). The story should feature a clear beginning, middle, and end. Use the verb "{verb}", the noun "{noun}", and the adjective "{adjective}" naturally within the story. The story should include at most three dialogues between the characters, keeping them simple and clear. Take inspiration from popular Hindi children's novels such as "Tenali Raman Stories", "Chacha Chaudhary" and "Panchatantra". Use themes like friendship, adventure, or learning a life lesson, and ensure the moral of the story is implicit through the resolution, without labeling it as a "moral". The story should also integrate the conclusion/tone "{feature1}" through actions and outcomes, without directly stating the tone (e.g., do not use "खुश" or similar words explicitly). Avoid overcomplicating sentences or introducing unfamiliar words, focusing on a gentle and engaging narrative suitable for young readers. Remember limit the story to 3-4 short paragraphs (around 200-300 words).\n\nReturn the output as a JSON dictionary in the following format:\n{{\n    "story": "your_generated_story"\n}}'''
                
                elif prompt_detail == "4+":
                    # Famous stories added
                    prompt = f'''Write a short story in {language} (in Devanagari script) story suitable for 5-to-7-year-old children. Use simple, easy-to-understand words and limit the story to 3-4 short paragraphs (around 350-500 words). The story should feature a clear beginning, middle, and end. Use the verb "{verb}", the noun "{noun}", and the adjective "{adjective}" naturally within the story. The story should include at most three dialogues between the characters, keeping them simple and clear. Take inspiration from popular Hindi children's novels such as "Tenali Raman Stories", "Chacha Chaudhary" and "Panchatantra". Use themes like friendship, adventure, or learning a life lesson, and ensure the moral of the story is implicit through the resolution, without labeling it as a "moral". The story should also integrate the conclusion/tone "{feature1}" through actions and outcomes, without directly stating the tone (e.g., do not use "खुश" or similar words explicitly). Avoid overcomplicating sentences or introducing unfamiliar words, focusing on a gentle and engaging narrative suitable for young readers. Remember limit the story to 3-4 short paragraphs (around 350-500 words).\n\nReturn the output as a JSON dictionary in the following format:\n{{\n    "story": "your_generated_story"\n}}'''
                    
                else:
                    # Supporting characters and natural elements 
                    prompt = f'''Write a short story in {language} (in Devanagari script) story suitable for 5-to-7-year-old children. Use simple, easy-to-understand words and limit the story to 3-4 short paragraphs (around 200-300 words). The story should feature a clear beginning, middle, and end. Use the verb "{verb}", the noun "{noun}", and the adjective "{adjective}" naturally within the story. The story should include at most three dialogues between the characters, keeping them simple and clear. Take inspiration from popular Hindi children's novels such as "Tenali Raman Stories", "Chacha Chaudhary" and "Panchatantra". Incorporate natural elements like trees, rivers, animals, or village life to make the story more engaging for children. Introduce supporting characters like animals, villagers, or magical creatures to add depth. Use themes like friendship, adventure, or learning a life lesson, and ensure the moral of the story is implicit through the resolution, without labeling it as a "moral". The story should also integrate the conclusion/tone "{feature1}" through actions and outcomes, without directly stating the tone (e.g., do not use "खुश" or similar words explicitly). Avoid overcomplicating sentences or introducing unfamiliar words, focusing on a gentle and engaging narrative suitable for young readers. Remember limit the story to 3-4 short paragraphs (around 200-300 words).\n\nReturn the output as a JSON dictionary in the following format:\n{{\n    "story": "your_generated_story"\n}}'''
                
                # Prompt dict   
                prompt_dict = {
                    "noun": noun,
                    "verb": verb,
                    "adjective": adjective,
                    "feature1": feature1,
                    "prompt": prompt,
                    "ID": id1
                }
                
                # Appending 
                prompts_dict_list.append(prompt_dict)
                
                # Write all the prompts at onec to the file
                if (prompts == num_prompts-1) and prompts !=0:
                    # If output file exists 
                    if os.path.exists(output_json):
                        # Open and load existing data
                        with open(output_json, 'r', encoding='utf-8') as file:
                            data = json.load(file)
                        data.extend(prompts_dict_list)
                        # Write to file
                        with open(output_json, 'w', encoding='utf-8') as file:
                            json.dump(data, file, ensure_ascii=False, indent=4)
                    # If output file does not exist
                    else:
                        with open(output_json, 'w', encoding='utf-8') as f:
                            json.dump(prompts_dict_list, f, ensure_ascii=False, indent=4)
                    # Reset
                    prompts_dict_list = []
                
                # Print 
                if p1: 
                    print(f"\nnouns[{idx_noun}]: {noun}")
                    print(f"verbs[{idx_verb}]: {verb}")
                    print(f"adjectives[{idx_adjective}]: {adjective}")
                    print(f"features[{idx_feature1}]: {feature1}")
                    print("Thus,")
                    print(f"Prompt: {prompt}")
                    if p2: print(id)
                
                # Update counter
                prompts += 1 
                pbar.update(1)
        
        # Verbose
        if p2: print(f"\nCurrent ids: {ids}\n")
        print(f"\n{num_prompts} random unique prompts created")
        print(f"There were {count_repeats} overlaps\n")



def create_json_from_txt_files(nouns_file, adjectives_file, verbs_file, english_features_file, hindi_features_file, output_file = "hindi_prompts-elements.json"):
    """
    Reads nouns, adjectives, and verbs from text files, processes them,
    and writes the data to a JSON file.
    
    Args:
    - nouns_file (str): Path to the text file containing nouns.
    - adjectives_file (str): Path to the text file containing adjectives.
    - verbs_file (str): Path to the text file containing verbs.
    - output_file (str): Path to the output JSON file.
    
    Returns:
    - None
    """
    
    # Function to read words and process them
    def read_words_from_file(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            return [line.strip().strip(",").strip("'") for line in file if line.strip()]
    
    # Read and process words
    nouns = read_words_from_file(nouns_file)
    adjectives = read_words_from_file(adjectives_file)
    verbs = read_words_from_file(verbs_file)
    english_features = read_words_from_file(english_features_file)
    hindi_features = read_words_from_file(hindi_features_file)
    
    # Create JSON structure
    data = {
        "nouns": nouns,
        "verbs": verbs,
        "adjectives": adjectives,
        "regional_features": hindi_features,
        "english_features": english_features
    }
    
    # Write to JSON file
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
    
    print(f"\nJSON file '{output_file}' created successfully!\n")


    