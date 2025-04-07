from create_prompts import create_json_from_txt_files, random_unique_prompts
import time

# Create json (<lanuage>_prompt-elements.json) to conveniently store nouns, verbs, adjectives and features
create_json = True
if create_json:
    create_json_from_txt_files(
        nouns_file              ="marathi/marathi-nouns.txt",
        adjectives_file         ="marathi/marathi-adjectives.txt",
        verbs_file              ="marathi/marathi-verbs.txt",
        english_features_file   ="marathi/english-features.txt",
        hindi_features_file     ="marathi/marathi-features.txt",
        output_file             ="marathi_prompt-elements.json",
    )
    time.sleep(2)
    
# Generate random unique prompts
random_prompts = True
if random_prompts:
    random_unique_prompts(
                        prompt_elements_json = "marathi_prompt-elements.json",
                        num_prompts          = 3000000,
                        lang                 = "mar",       # eng or mar or beng
                        feature_lang         = "mar",       # english (default) or "" for using regional features
                        prompt_detail        = "2+",
                        output_json          = f"prompts_complex-2+-marathi-[3M].json",
                        p1                   = False        # verbose
    )