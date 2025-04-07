# General
from tqdm.auto import tqdm
import json
import re
# API
from g4f.client import AsyncClient
# Async
from tqdm.asyncio import tqdm_asyncio
from aiolimiter import AsyncLimiter
import asyncio


# Shared rate limiter
rate_limit = AsyncLimiter(max_rate=1, time_period=1)

async def generate_story_async(prompt, client, debug=False):
    """
    Generate a story asynchronously using the API client.
    """
    raw_content = None  # Initialize to avoid unbound variable issues
    async with rate_limit:
        try:
            # Call the OpenAI API
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Extract content
            raw_content = response.choices[0].message.content.strip()
            if debug: print("\nRaw API Response:", raw_content)
            
            # Attempt to parse as JSON directly
            try:
                story_dict = json.loads(raw_content)
                return story_dict.get("story")
            # Fallback to regex-based JSON extraction
            except json.JSONDecodeError:
                match = re.search(r'\{\s*"story"\s*:\s*".*?"\s*\}', raw_content)
                if match:
                    json_content = match.group(0)
                    if debug: print("\nExtracted JSON:", json_content)
                    story_dict = json.loads(json_content)
                    return story_dict.get("story")
                else:
                    print("\nERROR: No valid JSON found in the response.")
                    return None

        except Exception as e:
            print(f"ERROR: {e}.\nPrompt: {prompt}\nRaw response: {raw_content}")
            return None


async def process_chunk_async(chunk_prompts_dict, client, debug=False):
    """
    Process a chunk of prompts asynchronously.
    """
    tasks = []
    for prompt_dict in chunk_prompts_dict:
        prompt = prompt_dict["prompt"]
        ID = prompt_dict["ID"]
        # Create async tasks
        tasks.append(generate_story_async(prompt, client, debug=debug))

    # Execute all tasks concurrently
    stories = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    stories_dict_list = []
    for story, prompt_dict in zip(stories, chunk_prompts_dict):
        stories_dict_list.append({"story": story, "ID": prompt_dict["ID"]})
    
    return stories_dict_list

async def async_api_calls(input_prompts_file, output_stories_file, save_mode="json", debug=False):
    """
    Process API calls asynchronously without multithreading, with live progress tracking.

    Args:
        input_prompts_file (str): Path to the input prompts file (JSON).
        output_stories_file (str): Path to the output stories file.
        save_mode (str): Format for saving the output, defaults to "json".
        debug (bool): Whether to enable debugging logs.
    """
    # Load prompts from the input file
    with open(input_prompts_file, "r", encoding="utf-8") as f:
        prompts_dict = json.load(f)

    # Define the client
    client = AsyncClient()

    # Progress bar setup
    total_prompts = len(prompts_dict)
    progress_bar = tqdm_asyncio(total=total_prompts, desc="Processing Stories", unit="story")

    # Track task completion with progress bar
    async def track_progress(prompt_dict):
        story = await generate_story_async(prompt_dict["prompt"], client, debug=debug)
        progress_bar.update(1)  # Increment progress bar
        return {"story": story, "ID": prompt_dict["ID"]}

    # Create and run tasks
    tasks = [track_progress(prompt_dict) for prompt_dict in prompts_dict]
    all_stories = await asyncio.gather(*tasks, return_exceptions=True)

    # Close the progress bar
    progress_bar.close()

    # Save generated stories to the output file
    if save_mode == "json":
        with open(output_stories_file, "w", encoding="utf-8") as out_file:
            json.dump(all_stories, out_file, ensure_ascii=False, indent=4)

    print(f"Generated {len(all_stories)} stories. Saved to {output_stories_file}.")
