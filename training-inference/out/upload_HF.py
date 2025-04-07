from huggingface_hub import HfApi

# Local directory containing model files
local_dir = r"out/train-more"

# Repository name
repo_name = "TinyStories-Regional/hindi-generated_4o-mini_2M"

# Initialize the HfApi
api = HfApi()
# Access token for private repo
access_token = ""

# Upload the folder
api.upload_folder(
    folder_path=local_dir,
    path_in_repo="",  # Root directory in the repo
    repo_id=repo_name,
    token=access_token
)