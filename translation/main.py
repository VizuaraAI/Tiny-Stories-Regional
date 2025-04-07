from hdf5 import save_dataset_to_hdf5, combine_hdf5, create_output_hdf
from translate_stories import translate_stories
import h5py

translate = False
if translate:
    # Data variables
    dataset_name = "roneneldan/TinyStories"  # Replace with your dataset name if different
    hdf5_path = "tiny_stories_text.h5"       # Desired path for the HDF5 file
    split = "train"                          # Replace with desired split ('train', 'validation'.)
    # Calling function
    save_dataset_to_hdf5(dataset_name, split, hdf5_path)
        
    # Debug parameters
    debug_dict = {
        "batch_size" : 2,
        "chunk_size" : 1,
        "num_batches": 2
    }

    # Translate Dataset
    translate_stories(
        input_hdf5        = "tiny_stories_text.h5",
        output_hdf5       = "translated_stories",
        batch_size        = 12, # RTX 4000Ada = 12
        chunk_size        = 5000,
        model_name        = "nllb",
        print_trans       = False, 
        debug             = (False, debug_dict),
        save_translations = True,
        gpu_mem           = False
    )
