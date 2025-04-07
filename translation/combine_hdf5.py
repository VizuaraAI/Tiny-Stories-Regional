from h5py import create_output_hdf, append_to_hdf5
import numpy as np
import h5py


# Combine multiple HDF5 files to form a Master file
def combine_hdf5(input_foler, combined_hdf, num_gpu):
    """ Iterates through each hdf5 appending it to master hdf5 """
    # Master file path
    master_file = f"{combined_hdf}.h5"
    
    # Iterate
    print(f"\nAppending {num_gpu} files to {combined_hdf}:")
    for i in range(num_gpu):
        # Skip a certain file
        if i == 5: continue
        # Current hdf5
        source_file = f"translated_stories_{i}.h5"
        # Open current hdf5
        with h5py.File(source_file, "r") as src, h5py.File(master_file, 'a') as dest:
            # Access the dataset in the source file
            src_data = src["translated_stories"]
            # Access the dataset in the destination file
            dest_data = dest["translated_stories"]
            # Calculate the new size of the destination dataset
            new_size = dest_data.shape[0] + src_data.shape[0]
            print(f"{source_file} has {src_data.shape[0]} stories")
            # Resize the destination dataset to accommodate the new data
            dest_data.resize((new_size,))
            # Append the data from the source dataset to the destination dataset
            dest_data[-src_data.shape[0]:] = src_data[:]
    
    # Check combined file
    with h5py.File(master_file, "r") as master:
        stories = master["translated_stories"]
        print("Thus,")
        print(f"{master_file} contains {stories.shape[0]} stories")
        print("Appending Complete\n")


# Create master output
input_foler = "."
combined_hdf5 = "hindi_translated_exc5_1M"
output_file = f"{combined_hdf5}"
# Combine HDF5
print("\nCreating Master hdf5:")
create_output_hdf(output_file, p1 = True)
# Append to master
combine_hdf5(input_foler, combined_hdf5, 8)