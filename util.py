def generate_chunks_from_list(list, max_total_chunks, divisor = 2):
    list_of_chunks = []
    chunk = []
    incrementor = 0
    list_length = len(list)
    list_length_remainder = list_length % divisor
    round_to_nearest_even_length = None
    # this allows odd list lengths to be divsible for chunking
    if list_length_remainder != 0:
        round_to_nearest_even_length = 2 * round((list_length / 2))
    processed_list_length = round_to_nearest_even_length if round_to_nearest_even_length != None else list_length
    # round chunk amount to avoid floats
    chunk_amount = round((processed_list_length / max_total_chunks))

    for index, item in enumerate(list):
        # to simplify the math, this allows any remaining items to be added to the end of the chunks, which will not be consistent with the configured chunk length
        if incrementor != (chunk_amount - 1) and index == (len(list) - 1):
            # flatten chunks to access the total indices of each item
            flattened_chunks = [item for sub_list in list_of_chunks for item in sub_list]
            # add the remaining items that are left starting at the last known added index
            list_of_chunks.append(list[len(flattened_chunks):])
            break
        # store item in temporary list used for individual chunk   
        chunk.append(item)
        # if incrementor reaches chunk amount, 
        if incrementor == (chunk_amount - 1):
            # to avoid mutating list because of pass by reference, must copy temp list
            copied_chunk = chunk.copy()
            # add to global chunks list
            list_of_chunks.append(copied_chunk)
            # reset chunk list for next chunk to be populated
            # loop continues and does not get incremented on because incrementor should start at zero when beginning new chunk
            chunk.clear()
            incrementor = 0
            continue  
        incrementor += 1
    return list_of_chunks
    