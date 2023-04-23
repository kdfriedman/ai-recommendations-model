import os
import openai
import re
from dotenv import load_dotenv
import json


# load env variables
load_dotenv('.env.dev')
openai.api_key = os.getenv("OPEN_AI_API_KEY")

def sum_entity_properties(entities_tree):
    # sum up all the upvotes and sentiment scores for each entity
    for entity in entities_tree:
        # remove all properties other than 'total' from entity
        entity['total'] = entity['upvotes'] + entity['sentiment'] + entity['entity_frequency']
        entity.pop('upvotes')
        entity.pop('sentiment')
        entity.pop('entity_frequency')
        entity.pop('id')
        entity.pop('should_keep')
    return entities_tree
        

def clean_open_ai_output(classified_entities):
    # remove all single quotes from entities
    entities_with_single_quotes_removed = list(map(lambda entity: entity.replace("'", ""), classified_entities))
    # remove white space and line breaks from entities
    entities_with_line_breaks_and_space_removed = list(map(lambda entity: entity.replace("\n", "").strip(), entities_with_single_quotes_removed))
    # omit all entities that have a score of 0, this helps remove non-categorically relevant entities
    entities_filtered_by_positive_score = list(filter(lambda entity: (entity.find('1') != -1), entities_with_line_breaks_and_space_removed)) 
    # format entity to only contain entity itself, excluding score
    entities_without_scores = list(map(lambda entity: re.match("(.+):([\s \d]+)", entity).group(1), entities_filtered_by_positive_score))
    return entities_without_scores

async def complete_text_gpt(prompt, max_tokens = 3000):
    print('calling open api service')
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": prompt}
        ],
            temperature=0,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        user=os.getenv("OPEN_AI_USER_HASH")
    )
    return response

async def init_open_ai_service(chunked_entities, category, entities_tree):
    prompt_binary_comparison = "Classify each word as either a 0 or 1. If a word is classified as 1, it means it's likely a " + category + ", if not, it's not likely a " + category + ". Always return a word wrapped in single quotes, followed by a colon, and then ending with the number (0 or 1). Do not include any line breaks or new lines in the output, separating each item with a comma only. Here is an example of what the output should look like: (e.g.'string1':1, 'string2':0).\n\n'{}'\n\nClassification scores:"
    prompt_product_classification = "This list contains items that are related to " + category + ". Some items are the companies themselves, some are the " + category + " , and some are neither. Parse through each item in this list and only keep the actual " + category + ". The item that is kept should be available in the world currently. Omit all items that are not actual used " + category + ".\n\n'{}'\n\n"
    entities_output = []
    api_called_count = 0
    print('chunked_entities length: ', len(chunked_entities))
    for entities in chunked_entities:
        # add list of entities into prompt to be sent to the open ai API
        prompt_with_injected_entities = prompt_binary_comparison.format("','".join(entities))
        try:
            classified_entities_result = await complete_text_gpt(prompt_with_injected_entities, 3000)  
            api_called_count += 1
            # store each entity and score in their own index in a list
            classified_entities = classified_entities_result['choices'][0]['message']['content'].split(',')
            cleaned_entities = clean_open_ai_output(classified_entities)
            for entity in cleaned_entities:
                entities_output.append(entity)
        except Exception as e:
            print(e)
            break      
    print('api_called_count: ', api_called_count)   
    # loop through classified entities and add should keep property
    for entity in entities_output:
        if entity in entities_tree:
            entities_tree[entity]["should_keep"] = True
    # remove entity from entity_tree if should keep property does not exist which means it is most likely not a credit card
    for entity_tree_key in list(entities_tree.keys()):
        if not "should_keep" in entities_tree[entity_tree_key]:
            entities_tree.pop(entity_tree_key)
    # sum up all the upvotes, sentiment scores, and entity frequencies for each entity
    summed_entities_tree = sum_entity_properties(entities_tree)
    summed_entities_tree_json = json.dumps(summed_entities_tree)
    try:
        finalized_entities_tree = await complete_text_gpt(prompt_product_classification.format(summed_entities_tree_json), 3000)  
        return finalized_entities_tree
    except Exception as e:
        print(e)