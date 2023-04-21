import os
import openai
import re
from dotenv import load_dotenv

# load env variables
load_dotenv('.env.dev')
openai.api_key = os.getenv("OPEN_AI_API_KEY")

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
    prompt = "Classify each word as either a 0 or 1. If a word is classified as 1, it means it's likely a " + category + ", if not, it's not likely a " + category + ". Always return a word wrapped in single quotes, followed by a colon, and then ending with the number (0 or 1). Do not include any line breaks or new lines in the output, separating each item with a comma only. Here is an example of what the output should look like: (e.g.'string1':1, 'string2':0).\n\n'{}'\n\nClassification scores:"
    entities_output = []
    api_called_count = 0
    print('chunked_entities length: ', len(chunked_entities))
    for entities in chunked_entities:
        # add list of entities into prompt to be sent to the open ai API
        prompt_with_injected_entities = prompt.format("','".join(entities))
        try:
            print('open ai is called')
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
    print(entities_output)      
    # loop through classified entities and add should keep property
    for entity in entities_output:
        if entity in entities_tree:
            entities_tree[entity]["should_keep"] = True
    # remove entity from entity_tree if should keep property does not exist which means it is most likely not a credit card
    for entity_tree_key in list(entities_tree.keys()):
        if not "should_keep" in entities_tree[entity_tree_key]:
            entities_tree.pop(entity_tree_key)

    return entities_tree