import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
import json
from model import process_comments, map_entities_to_tree 
from util import generate_chunks_from_list
from open_ai import init_open_ai_service
import asyncio

# Load English large model
roberta_nlp = spacy.load("en_core_web_trf")
roberta_nlp.add_pipe('spacytextblob')

#  set global variable to hold data that's read in from json
commentsJSON = None

# read in json data for model input
with open("threadComments.json", "rb") as f:
    commentsJSON = json.load(f)
    
def init_model(commentsJSON):
    entities_tree = {}
    comments_with_entities = []
    if commentsJSON == None:
        return None

    for index, commentData in enumerate(commentsJSON):
        print('index: ' + str(index))
        for comment in commentData['comments']:
            # get entities from every top + reply comments
            processed_comment = process_comments(roberta_nlp, comment)
            comments_with_entities.append(processed_comment)                
            entity_tree = map_entities_to_tree(processed_comment)
            if entity_tree is not None:
                if entity_tree['entity'] in entities_tree:
                    entities_tree[entity_tree['entity']]['upvotes'] += entity_tree['upvotes']
                    entities_tree[entity_tree['entity']]['sentiment'] += entity_tree['sentiment']
                    entities_tree[entity_tree['entity']]['entity_frequency'] += 1
                else:
                    entities_tree[entity_tree['entity']] = entity_tree

        # mutate comments list from comment thread dict and add comments with entities
        commentData['comments'] = comments_with_entities

    # chunk input for openAI
    # wrap keys() in list to allow for numerical index reference
    entity_chunks = generate_chunks_from_list(list(entities_tree.keys()), 10, 2)
    return [commentsJSON, entities_tree, entity_chunks]         


async def main():
    # store output and seralize for writing json file
    comment_threads, entities_tree, entity_chunks = init_model(commentsJSON)
    # run open ai service to classify model entities for category relevancy
    classified_entities_tree = await init_open_ai_service(entity_chunks, 'credit card', entities_tree)
    entities_tree_json = json.dumps(classified_entities_tree, indent=4)
    with open("entities_tree.json", "w") as outfile:
        outfile.write(entities_tree_json)

asyncio.run(main())