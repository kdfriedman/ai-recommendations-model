import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
import json
from model import process_comments, map_entities_to_tree
import transformers
from transformers import GPT2Tokenizer
transformers.logging.set_verbosity_error() 


# Load English large model
roberta_nlp = spacy.load("en_core_web_trf")
roberta_nlp.add_pipe('spacytextblob')

#  set global variable to hold data that's read in from json
commentsJSON = None

# read in json data for model input
with open("threadComments.json", "rb") as f:
    commentsJSON = json.load(f)

def get_tokens_length_from_entities(entities_tree):
    entities_tree_keys_string = ', '.join(entities_tree.keys())
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    entities_tree_keys_tokens = tokenizer.encode(entities_tree_keys_string)
    return len(entities_tree_keys_tokens)
    
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
    # get token amount to help chunk api calls to open ai for further processing 
    get_tokens_length_from_entities(entities_tree)
    return [commentsJSON, entities_tree]         

# store output and seralize for writing json file
comment_threads, entities_tree = init_model(commentsJSON)
entities_tree_json = json.dumps(entities_tree, indent=4)

with open("entities_tree.json", "w") as outfile:
    outfile.write(entities_tree_json)
