def parse_out_irrelevant_parts_of_speech(entities, pipeline):
    parsed_out_entities = []
    # if no entities exist, return null
    if len(entities) == 0:
        return []
    
    for entity in entities:
        parsed_entity = ''
        doc = pipeline(entity['entity_text'])
        for token in doc:
            # if token contains POS other than prop noun or noun, do not add to entity str
            if token.pos_ == "PROPN" or token.pos_ == "NOUN":
                parsed_entity += (token.text + ' ')     
        # if entity string is not empty, set it to entity_text inside entity dict
        if parsed_entity != '': 
            entity['entity_text'] = parsed_entity.strip().replace("\n", "")  
        else:
            entity['entity_text'] = ''
        
    # filter out for entities that have empty str 
    for entity in entities:
        if entity['entity_text'] != '':
            parsed_out_entities.append(entity)
    return parsed_out_entities

def switch_out_entity_fragments_for_noun_chunks(entities, noun_chunks):
    entities_with_noun_chunk_replacers = []
    for entity in entities: 
        for noun_chunk in noun_chunks:
            if entity['entity_text'].lower() in noun_chunk['text'].lower():
                entity['entity_text'] = noun_chunk['text'].lower()
        entities_with_noun_chunk_replacers.append(entity)
    return entities_with_noun_chunk_replacers

def map_entities_to_tree(processed_comment):
    entity_tree = {}    
    # if no entities exist, return null
    if len(processed_comment['entities']) == 0:
        return None

    for entity in processed_comment['entities']:
        entity_tree['entity'] = entity['entity_text']
        entity_tree['upvotes'] = int(processed_comment['upvotes'])
        entity_tree['sentiment'] = float(processed_comment['sentiment'])
        entity_tree['id'] = processed_comment['commentId']
        entity_tree['entity_frequency'] = 1
    return entity_tree

def get_entities_from_noun_chunks(noun_chunks, comment, pipeline):
    for noun_chunk in noun_chunks:
        noun_chunk_document = pipeline(noun_chunk['text'])
        noun_chunk_entities = get_entities(noun_chunk_document, comment)
        # if noun chunks have found entities store in place of comment entities
        if len(noun_chunk_entities) != 0:
            comment['entities'] = noun_chunk_entities
    return comment

def get_noun_chunks(document):
    noun_chunks = []
    # create noun chunks from comment text
    for chunk in document.noun_chunks:
        noun_chunks.append({"text": chunk.text})
    return noun_chunks

def get_entities(document, comment):
    entities = []
    # entity text & label extraction
    for entity in document.ents:
        if entity.label_ == "ORG" or entity.label_ == "PRODUCT":
            # store entities data with associated comment id
            entities.append({"entity_text": entity.text.lower(),"comment_id": comment['commentId']})
            
    return entities

def get_sentiment(document, comment):
    return format(document._.blob.polarity, '.2f')

def process_comments(pipeline, comment):
    noun_chunks = []
    cleaned_comment_text = comment['commentText'].strip().replace("\n", "")
    document = pipeline(cleaned_comment_text)

    # store noun chunks to help any possible failures with BERT powered NER model 
    noun_chunks = get_noun_chunks(document)

    # entity text & label extraction
    entities = get_entities(document, comment)
    entities = switch_out_entity_fragments_for_noun_chunks(entities, noun_chunks)
    entities = parse_out_irrelevant_parts_of_speech(entities, pipeline)

    # sentiment detection
    sentiment = get_sentiment(document, cleaned_comment_text)

    # add entities list onto comment
    comment['entities'] = entities
    comment['noun_chunks'] = noun_chunks
    comment['sentiment'] = sentiment

    # if entities have zero length, check if noun chunks may contain additional entities
    if len(entities) == 0:
        comment = get_entities_from_noun_chunks(noun_chunks, comment, pipeline)

    # filter out duped entities from single comments
    comment['entities'] = list({entity['entity_text']:entity for entity in comment['entities']}.values())
    return comment