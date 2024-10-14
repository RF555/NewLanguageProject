import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from TransformersUtilityFunctions import get_vec
import pickle as pkl
from sentence_similarity import compare_sentences

# Load NLTK stop words
stop_words = set(stopwords.words('english'))

# Current vocabulary
curr_vocab = {'mistake', 'cow', 'undergraduate', 'acknowledge', 'property', 'turning', 'color', 'playfulness',
              'personality',
              'fine-tune', 'bottle', 'road', 'call', 'core', 'build', 'replace', 'assemble', 'departure', 'blouse',
              'reorganize', 'slander', 'vigour', 'issue', 'makeup', 'dilate', 'impale', 'pull', 'centre',
              'invigorate',
              'restore', 'achieve', 'adaption', 'self-rule', 'stress', 'garment', 'simple', 'cupboard', 'collect',
              'sepulchre', 'didactics', 'survive', 'bed', 'demobilisation', 'nonfiction', 'headland', 'associate',
              'forbid',
              'jurist', 'component', 'fireplace', 'deceit', 'affect', 'salary', 'ask', 'commission', 'cry',
              'flare-up',
              'elide', 'submit', 'revolutionary', 'artefact', 'plead', 'endanger', 'drama', 'airplane',
              'standardization',
              'genre', 'psychotherapy', 'rate', 'mountain', 'being', 'family', 'discomfit', 'delete', 'objective',
              'cloth',
              'defy', 'cohesion', 'gown', 'aliveness', 'break', 'warmness', 'postponement', 'stand-in', 'hair',
              'deal',
              'suburb', 'stronghold', 'moment', 'sympathy', 'arouse', 'flavoring', 'counsel', 'remind', 'walk',
              'lodging',
              'quit', 'falseness', 'freebooter', 'future', 'flowing', 'first', 'misidentify', 'rum', 'compare',
              'quarters',
              'enclosing', 'forepart', 'happen', 'part', 'bureau', 'memorize', 'transmitting', 'speech', 'man',
              'mountebank',
              'emphasize', 'element', 'tower', 'dispute', 'broadness', 'distract', 'conservationist', 'honor',
              'military',
              'get', 'coolness', 'revenue', 'framework', 'standardize', 'car', 'unresponsiveness', 'criticize',
              'action',
              'discover', 'sickness', 'unwillingness', 'loss', 'depositary', 'attain', 'authorization', 'bewilder',
              'give',
              'holiday', 'acceptance', 'agenda', 'cut', 'cooling', 'print', 'effect', 'scream', 'missile',
              'familiarize',
              'erode', 'lengthen', 'movie', 'pushing', 'distort', 'affection', 'collection', 'interrogation',
              'legislature',
              'epitome', 'mind', 'oxidization', 'disfavor', 'renew', 'denunciation', 'happening', 'clergyman',
              'depression',
              'business', 'reduction', 'gathering', 'exhort', 'commerce', 'judge', 'name', 'ruler', 'adeptness',
              'boredom',
              'laborer', 'jacket', 'declaration', 'straight', 'choreography', 'result', 'bike', 'penalty',
              'promontory',
              'evolve', 'vicinity', 'food', 'regularise', 'stage', 'justification', 'depository', 'violence',
              'period',
              'retailer', 'rebuff', 'subsidiary', 'closet', 'standing', 'somebody', 'demarcation', 'clause', 'time',
              'quantity', 'cook', 'fire', 'crew', 'thank', 'compose', 'remark', 'sale', 'vessel', 'theory',
              'entree',
              'cognizance', 'validity', 'indicate', 'plainness', 'understanding', 'manage', 'purchasing',
              'specialness',
              'forget', 'exactitude', 'occur', 'organization', 'apportioning', 'remain', 'option', 'treatment',
              'round',
              'heartlessness', 'dark', 'invigoration', 'character', 'texture', 'opposition', 'fund', 'tobacco',
              'proposal',
              'paper', 'seizing', 'hybridisation', 'exhilaration', 'dejection', 'disappointment', 'stoppage',
              'deceive',
              'touch', 'recording', 'crimp', 'creativeness', 'kidnap', 'rise', 'amazement', 'wander', 'ease',
              'lessen',
              'diameter', 'excitement', 'starter', 'staircase', 'energize', 'photo', 'functionary', 'monarch',
              'chain',
              'lawmaker', 'reassert', 'furniture', 'bring', 'pillaging', 'shower', 'learn', 'ideology', 'evolution',
              'disaffection', 'quiet', 'office', 'tantalize', 'roast', 'lover', 'vulgarian', 'march', 'prosecute',
              'member',
              'self-sacrifice', 'feeding', 'success', 'favour', 'prognosticate', 'disrupt', 'subjugation', 'train',
              'suppose', 'relieve', 'develop', 'interest', 'endorse', 'put', 'precariousness', 'fertilizer',
              'follow',
              'election', 'enterprise', 'confound', 'halt', 'permit', 'staff', 'shopping', 'award', 'folk',
              'explain',
              'alarm', 'live', 'publishing', 'predict', 'telecom', 'attract', 'impression', 'depress', 'jet',
              'research',
              'wickedness', 'boost', 'hunting', 'calculate', 'resurgence', 'leading', 'court', 'pass', 'corrupt',
              'torment',
              'premise', 'nobleman', 'dissatisfaction', 'press', 'calibration', 'cigarette', 'categorise',
              'current',
              'requirement', 'wrapping', 'problem', 'destruction', 'talking', 'credit', 'disgrace', 'stabilize',
              'division',
              'quietness', 'advantage', 'cohesiveness', 'attend', 'role', 'profit', 'hope', 'thoughtfulness',
              'energy',
              'sculpture', 'spreading', 'six-shooter', 'sector', 'shorten', 'toughness', 'make', 'safeguard',
              'willingness',
              'hamper', 'mixture', 'stabilise', 'funeral', 'access', 'cocaine', 'exhibit', 'protract', 'spending',
              'estate',
              'money', 'ambitiousness', 'attentiveness', 'favourite', 'end', 'defence', 'plunge', 'set',
              'grassland',
              'throw', 'saying', 'counseling', 'web', 'inhabitation', 'transmit', 'clarity', 'tedium', 'railway',
              'watercolor', 'leadership', 'evaluation', 'anesthetize', 'humanities', 'sailing', 'colonization',
              'clothing',
              'disguise', 'vitality', 'seeing', 'bacteria', 'punish', 'retaliation', 'confection', 'understand',
              'directness', 'reword', 'arriviste', 'examine', 'layover', 'wound', 'impotency', 'squelch', 'armour',
              'accomplish', 'fact', 'repress', 'management', 'alienation', 'instigate', 'colorize', 'perspicacity',
              'pet',
              'palliate', 'station', 'utilisation', 'assigning', 'governance', 'unhinge', 'announce', 'coating',
              'strengthen', 'deviltry', 'telephone', 'leg', 'shop', 'personation', 'ready', 'conglomerate',
              'database',
              'defendant', 'renounce', 'standardisation', 'acquire', 'lumber', 'agency', 'please', 'panel',
              'remainder',
              'fulfill', 'frighten', 'power', 'bother', 'consequence', 'looking', 'attacker', 'dishonesty',
              'artifact',
              'malady', 'act', 'heroin', 'defeat', 'request', 'temblor', 'factory', 'offspring', 'reference',
              'penance',
              'policeman', 'disc', 'pretending', 'musicality', 'institution', 'humour', 'reliever', 'rage',
              'watercolour',
              'healthiness', 'filling', 'derision', 'crop', 'colonisation', 'fascinate', 'average', 'injustice',
              'liberation', 'running', 'conditions', 'mural', 'pedal', 'link', 'depart', 'photography',
              'irritability',
              'prolong', 'ordain', 'uncertainty', 'phone', 'provoke', 'danger', 'insist', 'cultivate', 'earthquake',
              'divulge', 'initiate', 'demand', 'worsen', 'assuage', 'ingress', 'oil', 'stand', 'abandon',
              'consumer',
              'personify', 'crystallization', 'depiction', 'negotiate', 'sentimentality', 'tighten', 'seize',
              'infection',
              'wrongness', 'bewitch', 'weakening', 'purchase', 'cape', 'share', 'deftness', 'selflessness',
              'prayer',
              'withdrawal', 'mesmerize', 'knowing', 'broaden', 'young', 'go', 'deport', 'impermeability', 'right',
              'assembling', 'conference', 'neighborhood', 'regret', 'value', 'inhabit', 'malefactor', 'cognisance',
              'surpass', 'classify', 'passion', 'think', 'discourage', 'boy', 'celebration', 'passiveness',
              'evidence',
              'relinquishing', 'wandering', 'fry', 'overcome', 'script', 'see', 'police', 'forego', 'constrict',
              'past',
              'rainfall', 'yell', 'activeness', 'impact', 'mawkishness', 'indication', 'tractability', 'ancestry',
              'deepen',
              'chance', 'analyze', 'landing', 'liquor', 'harass', 'tune', 'oppression', 'giving', 'exclaiming',
              'adjust',
              'dip', 'quash', 'direct'}


# Step 1: Preprocess the sentence by tokenizing, handling stop words, and identifying out-of-vocabulary words
def preprocess_sentence(sentence, vocab):
    # Tokenize sentence into words
    words = sentence.split()

    # Remove stop words only if they are NOT in the limited vocabulary
    processed_words = [
        word for word in words
        if word.lower() not in stop_words or word in vocab
    ]

    # Check which words are out of vocabulary
    out_of_vocab_words = [word for word in processed_words if word not in vocab]

    return processed_words, out_of_vocab_words


# Step 2: Find the closest word in the limited vocabulary using cosine similarity
def find_closest_word(word, vocab_embeddings, word_embeddings):
    if word not in word_embeddings:
        return None  # Handle if the word has no embedding

    word_embedding = word_embeddings[word].reshape(1, -1)  # Reshape to (1, n) for cosine similarity

    # Extract embeddings from the vocabulary embeddings dictionary
    vocab_embedding_list = np.array(list(vocab_embeddings.values()))

    # Find the most similar word from the allowed vocabulary
    # Ensure both embeddings are 2D arrays (reshape if necessary)
    word_embedding_2d = word_embedding.reshape(1, -1)  # Convert to 2D array if it's 1D
    vocab_embeddings_2d = np.vstack(list(vocab_embeddings.values()))  # Stack vocab embeddings into a 2D array

    similarities = cosine_similarity(word_embedding_2d, vocab_embeddings_2d)

    # Find the index of the most similar word
    closest_idx = np.argmax(similarities)

    # Return the word corresponding to the closest_idx
    closest_word = list(vocab_embeddings.keys())[closest_idx]

    return closest_word


# Step 3: Reconstruct the sentence using substitutions
def reconstruct_sentence(original_words, substitutions):
    # Rebuild the sentence with substitutions
    transformed_sentence = [
        substitutions.get(word, word) for word in original_words
    ]

    return " ".join(transformed_sentence)


# Step 5: Transform the sentence by combining all steps
def transform_sentence(sentence, vocab, vocab_embeddings, word_embeddings):
    # Step 1: Preprocess
    original_words, out_of_vocab_words = preprocess_sentence(sentence, vocab)

    # Step 2: Substitute words
    substitutions = {}
    for word in out_of_vocab_words:
        closest_word = find_closest_word(word, vocab_embeddings, word_embeddings)
        if closest_word:
            substitutions[word] = closest_word

    # Step 3: Reconstruct the sentence
    transformed_sentence = reconstruct_sentence(original_words, substitutions)

    return transformed_sentence


# Populate embeddings for the limited vocabulary
def build_vocab_embeddings(vocab):
    vocab_embeddings = {}
    for word in vocab:
        vec = get_vec(word)
        if vec is not None:  # Only add the embedding if the word has an embedding
            print(f'{len(vocab_embeddings)}/{len(vocab)}: {word}')
            vocab_embeddings[word] = vec
    return vocab_embeddings


# Populate embeddings for the entire vocabulary (words that can appear in input sentences)
def build_word_embeddings(sentence):
    words = set(sentence.split())  # Get all unique words in the sentence
    word_embeddings = {}
    for word in words:
        vec = get_vec(word)
        if vec is not None:  # Only add the embedding if the word has an embedding
            # print(f'{len(word_embeddings)}/{len(words)}: {word}')
            word_embeddings[word] = vec
    return word_embeddings


def similarity_checker(original_sentence, transformed_sentence):
    similarities = compare_sentences(original_sentence, transformed_sentence)
    print(f"Cosine Similarity (Sentence-BERT): {similarities['cosine_BERT']}")


def create_sentence(original_sentence, embbedings_path="vocab_embeddings_dict.pkl"):
    # vocab_embeddings = build_vocab_embeddings(curr_vocab)
    with open(embbedings_path, 'rb') as file:
        vocab_embeddings = pkl.load(file)

    word_embeddings = build_word_embeddings(original_sentence)

    return transform_sentence(original_sentence, curr_vocab, vocab_embeddings, word_embeddings)


# Example usage
if __name__ == "__main__":
    sentence = "The quick brown fox jumps over the lazy dog."

    vocabulary = curr_vocab

    with open("vocab_embeddings_dict.pkl", 'rb') as file:
        vocab_embeddings = pkl.load(file)
    print(f'vocab_embeddings.....DONE!')
    word_embeddings = build_word_embeddings(sentence)
    print(f'word_embeddings.....DONE!')

    # Transform the sentence
    transformed_sentence = transform_sentence(sentence, vocabulary, vocab_embeddings, word_embeddings)

    print("Original Sentence:", sentence)
    print("Transformed Sentence:", transformed_sentence)
    similarity_checker(sentence, transformed_sentence)
