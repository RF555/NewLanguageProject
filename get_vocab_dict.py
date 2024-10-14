from TransformersUtilityFunctions import get_vec
import pickle as pkl
import pandas as pd


# Populate embeddings for the limited vocabulary
def build_vocab_embeddings(vocab, get_vec):
    vocab_embeddings = {}
    for word in vocab:
        vec = get_vec(word)
        if vec is not None:  # Only add the embedding if the word has an embedding
            print(f'{len(vocab_embeddings)}/{len(vocab)}: {word}')
            vocab_embeddings[word] = vec
    return vocab_embeddings


# vocab = {'mistake', 'cow', 'undergraduate', 'acknowledge', 'property', 'turning', 'color', 'playfulness', 'personality',
#          'fine-tune', 'bottle', 'road', 'call', 'core', 'build', 'replace', 'assemble', 'departure', 'blouse',
#          'reorganize', 'slander', 'vigour', 'issue', 'makeup', 'dilate', 'impale', 'pull', 'centre', 'invigorate',
#          'restore', 'achieve', 'adaption', 'self-rule', 'stress', 'garment', 'simple', 'cupboard', 'collect',
#          'sepulchre', 'didactics', 'survive', 'bed', 'demobilisation', 'nonfiction', 'headland', 'associate', 'forbid',
#          'jurist', 'component', 'fireplace', 'deceit', 'affect', 'salary', 'ask', 'commission', 'cry', 'flare-up',
#          'elide', 'submit', 'revolutionary', 'artefact', 'plead', 'endanger', 'drama', 'airplane', 'standardization',
#          'genre', 'psychotherapy', 'rate', 'mountain', 'being', 'family', 'discomfit', 'delete', 'objective', 'cloth',
#          'defy', 'cohesion', 'gown', 'aliveness', 'break', 'warmness', 'postponement', 'stand-in', 'hair', 'deal',
#          'suburb', 'stronghold', 'moment', 'sympathy', 'arouse', 'flavoring', 'counsel', 'remind', 'walk', 'lodging',
#          'quit', 'falseness', 'freebooter', 'future', 'flowing', 'first', 'misidentify', 'rum', 'compare', 'quarters',
#          'enclosing', 'forepart', 'happen', 'part', 'bureau', 'memorize', 'transmitting', 'speech', 'man', 'mountebank',
#          'emphasize', 'element', 'tower', 'dispute', 'broadness', 'distract', 'conservationist', 'honor', 'military',
#          'get', 'coolness', 'revenue', 'framework', 'standardize', 'car', 'unresponsiveness', 'criticize', 'action',
#          'discover', 'sickness', 'unwillingness', 'loss', 'depositary', 'attain', 'authorization', 'bewilder', 'give',
#          'holiday', 'acceptance', 'agenda', 'cut', 'cooling', 'print', 'effect', 'scream', 'missile', 'familiarize',
#          'erode', 'lengthen', 'movie', 'pushing', 'distort', 'affection', 'collection', 'interrogation', 'legislature',
#          'epitome', 'mind', 'oxidization', 'disfavor', 'renew', 'denunciation', 'happening', 'clergyman', 'depression',
#          'business', 'reduction', 'gathering', 'exhort', 'commerce', 'judge', 'name', 'ruler', 'adeptness', 'boredom',
#          'laborer', 'jacket', 'declaration', 'straight', 'choreography', 'result', 'bike', 'penalty', 'promontory',
#          'evolve', 'vicinity', 'food', 'regularise', 'stage', 'justification', 'depository', 'violence', 'period',
#          'retailer', 'rebuff', 'subsidiary', 'closet', 'standing', 'somebody', 'demarcation', 'clause', 'time',
#          'quantity', 'cook', 'fire', 'crew', 'thank', 'compose', 'remark', 'sale', 'vessel', 'theory', 'entree',
#          'cognizance', 'validity', 'indicate', 'plainness', 'understanding', 'manage', 'purchasing', 'specialness',
#          'forget', 'exactitude', 'occur', 'organization', 'apportioning', 'remain', 'option', 'treatment', 'round',
#          'heartlessness', 'dark', 'invigoration', 'character', 'texture', 'opposition', 'fund', 'tobacco', 'proposal',
#          'paper', 'seizing', 'hybridisation', 'exhilaration', 'dejection', 'disappointment', 'stoppage', 'deceive',
#          'touch', 'recording', 'crimp', 'creativeness', 'kidnap', 'rise', 'amazement', 'wander', 'ease', 'lessen',
#          'diameter', 'excitement', 'starter', 'staircase', 'energize', 'photo', 'functionary', 'monarch', 'chain',
#          'lawmaker', 'reassert', 'furniture', 'bring', 'pillaging', 'shower', 'learn', 'ideology', 'evolution',
#          'disaffection', 'quiet', 'office', 'tantalize', 'roast', 'lover', 'vulgarian', 'march', 'prosecute', 'member',
#          'self-sacrifice', 'feeding', 'success', 'favour', 'prognosticate', 'disrupt', 'subjugation', 'train',
#          'suppose', 'relieve', 'develop', 'interest', 'endorse', 'put', 'precariousness', 'fertilizer', 'follow',
#          'election', 'enterprise', 'confound', 'halt', 'permit', 'staff', 'shopping', 'award', 'folk', 'explain',
#          'alarm', 'live', 'publishing', 'predict', 'telecom', 'attract', 'impression', 'depress', 'jet', 'research',
#          'wickedness', 'boost', 'hunting', 'calculate', 'resurgence', 'leading', 'court', 'pass', 'corrupt', 'torment',
#          'premise', 'nobleman', 'dissatisfaction', 'press', 'calibration', 'cigarette', 'categorise', 'current',
#          'requirement', 'wrapping', 'problem', 'destruction', 'talking', 'credit', 'disgrace', 'stabilize', 'division',
#          'quietness', 'advantage', 'cohesiveness', 'attend', 'role', 'profit', 'hope', 'thoughtfulness', 'energy',
#          'sculpture', 'spreading', 'six-shooter', 'sector', 'shorten', 'toughness', 'make', 'safeguard', 'willingness',
#          'hamper', 'mixture', 'stabilise', 'funeral', 'access', 'cocaine', 'exhibit', 'protract', 'spending', 'estate',
#          'money', 'ambitiousness', 'attentiveness', 'favourite', 'end', 'defence', 'plunge', 'set', 'grassland',
#          'throw', 'saying', 'counseling', 'web', 'inhabitation', 'transmit', 'clarity', 'tedium', 'railway',
#          'watercolor', 'leadership', 'evaluation', 'anesthetize', 'humanities', 'sailing', 'colonization', 'clothing',
#          'disguise', 'vitality', 'seeing', 'bacteria', 'punish', 'retaliation', 'confection', 'understand',
#          'directness', 'reword', 'arriviste', 'examine', 'layover', 'wound', 'impotency', 'squelch', 'armour',
#          'accomplish', 'fact', 'repress', 'management', 'alienation', 'instigate', 'colorize', 'perspicacity', 'pet',
#          'palliate', 'station', 'utilisation', 'assigning', 'governance', 'unhinge', 'announce', 'coating',
#          'strengthen', 'deviltry', 'telephone', 'leg', 'shop', 'personation', 'ready', 'conglomerate', 'database',
#          'defendant', 'renounce', 'standardisation', 'acquire', 'lumber', 'agency', 'please', 'panel', 'remainder',
#          'fulfill', 'frighten', 'power', 'bother', 'consequence', 'looking', 'attacker', 'dishonesty', 'artifact',
#          'malady', 'act', 'heroin', 'defeat', 'request', 'temblor', 'factory', 'offspring', 'reference', 'penance',
#          'policeman', 'disc', 'pretending', 'musicality', 'institution', 'humour', 'reliever', 'rage', 'watercolour',
#          'healthiness', 'filling', 'derision', 'crop', 'colonisation', 'fascinate', 'average', 'injustice',
#          'liberation', 'running', 'conditions', 'mural', 'pedal', 'link', 'depart', 'photography', 'irritability',
#          'prolong', 'ordain', 'uncertainty', 'phone', 'provoke', 'danger', 'insist', 'cultivate', 'earthquake',
#          'divulge', 'initiate', 'demand', 'worsen', 'assuage', 'ingress', 'oil', 'stand', 'abandon', 'consumer',
#          'personify', 'crystallization', 'depiction', 'negotiate', 'sentimentality', 'tighten', 'seize', 'infection',
#          'wrongness', 'bewitch', 'weakening', 'purchase', 'cape', 'share', 'deftness', 'selflessness', 'prayer',
#          'withdrawal', 'mesmerize', 'knowing', 'broaden', 'young', 'go', 'deport', 'impermeability', 'right',
#          'assembling', 'conference', 'neighborhood', 'regret', 'value', 'inhabit', 'malefactor', 'cognisance',
#          'surpass', 'classify', 'passion', 'think', 'discourage', 'boy', 'celebration', 'passiveness', 'evidence',
#          'relinquishing', 'wandering', 'fry', 'overcome', 'script', 'see', 'police', 'forego', 'constrict', 'past',
#          'rainfall', 'yell', 'activeness', 'impact', 'mawkishness', 'indication', 'tractability', 'ancestry', 'deepen',
#          'chance', 'analyze', 'landing', 'liquor', 'harass', 'tune', 'oppression', 'giving', 'exclaiming', 'adjust',
#          'dip', 'quash', 'direct'}

vocab = pd.read_csv('OLD/vocabulary_20241008-181731.csv')

vocab_embeddings = build_vocab_embeddings(vocab['vocabulary'], get_vec)
print(f'vocab_embeddings.....DONE!')

with open('vb_600.pkl', 'wb') as file:
    pkl.dump(vocab_embeddings, file)
