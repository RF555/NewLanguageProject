from TransformersUtilityFunctions import get_vec
import pickle as pkl


# Populate embeddings for the limited vocabulary
def build_vocab_embeddings(vocab, get_vec):
    vocab_embeddings = {}
    for word in vocab:
        vec = get_vec(word)
        if vec is not None:  # Only add the embedding if the word has an embedding
            print(f'{len(vocab_embeddings)}/{len(vocab)}: {word}')
            vocab_embeddings[word] = vec
    return vocab_embeddings

with open('vocab_words_formatted.txt', 'r') as file:
    vocab = eval(file.read())
vocab_embeddings = build_vocab_embeddings(vocab, get_vec)
print(f'vocab_embeddings.....DONE!')

with open('vocab_embeddings_dict.pkl', 'wb') as file:
    pkl.dump(vocab_embeddings, file)
