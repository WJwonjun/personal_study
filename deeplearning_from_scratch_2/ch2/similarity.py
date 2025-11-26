import sys
sys.path.append('/content/myproject')
from common.util import preprocess, create_co_matrix, cos_similarity

text = 'you say goodbye and i say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
co_matrix = create_co_matrix(corpus,len(word_to_id))
similarity = cos_similarity(co_matrix[word_to_id['you']],co_matrix[word_to_id['hello']])
print(similarity)
