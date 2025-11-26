
import sys
sys.path.append('/content/drive/MyDrive/myproject')
from common.util import most_similar
import pickle

pkl_file = 'cbow_params.pkl'
with open('/content/drive/MyDrive/myproject/data.pkl','rb') as f:
  params = pickle.road(f)
  word_vecs = params['word_vecs']
  word_to_id = params['word_to_id']
  id_to_word = params['id_to_word']

querys = ['you','year','car','toyota']
for query in querys :
  most_similar(query, word_to_id, id_to_word, word_vecs,top=5)

