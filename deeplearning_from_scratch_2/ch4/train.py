
import sys
sys.path.append('/content/drive/MyDrive/myproject')
from common import config
import pickle
from common.trainer import Trainer
from common.optimizer import Adam
from cbow import CBOW
from common.util import create_contexts_target, to_cpu, to_gpu
from dataset import ptb

window_size = 5
hidden_size = 100
batch_size = 100
max_epoch = 10

corpus, word_to_id, id_to_word = ptb.load_data('train')
contexts, targets = create_contexts_target(corpus, window_size)
V= len(word_to_id)

if config.GPU:
  contexts,targets = to_gpu(contexts), to_gpu(targets)

Model = CBOW(V,hidden_size,window_size,corpus)
Opt = Adam()
Train = Trainer(Model,Opt)

Train.fit(contexts,targets,max_epoch,batch_size)
Train.plot()

word_vecs = Model.word_vecs
if config.GPU:
  word_vecs = to_gpu(word_vecs)

params = {}
params['word_vecs'] = word_vecs.astype(np.float16)
params['word_to_id'] = word_to_id
params['id_to_word'] = id_to_word
pkl_file = 'cbow_params.pkl'
with open('/content/drive/MyDrive/myproject/data.pkl','wb') as f:
  pickle.dump(params,f, -1)
