
import sys
sys.path.append('/content/drive/MyDrive/myproject')
from common.trainer import Trainer
from common.optimizer import Adam
from simple_cbow import SimpleCBOW
from common.util import preprocess, create_contexts_target, convert_one_hot

window_size = 1
hidden_size = 5
batch_size = 3
max_epoch = 1000

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
contexts, targets = create_contexts_target(corpus, window_size)
V = len(word_to_id)
x = convert_one_hot(contexts,V)
y = convert_one_hot(targets,V)


Model = SimpleCBOW(V, hidden_size)
Opt = Adam()
Train = Trainer(Model, Opt)
Train.fit(x,y,max_epoch,batch_size)
Train.plot()
