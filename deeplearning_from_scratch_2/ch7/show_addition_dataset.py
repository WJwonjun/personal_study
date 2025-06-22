import sys
sys.path.append('/content/drive/MyDrive/myproject')
from dataset import sequence

(x_train, t_train), (x_test, t_test) = sequence.load_data('addition.txt',seed=1984)
char_to_id, id_to_char = sequence.get_vocab()

print(x_train.shape,t_train.shape)
print(x_test.shape, t_test.shape)
