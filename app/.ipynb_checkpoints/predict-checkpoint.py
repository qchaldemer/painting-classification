# load module
import json
import predict_utils
from predict_utils import process_image, load_checkpoint, predict
import argparse
import torch

parser = argparse.ArgumentParser(
    description='Parameters for predict')
parser.add_argument('--input', action="store",
                    dest="input", default = '../checkpoint5.pth')
parser.add_argument('--top_k', action="store",
                    dest="top_k", default = '5')
parser.add_argument('--image', action="store",
                    dest="image", default = 'example/100152.jpg')

args = vars(parser.parse_args())

#imputs
image_path = args['image']
checkpoint = args['input']
topk = int(args['top_k'])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# class and indixes
cat_to_name = {0: 'Art Nouveau (Modern)',
                 1: 'Baroque',
                 2: 'Expressionism',
                 3: 'Impressionism',
                 4: 'Post-Impressionism',
                 5: 'Rococo',
                 6: 'Romanticism',
                 7: 'Surrealism',
                 8: 'Symbolism'}

# load the model
model, learning_rate, hidden_units, class_to_idx = load_checkpoint(checkpoint)

# prediction
probs, top_labels = predict(image_path, model, topk)

# print results

res = "\n".join("{} {}".format(x, y) for x, y in zip(probs, top_labels))

print(res)