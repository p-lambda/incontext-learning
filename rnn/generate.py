###############################################################################
# Language Modeling on Wikitext-2
#
# This file generates new sentences sampled from the language model
#
###############################################################################

import argparse
import torch
import data
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import itertools


def generate(model, input, length):
    batch_size = input.shape[1]
    input_length = input.shape[0]
    # get the final hidden state
    hidden = model.init_hidden(batch_size)

    # read history
    if len(input) > 1:
        hist_outputs, hidden = model.recurrent_forward(input[:-1], hidden)
    else:
        hist_outputs = None

    # generate
    curr_input = input[-1].unsqueeze(0).clone()
    gen_outputs = []
    for i in range(length):
        output, hidden = model.recurrent_forward(curr_input, hidden)
        gen_outputs.append(output)

    if hist_outputs is None:
        all_outputs = torch.cat(gen_outputs, dim=0)
    else:
        all_outputs = torch.cat([hist_outputs] + gen_outputs, dim=0)
    all_outputs = model.decode(all_outputs, softmax=False)
    gen_outputs = all_outputs[-length:]

    generation = []
    for output in gen_outputs:
        if args.temperature == 0.0:
            word_idx = torch.argmax(output, dim=1)
        else:
            word_weights = output.div(args.temperature).exp()
            word_idx = torch.multinomial(word_weights, 1)[0]
        curr_input = word_idx.unsqueeze(0)
        generation.append(word_idx.cpu())

    return torch.stack(generation).type(torch.int64).T


def tokenize(text, dictionary):
    idss = []

    words = text.split()
    ids = [dictionary.word2idx[word] for word in words]
    return torch.tensor(ids).type(torch.int64)


parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 Language Model')

# Model parameters.
parser.add_argument('--data', type=str, default='./data/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='./model.pt',
                    help='model checkpoint to use')
parser.add_argument('--words', type=int, default=1,
                    help='number of words to generate')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--temperature', type=float, default=0.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')
parser.add_argument('--results_name', type=str, default='rnn_in_context_results.tsv',
                    help='output file for in context results')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

model_dir = Path(args.checkpoint).parent
with open(args.checkpoint, 'rb') as f:
    model = torch.load(f).to(device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"NUMBER OF PARAMETERS: {count_parameters(model)}")

model.eval()

with open(Path(args.data) / 'rnn_dictionary.pkl', 'rb') as f:
    dictionary = pickle.load(f)
ntokens = len(dictionary)

# read prompts
data_dir = Path(args.data)
paths = [
    [
    [
        data_dir / f'id_prompts_randomsample_{prompt_length}.json',
        data_dir / f'ood_prompts_randomsample_{prompt_length}.json'
    ]]
    for prompt_length in [2, 3, 5, 8, 10]
]
paths = list(itertools.chain.from_iterable(paths))
results_name = args.results_name
results_names = [[Path(results_name).stem + f'_{prompt_length}.tsv', Path(results_name).stem + f'_randomsample_{prompt_length}.tsv'] for prompt_length in [2,3,5,8,10]]
results_names = list(itertools.chain.from_iterable(results_names))

results = []

with torch.no_grad():
    for path_ls, curr_results_name in zip(paths, results_names):
        for path in path_ls:
            if not path.exists():
                continue
            df = pd.read_json(path, lines=True)

            num_examples_list = df['n_examples'].unique()
            for num_examples in num_examples_list:
                df_n = df[df['n_examples'] == num_examples]

                ex_len = len(tokenize(df_n.iloc[0]['text'], dictionary))
                if ex_len > 1024:
                    continue

                batch_size = (4 * 1024) // ex_len
                num_batches = len(df_n) // batch_size
                if len(df_n) % batch_size != 0:
                    num_batches += 1

                delimiter_idx = dictionary.word2idx["/"]
                preds = []
                labels = []
                for i in range(num_batches):
                    batch = df_n['text'].iloc[i*batch_size: (i+1)*batch_size]
                    batch_labels = df_n['label'].iloc[i*batch_size: (i+1)*batch_size]
                    batch = torch.stack([tokenize(b, dictionary) for b in batch.tolist()]).cuda()

                    if len(batch.shape) == 1:
                        batch = batch.unsqueeze(1)
                    batch = batch.T
                    length = args.words
                    out = generate(model, batch, length)

                    out = out.detach().cpu().numpy()
                    pred = out[:, 0]
                    preds += list(pred)
                    label = [tokenize(bl, dictionary)[0] for bl in batch_labels.tolist()]
                    labels += label

                preds = np.asarray(preds)
                labels = np.asarray(labels)
                acc = np.mean(preds ==labels)
                print(f"PATH: {path}, NUM EXAMPLES: {num_examples}, ACC: {acc}")
                results.append({'path': path, 'num_examples': num_examples, 'acc': acc})

        df = pd.DataFrame(results)
        df.to_csv(model_dir / curr_results_name, sep='\t')
        print(df)
        results = []
