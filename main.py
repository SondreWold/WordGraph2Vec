from smart_open import open 
import random
from tqdm import tqdm, trange
import argparse
import logging
import numpy as np
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
import pathlib
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from device import device
import math


def parse_args() -> argparse.Namespace:
    """ Return namespace cotnaining CLI arguments. """
    parser = argparse.ArgumentParser(description="CLI for WordGraph2Vec")
    parser.add_argument("--corpus", type=pathlib.Path, default=None, help="Path to the training corpus")
    parser.add_argument("--model_output_path", type=pathlib.Path, default=None, help="Save model to path")
    parser.add_argument("--window", type=int, default=5, help="Context window size")
    parser.add_argument("--layers", type=int, default=3, help="Number of GNN layers")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--hidden_size", type=int, default=300, help="Embedding size")
    args = parser.parse_args()
    return args


class TextGraphDataset(Dataset):
    def __init__(self, corpus: pathlib.Path, window:int):
        self.corpus: pathlib.Path = corpus
        CONTEXT_SIZE = window
        
        self.vocab = set()
        self.data = []
        for line in tqdm(open(self.corpus, 'r')):
            sentence = line.split()
            self.vocab.update(sentence)
            for i in range(CONTEXT_SIZE, len(sentence)):
                context = [sentence[i - (c + 1)] for c in range(CONTEXT_SIZE)]
                target = sentence[i]
                self.data.append((context, target))
        self.word_to_ix = {word: i for i, word in enumerate(self.vocab)}
        self.ix_to_word = {self.word_to_ix[word]: word for word in self.word_to_ix}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        context, target = self.data[idx]
        nodes = torch.tensor([self.word_to_ix[w] for w in context])
        n = nodes.shape[0]
        adj_matrix = torch.ones((n, n)) - torch.eye(n)
        adj_matrix = adj_matrix.nonzero().t().contiguous()
        return Data(nodes, adj_matrix), torch.tensor(self.word_to_ix[target])


class Grapher(torch.nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, layers: int):
        super(Grapher, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.convs = torch.nn.ModuleList(
                [GATConv(hidden_size, hidden_size, 1) for _ in range(layers)]
                )
        self.lins = torch.nn.ModuleList(
                [nn.Linear(hidden_size, hidden_size) for _ in range(layers)]
                )
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, data):
        if self.training:
            batch = data.batch.to(device)
        else:
            batch = None
        nodes = data.x.to(device)
        edges = data.edge_index.to(device)
        x = self.embedding(nodes)
        for gnn, lin in zip(self.convs, self.lins):
            x = gnn(x, edges)
            x = nn.functional.relu(x)
            x = lin(x)
        out = global_mean_pool(x, batch=batch)
        return self.out(out)

    def generate(self, prompt, indexer, inverter, max_new_tokens):
        output = prompt.split()
        for i in range(max_new_tokens):
            nodes = torch.tensor([indexer[w] for w in output[i:]])
            n = nodes.shape[0]
            adj_matrix = torch.ones((n, n)) - torch.eye(n)
            adj_matrix = adj_matrix.nonzero().t().contiguous()
            test_graph = Data(nodes, adj_matrix)
            logits = self(test_graph)
            probs = nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            w = inverter[idx_next.item()]
            output.append(w)    
        return " ".join(output)


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)
    logging.info("WordGraph2Vec")

    args: argparse.Namespace = parse_args()
    logging.info(f"{args.corpus}")
    training_data = TextGraphDataset(args.corpus, window=args.window)
    train_loader = DataLoader(training_data, batch_size=32, drop_last=True)

    criterion = nn.CrossEntropyLoss()
    model = Grapher(len(training_data.vocab), hidden_size=args.hidden_size, layers=args.layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    num_params = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of parameters: {num_params}")
    logging.info(f"Training sample: {training_data.data[3]}")
    device_max_steps = args.epochs * len(train_loader)
    warmup_proportion = 0.016


    def cosine_schedule_with_warmup(optimizer, num_warmup_steps: int, num_training_steps: int, min_factor: float):
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            lr = max(min_factor, min_factor + (1 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * progress)))
            return lr
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scheduler = cosine_schedule_with_warmup(optimizer, int(device_max_steps * warmup_proportion), device_max_steps, 0.1)


    for epoch in range(args.epochs):
        total_loss = 0
        model.train()
        for graph, target in tqdm(train_loader):
            target = target.to(device)
            optimizer.zero_grad()
            output = model(graph)
            loss = criterion(output, target)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
        logger.info('Epoch %d, loss %f' % (epoch, total_loss))

        model.eval()
        with torch.no_grad():
            prompt = "The capital of Norway is a city that has a lot to do , such as"
            output = model.generate(prompt, training_data.word_to_ix, training_data.ix_to_word, 20)
        logging.info(f"Prompt: {prompt}: Response: {output}")

    if args.model_output_path:
        torch.save({
            'model_state_dict': model.state_dict(),
            'vocab': len(training_data.vocab),
            'indexer': training_data.word_to_ix,
            'inverter': training_data.ix_to_word
            }, args.model_output_path)
