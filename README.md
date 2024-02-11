# Grapher, a GNN-based generative language model 

Text sequences can be transformed into a corresponding word graph. Doing so creates an unordered set with an adjacency matrix defining a complete graph. In this toy experiment, I use this exact approach to enable the use of Graph Attentional Networks for text generation. Using complete word graphs to predict the next token in a sequence (or rather, the missing node in the graph) completely disregards word-ordering, though, so adding something that accounts for this, such as the positional encoding in the Transformer, is going to be necessary.
