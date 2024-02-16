# Grapher, a GNN-based generative language model 

Text sequences can be transformed into a corresponding word graph. Doing so creates an unordered set with an adjacency matrix defining a complete graph. 

In this toy experiment, I use this insight to enable the use of Graph Attentional Networks for text generation. This is done by using complete word graphs to predict the next token in a sequence (or rather, the missing node in the graph). With this task description, applying GAT should be conceptually the same as applying the Transformer. However, as the GAT is permutation invariant to the set of the nodes I completely disregard word-ordering.  Consequently, it is necessary to add something that accounts for this, such as the positional encoding in the Transformer. 

### Status
The model is absolutely terrible. 
