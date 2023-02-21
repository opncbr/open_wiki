from src.dynamic_graph import DynamicGraph
from torch import pow, topk

# The default growing neural gas model as presented in section 2 of http://www.booru.net/download/MasterThesisProj.pdf
class GNG_v1(DynamicGraph):

    def __init__(
        self,
        input_length,
        node_num,
        initial_node_errors,
        initial_node_age,
        lr
    ):
        super().__init__(
            input_length = input_length,
            node_num = node_num,
            initial_node_errors = initial_node_errors,
            initial_node_age = initial_node_age
        )
        self.lr = lr

    def _euclidean(self, X):
        return pow(pow(self.w - X.unsqueeze(dim=1), 2).sum(dim=2), 1/2)
    
    def forward(self, X):
        # For each code vector, calculate the euclidean distance to the input $x_i$.
        scores = self._euclidean(X)

        # Locate the best matching code vector $k_p$ and the second best matching code vector $k_q$ within $w$, i.e. that have the smallest and second smalles euclidean distance with $x_i$.
        best_matching_code_vectors, best_matching_indices = topk(scores, 2, dim=1, sorted = True, largest = False)
