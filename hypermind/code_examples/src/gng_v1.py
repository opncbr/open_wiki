from src.dynamic_graph import DynamicGraph
from torch import pow, topk, unique
from torch_scatter import scatter_mean

# The default growing neural gas model as presented in section 2 of http://www.booru.net/download/MasterThesisProj.pdf
class GNG_v1(DynamicGraph):

    def __init__(
        self,
        input_length,
        node_num,
        initial_node_errors,
        initial_node_age,
        lr_alpha,
        lr_beta
    ):
        super().__init__(
            input_length = input_length,
            node_num = node_num,
            initial_node_errors = initial_node_errors,
            initial_node_age = initial_node_age
        )
        self.lr_alpha = lr_alpha
        self.lr_beta = lr_beta

    def _euclidean(self, X):
        return pow(pow(self.weight - X.unsqueeze(dim=1), 2).sum(dim=2), 1/2)

    def _scatter_mean_errors(self, winner_errors, winner_indices):
        mean_winner_error = scatter_mean(winner_errors, winner_indices)
        unique_winner_indices = unique(winner_indices)
        mean_winner_error = mean_winner_error[unique_winner_indices]
        return unique_winner_indices, mean_winner_error
    
    def _scatter_mean_weight_update(self, X, winner_indices, unique_winner_indices):
        mean_winner_input = scatter_mean(X, winner_indices, dim=0)[unique_winner_indices]
        return self.lr_alpha * (mean_winner_input - self.weight[unique_winner_indices])
    
    def forward(self, X):
        # For each code vector, calculate the euclidean distance to the input x_i.
        scores = self._euclidean(X)

        # Locate the best matching code vector s and the second best matching code vector t within w, i.e. the vectors that have the smallest and second smallest euclidean distance with x_i.
        best_matching_scores, best_matching_indices = topk(scores, 2, dim=1, sorted = True, largest = False)
        s_scores, s_indices = best_matching_scores[:, 0], best_matching_indices[:, 0]
        
        # The winner-node s must update its local error variable so we add its euclidean distance to the node's error.
        unique_winner_indices, mean_winner_errors = self._scatter_mean_errors(s_scores, s_indices)
        self.node_errors[unique_winner_indices] += mean_winner_errors

        # Move s closer towards the input that caused s to become the best matching code vector.
        self.weight[unique_winner_indices] += self._scatter_mean_weight_update(X, s_indices, unique_winner_indices)

        # Move all the adjacent code vectors n that are connected to s closer towards the input which caused s to become the best matching code vector.
