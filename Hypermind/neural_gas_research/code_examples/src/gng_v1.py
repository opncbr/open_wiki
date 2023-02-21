from src.DynamicGraph import DynamicGraph

# The default growing neural gas model as presented in section 2 of http://www.booru.net/download/MasterThesisProj.pdf
class GNGV1(DynamicGraph):

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

    def _equ
    
    def forward(self, X):
