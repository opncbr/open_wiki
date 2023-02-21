from torch import Tensor, tensor, transpose, cat, arange, sort, is_floating_point, ones, float64, DoubleTensor
from torch.nn import Module, Parameter
from torch_scatter import scatter_mean
from torch_sparse import coalesce
from typing import Optional, Union, List
    

class DynamicGraph(Module):
    
    def __init__(self, 
                 input_length: int,
                 node_num: Optional[int] = 2,
                 initial_node_errors: Optional[int] = 0.,
                 initial_node_age: Optional[float] = 0.
                ):
        super(DynamicGraph, self).__init__()
        self.weight = Parameter(Tensor(node_num, input_length), requires_grad = False)
        self.node_errors = ones(node_num, dtype = float64, requires_grad = False) * initial_node_errors
        self.edge_ages = ones(node_num, dtype = float64, requires_grad = False) * initial_node_age
        self.edges = tensor([])
        
    def get_nodes_num(self):
        return self.weight.shape[0]
    
    def get_edges(self):
        U = self.edges[0,:]
        V = self.edges[1,:]
        values = self.edge_ages
        return U, V, values
    
    def add_nodes(self, 
                  new_tensor: Tensor, 
                  new_errors: Union[Tensor, int]
                 ):
        assert type(new_tensor) == Tensor, "invalid type for new_keys"
        if len(new_tensor.size()) == 1:
            new_tensor = new_tensor.unsqueeze(0)
        self.weight = Parameter(data = cat((self.weight, new_tensor), dim=0))
        if type(new_errors) == int:
            new_errors = ones(new_tensor.size(0)) * new_errors
        self.node_errors = cat((self.node_errors, new_errors), dim=0)
    
    def add_edges(self, 
                  U: Tensor, 
                  V: Tensor, 
                  ages: Union[Tensor, int], 
                  reduce_op: Optional[str] = 'mean'
                 ):
        assert (U.max() <= len(self.weight)-1) and (V.max() <= len(self.weight)-1), "edge index exceeds node count"
        new_e = sort(cat((U.unsqueeze(0),V.unsqueeze(0)), dim=0), dim=0).values
        if type(ages) == int:
            ages = ones(new_e.size(1)) * ages
        assert ages.size(0) == new_e.size(1)
        if self.edges.shape[0] == 0:
            m = int(new_e[0].max()) + 1
            n = int(new_e[1].max()) + 1
            self.edges, self.edge_ages = coalesce(new_e, ages, m=m, n=n, op = reduce_op)
        else:
            concatinated_e = cat((self.edges, new_e),dim=1)
            concatinated_v = cat((self.edge_ages, ages),dim=0)
            m = int(concatinated_e[0].max()) + 1
            n = int(concatinated_e[1].max()) + 1
            self.edges, self.edge_ages = coalesce(concatinated_e, concatinated_v, m=m, n=n, op = reduce_op)
    
    def remove_edges(self, 
                     U: Tensor, 
                     V: Tensor
                    ):
        N = self.get_nodes_num()
        E = self.edges.T
        Erem = sort(cat((U.unsqueeze(0), V.unsqueeze(0)), dim=0), dim=0).values.T
        mask = E.unsqueeze(1) == Erem
        mask = mask.all(-1)
        non_repeat_mask = ~mask.any(-1)
        new_v = self.edge_ages[non_repeat_mask]
        new_e = self.edges[:,non_repeat_mask]
        m = int(new_e[0].max()) + 1
        n = int(new_e[1].max()) + 1
        self.edges, self.edge_ages = coalesce(new_e, new_v, m=m, n=n)
        
    def remove_nodes(self, 
                     node_indices: Union[Tensor, List]
                    ):
        if type(node_indices) == list:
            node_indices = tensor(node_indices)
        node_mask = (arange(self.weight.size(0)).unsqueeze(1) == node_indices).sum(1) == 0
        self.weight = Parameter(self.weight[node_mask,:])
        self.node_errors = self.node_errors[node_mask]
        if self.edges.shape[0] != 0:
            edge_mask = self.edges.unsqueeze(2) == node_indices
            edge_mask = transpose(edge_mask, 1, 2).sum(1).sum(0) < 1
            edge_updates = self.edges.unsqueeze(2) > node_indices
            edge_updates = transpose(edge_updates, 1,2).sum(1)
            self.edges = self.edges[:,edge_mask]
            self.edge_ages = self.edge_ages[edge_mask]
            edge_updates = edge_updates[:,edge_mask]
            self.edges -= edge_updates
    
    def update_node_errors(self, 
                           node_indices: Tensor, 
                           node_errors: Tensor, 
                           operation: str
                          ):
        updated_errors = scatter_mean(node_errors, node_indices)
        if operation == "add":
            self.node_errors += updated_errors
        elif operation == "sub":
            self.node_errors -= updated_errors
        else:
            raise ValueError('operation not supported. Must be either add or sub')
        
    
    def update_edge_ages(self,
                         U: Tensor,
                         V: Tensor,
                         ages: Tensor,
                         operation: str
                        ):
        if not is_floating_point(ages):
            ages = ages.type(DoubleTensor) 
        new_e = sort(cat((U.unsqueeze(0),V.unsqueeze(0)), dim=0), dim=0).values
        m = int(new_e[0].max()) + 1
        n = int(new_e[1].max()) + 1
        e, new_ages = coalesce(new_e, ages, m=m, n=n, op = 'mean')
        E = self.edges.T
        mask = E.unsqueeze(1) == e.T
        mask = mask.all(-1).int().sum(1).bool()
        if operation == "add":
            self.edge_ages[mask] += new_ages
        elif operation == "sub":
            self.edge_ages[mask] -= new_ages
        elif operation == "set":
            self.edges[mask] = new_ages
        else: 
            raise ValueError('operation not supported. Must be either add, sub, or set')