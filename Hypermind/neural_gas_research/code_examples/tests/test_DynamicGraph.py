from src.DynamicGraph import DynamicGraph
import torch

def test_createGraphDefault():
    dg = DynamicGraph(input_length = 10)
    assert list(dg.tensor.shape) == [2,10]
    assert list(dg.node_errors.shape) == [2]
    assert dg.edges.sum() == 0
    assert dg.edge_ages.sum() == 0
    
def test_createGraphCustom():
    dg = DynamicGraph(input_length = 10, node_num = 15)
    assert list(dg.tensor.shape) == [15,10]
    assert list(dg.node_errors.shape) == [15]
    assert dg.edges.sum() == 0
    assert dg.edge_ages.sum() == 0
    
def test_add_nodes():
    dg = DynamicGraph(input_length = 10, node_num = 7)
    dg.add_nodes(new_tensor= torch.rand(4,10), 
                 new_errors = 0
                )
    assert list(dg.tensor.shape) == [11,10]
    assert list(dg.node_errors.shape) == [11]
    assert dg.edges.sum() == 0
    assert dg.edge_ages.sum() == 0
    
def test_remove_nodes():
    node_num = 7
    dg = DynamicGraph(input_length = 10, node_num = node_num)
    indices2remove = [1,2,5]
    left_indices = sorted(list(set(list(range(node_num))) - set(indices2remove)))
    cloned_tensor = dg.tensor[left_indices,:].detach().clone()
    dg.remove_nodes(node_indices = torch.tensor(indices2remove))
    assert torch.all(torch.eq(cloned_tensor, dg.tensor)).item()
    assert list(dg.node_errors.shape) == [4]
    assert dg.edges.sum() == 0
    assert dg.edge_ages.sum() == 0
    
def test_empty_graph():
    dg = DynamicGraph(input_length = 10, node_num = 0)
    assert list(dg.tensor.shape) == [0,10]

def test_add_then_remove_nodes():
    dg = DynamicGraph(input_length = 10, node_num = 0)
    node_num = 7
    dg.add_nodes(new_tensor= torch.rand(node_num,10), 
                 new_errors = 0
                )
    indices2remove = [1,2,5,6]
    left_indices = sorted(list(set(list(range(node_num))) - set(indices2remove)))
    cloned_tensor = dg.tensor[left_indices,:].detach().clone()
    dg.remove_nodes(node_indices = torch.tensor(indices2remove))
    assert torch.all(torch.eq(cloned_tensor, dg.tensor)).item()
    assert list(dg.node_errors.shape) == [3]
    assert dg.edges.sum() == 0
    assert dg.edge_ages.sum() == 0
    
def test_get_nodes_num():
    dg = DynamicGraph(input_length = 10, node_num = 0)
    num = dg.get_nodes_num()
    assert num == 0
    node_num = 500
    dg.add_nodes(new_tensor= torch.rand(node_num,10), 
                 new_errors = 0
                )
    num = dg.get_nodes_num()
    assert num == node_num

def test_add_edges():
    dg = DynamicGraph(input_length = 10, node_num = 7)
    dg.add_edges(U = torch.tensor([1,1,0,1,1,3,4,6]), 
                 V = torch.tensor([0,1,0,0,1,4,2,5]), 
                 ages = 1
                )
    res = torch.tensor([[0, 0, 1, 2, 3, 5],
                        [0, 1, 1, 4, 4, 6]])
    assert torch.all(torch.eq(dg.edges, res)).item()
    assert torch.all(torch.eq(dg.edge_ages, torch.ones(6))).item()
    

def test_remove_edges():
    dg = DynamicGraph(input_length = 10, node_num = 7)
    dg.add_edges(U = torch.tensor([1,1,0,1,1,3,4,6]), 
                 V = torch.tensor([0,1,0,0,1,4,2,5]), 
                 ages = 1
                )
    dg.remove_edges(U = torch.tensor([0,2,5]), 
                    V = torch.tensor([0,4,6])
                   )
    res = torch.tensor([[0,1,3],[1,1,4]])
    assert torch.all(torch.eq(dg.edges, res)).item()
    
    
def test_add_to_node_errors():
    dg = DynamicGraph(input_length = 10, node_num = 7)
    dg.update_node_errors(node_indices = torch.tensor([2,3,6,6]), 
                          node_errors = torch.tensor([1,2,6,3]),
                          operation = 'add'
                         )
    res = torch.tensor([0.,0.,1.,2.,0.,0.,4.])
    assert torch.all(torch.eq(dg.node_errors, res)).item()
    

def test_reduce_from_node_errors():
    dg = DynamicGraph(input_length = 10, node_num = 7)
    dg.update_node_errors(node_indices = torch.tensor([2,3,6,6]), 
                             node_errors = torch.tensor([1,2,6,3]),
                             operation = 'sub'
                          )
    res = torch.tensor([0.,0.,-1.,-2.,0.,0.,-4.])
    assert torch.all(torch.eq(dg.node_errors, res)).item()


def test_update_edge_ages_1():
    dg = DynamicGraph(input_length = 10, node_num = 7)
    dg.add_edges(U = torch.tensor([1,1,0,1,1,3,4,6]), 
                 V = torch.tensor([0,1,0,0,1,4,2,5]), 
                 ages = 1
                )
    assert torch.all(torch.eq(dg.edge_ages, torch.ones(6))).item()
    res = torch.tensor([[0, 0, 1, 2, 3, 5],
                        [0, 1, 1, 4, 4, 6]])
    dg.update_edge_ages(U = res[0,:],
                        V = res[1,:],
                        ages = torch.tensor([1,2,3,4,5,6]),
                        operation = "add"
                        ) 
    expected_ages = torch.tensor([2,3,4,5,6,7])
    assert torch.all(torch.eq(dg.edge_ages, expected_ages)).item()


def test_update_edge_ages_2():
    dg = DynamicGraph(input_length = 10, node_num = 7)
    dg.add_edges(U = torch.tensor([1,1,0,1,1,3,4,6]), 
                 V = torch.tensor([0,1,0,0,1,4,2,5]), 
                 ages = 1
                )
    assert torch.all(torch.eq(dg.edge_ages, torch.ones(6))).item()
    res = torch.tensor([[0, 0, 1, 2, 3, 5, 0],
                        [0, 1, 1, 4, 4, 6, 1]])
    dg.update_edge_ages(U = res[0,:],
                        V = res[1,:],
                        ages = torch.tensor([1,2,3,4,5,6,1]),
                        operation = "add"
                        ) 
    expected_ages = torch.tensor([2., 2.5, 4., 5., 6., 7.])
    assert torch.all(torch.eq(dg.edge_ages, expected_ages)).item()