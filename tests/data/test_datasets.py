import torch

from hdpo_gnn.data.datasets import create_synthetic_data_dict


def test_create_synthetic_data_dict():
  dummy_config = {
    'problem_params': {
      'n_stores': 3,
      'n_warehouses': 2,
      'periods': 5,
    },
    'data_params': {
      'n_samples': 4,
    },
  }

  data = create_synthetic_data_dict(dummy_config)

  assert isinstance(data, dict)
  for key in ['inventories', 'demands', 'cost_params', 'lead_times']:
    assert key in data

  B = dummy_config['data_params']['n_samples']
  S = dummy_config['problem_params']['n_stores']
  W = dummy_config['problem_params']['n_warehouses']
  T = dummy_config['problem_params']['periods']

  inv = data['inventories']
  assert isinstance(inv, dict)
  assert 'stores' in inv and 'warehouses' in inv
  assert inv['stores'].shape == torch.Size([B, S])
  assert inv['warehouses'].shape == torch.Size([B, W])

  demands = data['demands']
  assert isinstance(demands, torch.Tensor)
  assert demands.shape == torch.Size([T, B, S])
  assert torch.all(demands >= 0)

  cost_params = data['cost_params']
  assert isinstance(cost_params, dict)
  for key in ['holding_store', 'underage_store', 'holding_warehouse']:
    assert key in cost_params and isinstance(cost_params[key], torch.Tensor)

  lead_times = data['lead_times']
  assert isinstance(lead_times, dict)
  assert lead_times == {'stores': 2, 'warehouses': 3}


