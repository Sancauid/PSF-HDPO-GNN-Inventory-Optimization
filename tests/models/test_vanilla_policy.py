import torch

from hdpo_gnn.models.vanilla import VanillaPolicy


def test_vanilla_policy_forward_pass():
  B = 4
  n_stores = 3
  input_size = 10
  output_size = n_stores + 1

  model = VanillaPolicy(input_size=input_size, output_size=output_size)
  dummy_input = torch.randn(B, input_size)

  out = model(dummy_input)

  assert isinstance(out, dict)
  assert 'stores' in out and 'warehouses' in out
  assert out['stores'].shape == torch.Size([B, n_stores])
  assert out['warehouses'].shape == torch.Size([B, 1])


