# src/hdpo_gnn/engine/pipeline.py
import torch
import torch.nn as nn
from typing import Dict, List

class DifferentiablePipeline(nn.Module):
    """
    Manages multiple in-transit inventory queues, one for each unique lead time.
    """
    def __init__(self, batch_size: int, num_locations: int, unique_lead_times: List[int]):
        super().__init__()
        
        self.queues: Dict[int, torch.Tensor] = {}
        self.lead_time_map = {lt: i for i, lt in enumerate(unique_lead_times)}

        for lt in unique_lead_times:
            if lt > 0:
                # Register each queue tensor so it's moved to the correct device
                queue_shape = (batch_size, num_locations, lt)
                self.register_buffer(f"queue_{lt}", torch.zeros(queue_shape))
                self.queues[lt] = getattr(self, f"queue_{lt}")

    def get_arrivals(self) -> torch.Tensor:
        """Sums arrivals from all pipelines that are delivering in this step."""
        total_arrivals = None
        for buffer in self.buffers():
            if buffer.dim() == 3 and buffer.shape[2] > 0:
                arrivals_from_queue = buffer[:, :, 0]
                if total_arrivals is None:
                    total_arrivals = arrivals_from_queue
                else:
                    total_arrivals += arrivals_from_queue
        
        if total_arrivals is None:
            first_buffer = next(self.buffers(), None)
            if first_buffer is not None:
                return torch.zeros(first_buffer.shape[0:2], device=first_buffer.device, dtype=first_buffer.dtype)
            else:
                return torch.zeros(1, 1)
            
        return total_arrivals

    def place_orders(self, orders: torch.Tensor, lead_times: torch.Tensor):
        """
        Places orders into the appropriate queue based on their lead time.
        `orders`: Tensor of shape [B, N]
        `lead_times`: Tensor of shape [N]
        """
        for name, buffer in self.named_buffers():
            if name.startswith("queue_") and buffer.dim() == 3:
                lt = int(name.split("_")[1])
                if lt > 0:
                    mask = (lead_times == lt)
                    orders_for_this_lt = orders * mask.unsqueeze(0)
                    buffer[:, :, -1] += orders_for_this_lt

    def advance_step(self):
        """Advances time by one step for all queues."""
        for name, buffer in self.named_buffers():
            if name.startswith("queue_") and buffer.dim() == 3:
                lt = int(name.split("_")[1])
                if lt > 0:
                    buffer.data.copy_(torch.roll(buffer, shifts=-1, dims=2))
                    buffer[:, :, -1] = 0