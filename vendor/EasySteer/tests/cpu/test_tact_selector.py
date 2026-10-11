"""GPU singleton fast path preserves native phase/slot/padding semantics."""
from types import SimpleNamespace

import numpy as np
import torch

from vllm.v1.worker.gpu.steer_vector_utils import _fill_single_generated_token_masks


def test_singleton_selector_never_reads_tokens_on_host_and_excludes_prompt():
    masks=torch.zeros(2,8)
    first=SimpleNamespace(graph_mask=masks[0]);second=SimpleNamespace(graph_mask=masks[1])
    def clause(token=2, **extra):
        return SimpleNamespace(vectors=[SimpleNamespace(algorithm="tact_cap",
            apply_spec={"generation_tokens":[token],**extra})])
    entries={0:(1,clause(),[first]),1:(2,clause(),[second])}
    geo=SimpleNamespace(query_start_loc_cpu=np.array([0,3,5]),
                        num_computed=torch.tensor([1,0]),num_prompt=torch.tensor([2,1]),
                        token_ids=torch.tensor([2,2,2,2,2]))
    geo.token_ids_cpu=lambda: (_ for _ in ()).throw(AssertionError("Unexpected D2H"))
    manager=SimpleNamespace(graph_masks_buf=masks)
    assert _fill_single_generated_token_masks(manager,geo,entries,np.array([0,1]),
                                              np.array([1,1,1,2,2]),5)
    assert torch.equal(masks[0],torch.tensor([0.,1,1,0,0,0,0,0]))
    assert torch.equal(masks[1],torch.tensor([0.,0,0,0,1,0,0,0]))
    entries[1]=(2,clause(generation="all"),[second])
    assert not _fill_single_generated_token_masks(manager,geo,entries,np.array([0,1]),
                                                  np.array([1,1,1,2,2]),5)
