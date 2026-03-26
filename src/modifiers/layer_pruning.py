import torch
import threading
from typing import Iterable, Optional, Dict

class LayerZeroerVLLM:
    def __init__(self, vllm_model):
        self.vllm_model = vllm_model
        self.model_internal = vllm_model.llm_engine.model_executor.driver_worker.worker.model_runner.model.model
        self._saved_weights: Dict[int, Dict[str, int]] = {}
        self._storage_backup: Dict[int, torch.Tensor] = {}
        self._storage_refcount: Dict[int, int] = {}
        self._lock = threading.Lock()

    def _get_layer(self, idx: int):
        n = len(self.model_internal.layers)
        if idx < 0 or idx >= n:
            raise IndexError(f"Layer index {idx} fuori range [0, {n-1}]")
        return self.model_internal.layers[idx]

    def _storage_ptr(self, tensor: torch.Tensor) -> int:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("_storage_ptr expects a torch.Tensor")
        t = tensor.detach()
        try:
            return t.untyped_storage().data_ptr()
        except AttributeError:
            return t.storage().data_ptr()

    def skip_layers(self, layers_to_zero: Iterable[int], verbose=False):
        self.restore_layers()
        with self._lock:
            for idx in layers_to_zero:
                if idx in self._saved_weights:
                    print(f"[LayerZeroerVLLM] Layer {idx} già azzerato, skip.")
                    continue
                layer = self._get_layer(idx)
                backup: Dict[str, int] = {}
                with torch.no_grad():
                    for name, param in layer.named_parameters(recurse=True):
                        if not isinstance(param, torch.nn.Parameter):
                            continue
                       # if not param.requires_grad:
                        #    continue
                        sid = self._storage_ptr(param.data)
                        if sid not in self._storage_backup:
                            self._storage_backup[sid] = param.detach().cpu().clone()
                            self._storage_refcount[sid] = 1
                        else:
                            self._storage_refcount[sid] += 1
                        backup[name] = sid
                        param.data.zero_()
                self._saved_weights[idx] = backup
                if verbose:
                    print(f"[LayerZeroerVLLM] Layer {idx}: pesi azzerati ({len(backup)} parametri salvati, storages uniche: {len(set(backup.values()))})")

    def restore_layers(self, layers_to_restore: Optional[Iterable[int]] = None, verbose=False):
        with self._lock:
            if layers_to_restore is None:
                layers_to_restore = list(self._saved_weights.keys())
            for idx in list(layers_to_restore):
                if idx not in self._saved_weights:
                    print(f"[LayerZeroerVLLM] Layer {idx} non ha backup, skip.")
                    continue
                layer = self._get_layer(idx)
                backup = self._saved_weights.pop(idx)
                restored = 0
                with torch.no_grad():
                    for name, param in layer.named_parameters(recurse=True):
                        if name in backup:
                            sid = backup[name]
                            saved_tensor = self._storage_backup.get(sid, None)
                            if saved_tensor is None:
                                print(f"[LayerZeroerVLLM] Warning: backup storage {sid} mancante per {name}")
                                continue
                            param.data.copy_(saved_tensor.to(param.device))
                            restored += 1
                            self._storage_refcount[sid] -= 1
                            if self._storage_refcount[sid] <= 0:
                                del self._storage_refcount[sid]
                                del self._storage_backup[sid]
                if verbose:
                    print(f"[LayerZeroerVLLM] Layer {idx}: {restored} parametri ripristinati")