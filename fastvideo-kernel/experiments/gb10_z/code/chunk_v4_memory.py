"""Read-only phase/allocation observers; no module release or training arithmetic changes."""
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
import json
import weakref


class MemoryObserver:
    def __init__(self, output, history=False):
        self.output, self.history = Path(output), history
        self.rows, self.peaks = [], []
        self.student = None
        self.method = None
        self.encoder = None
        self.encoder_parameters = []
        self.encoder_outputs = []
        self.phase, self.step = 'setup', 0
        self.stack = ExitStack()

    def save(self):
        (self.output / 'memory.json').write_text(json.dumps({'phases': self.rows, 'peak_snapshots': self.peaks}, indent=2) + '\n')

    def inventory(self, temporaries=()):
        from peak_memory import inventory, local
        import torch
        addresses = {}
        method = self.method() if self.method else None
        student = self.student() if self.student else None
        if method is not None:
            addresses.update(inventory(list(method.student.transformer.parameters()), method.get_optimizers(self.step)[0]))
        def add(category, tensors):
            for tensor in tensors:
                if tensor is None or not isinstance(tensor, torch.Tensor):
                    continue
                tensor = local(tensor)
                if tensor.device.type == 'cuda':
                    s = tensor.untyped_storage()
                    addresses.setdefault(s.data_ptr(), {'category': category, 'bytes': s.nbytes()})
        if student is not None:
            add('parameters_master_fp32', student.transformer.parameters())
            add('transformer_buffers', student.transformer.buffers())
            add('negative_condition_cache', [getattr(student, 'negative_prompt_embeds', None),
                                             getattr(student, 'negative_prompt_attention_mask', None)])
            vae = getattr(student, 'vae', None)
            if vae is not None:
                add('vae', vae.parameters()); add('vae', vae.buffers())
        add('text_encoder_parameters_buffers', (ref() for ref in self.encoder_parameters))
        add('text_encoder_outputs_not_cache', (ref() for ref in self.encoder_outputs))
        add('adam_sqrt_temporaries', temporaries)
        return addresses

    def mark(self, phase):
        import torch
        self.phase = phase
        inv = self.inventory()
        sizes = {}
        for value in inv.values():
            sizes[value['category']] = sizes.get(value['category'], 0) + value['bytes']
        self.rows.append({'phase': phase, 'step': self.step, 'allocated': torch.cuda.memory_allocated(),
                          'reserved': torch.cuda.memory_reserved(), 'peak_allocated': torch.cuda.max_memory_allocated(),
                          'peak_reserved': torch.cuda.max_memory_reserved(), 'known_storage_bytes': sizes,
                          'text_encoder_object_alive': bool(self.encoder and self.encoder() is not None),
                          'live_encoder_tensor_refs': sum(ref() is not None for ref in self.encoder_parameters)})
        self.save()

    def peak(self, tensors):
        import torch
        if not self.history or self.phase != 'optimizer':
            return
        snapshot = torch.cuda.memory._snapshot()
        addresses = self.inventory(tensors)
        categories, origins = {}, {}
        for segment in snapshot['segments']:
            for block in segment['blocks']:
                if block['state'] != 'active_allocated':
                    continue
                item = addresses.get(block['address'])
                if item:
                    assert block['size'] >= item['bytes']
                    for key, size in ((item['category'], item['bytes']), ('allocator_rounding', block['size'] - item['bytes'])):
                        categories[key] = categories.get(key, 0) + size
                else:
                    categories['other'] = categories.get('other', 0) + block['size']
                    frames = block.get('frames', [])
                    frame = frames[0] if frames else {}
                    key = '{}:{}:{}'.format(Path(frame.get('filename', 'unknown')).name, frame.get('line', 0), frame.get('name', 'unknown'))
                    origins[key] = origins.get(key, 0) + block['size']
        allocated = torch.cuda.memory_allocated()
        assert sum(categories.values()) == allocated
        row = {'step': self.step, 'phase': 'inside_optimizer_foreach_sqrt', 'allocated': allocated,
               'peak_allocated': torch.cuda.max_memory_allocated(), 'category_bytes': categories,
               'other_origins': origins, 'is_recorded_peak': allocated == torch.cuda.max_memory_allocated()}
        self.peaks.append(row)
        torch.cuda.memory._dump_snapshot(str(self.output / f'peak-step-{self.step}.pickle'))
        self.save()

    def __enter__(self):
        import torch
        from fastvideo.train.models.wan.wan import WanModel
        from fastvideo.train.models.wan import wan
        from fastvideo.models.loader.component_loader import TextEncoderLoader
        from fastvideo.train.methods.fine_tuning.finetune import FineTuneMethod
        from fastvideo.train.methods.base import TrainingMethod
        from fastvideo.train.callbacks.grad_clip import GradNormClipCallback
        if self.history:
            torch.cuda.memory._record_memory_history(enabled='all', context='all', stacks='python', max_entries=200000)
        def wrap(cls, name, before=None, after=None):
            original = getattr(cls, name)
            def observed(obj, *args, **kwargs):
                if before: before(obj, args, kwargs)
                result = original(obj, *args, **kwargs)
                if after: after(obj, result)
                return result
            self.stack.enter_context(patch.object(cls, name, observed))
        def model_loaded(obj, result):
            self.student = weakref.ref(obj); self.mark('model_loaded')
        def encoder_loaded(obj, result):
            self.encoder = weakref.ref(result)
            self.encoder_parameters = [weakref.ref(p) for p in (*result.parameters(), *result.buffers())]
            self.mark('text_encoder_loaded')
        wrap(WanModel, '__init__', after=model_loaded)
        wrap(WanModel, 'init_preprocessors', after=lambda o, r: self.mark('preprocessors_loaded'))
        wrap(TextEncoderLoader, 'load', after=encoder_loaded)
        original_encode = wan.encode_negative_prompt
        def encode(*args, **kwargs):
            result = original_encode(*args, **kwargs)
            self.encoder_outputs = [weakref.ref(t) for t in result]
            self.mark('negative_encoding_returned')
            return result
        self.stack.enter_context(patch.object(wan, 'encode_negative_prompt', encode))
        wrap(TrainingMethod, 'on_train_start', after=lambda o, r: self.mark('negative_condition_cached'))
        def forward_start(obj, args, kwargs):
            self.method = weakref.ref(obj)
            self.step = args[1] if len(args) > 1 else kwargs['iteration']
            self.mark('before_forward')
        wrap(FineTuneMethod, 'single_train_step', before=forward_start, after=lambda o, r: self.mark('forward_end'))
        wrap(FineTuneMethod, 'backward', before=lambda o, a, k: setattr(self, 'phase', 'backward'), after=lambda o, r: self.mark('backward_end'))
        wrap(GradNormClipCallback, 'on_before_optimizer_step', after=lambda o, r: self.mark('clip_end'))
        wrap(TrainingMethod, 'optimizers_schedulers_step', before=lambda o, a, k: setattr(self, 'phase', 'optimizer'), after=lambda o, r: self.mark('optimizer_end'))
        wrap(TrainingMethod, 'optimizers_zero_grad', after=lambda o, r: self.mark('zero_grad_end'))
        if self.history:
            original_sqrt = torch._foreach_sqrt
            def sqrt(*args, **kwargs):
                result = original_sqrt(*args, **kwargs)
                self.peak(result)
                return result
            self.stack.enter_context(patch.object(torch, '_foreach_sqrt', sqrt))
        return self

    def __exit__(self, *args):
        import torch
        self.stack.close()
        if self.history:
            torch.cuda.memory._record_memory_history(enabled=None)
        self.save()
