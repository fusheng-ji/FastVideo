"""Inspect negative encoder lifetime in a fresh process; run under both GPU locks."""
import argparse
import collections
import gc
import json
import os
from pathlib import Path
import weakref

import torch
from transformers import AutoTokenizer
from fastvideo.forward_context import set_forward_context
from fastvideo.models.loader.component_loader import TextEncoderLoader
from fastvideo.train.utils.config import load_run_config
from fastvideo.train.utils.moduleloader import make_inference_args
from fastvideo.utils import maybe_download_model
from fastvideo.api.sampling_param import SamplingParam


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--disable-gc', action='store_true')
    parser.add_argument('--debug-cycles', action='store_true')
    args = parser.parse_args()
    out = Path(args.output)
    if out.exists():
        raise RuntimeError('Output already exists')
    out.mkdir(parents=True)
    report = {'status': 'started', 'disable_gc': args.disable_gc, 'debug_cycles': args.debug_cycles, 'checkpoints': []}
    def save():
        (out / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    save()
    tc = load_run_config(args.config).training
    pipeline = tc.pipeline_config
    ec = pipeline.text_encoder_configs[0]
    post = pipeline.postprocess_text_funcs[0]
    preprocess = getattr(pipeline, 'preprocess_text_funcs', None)
    preprocess = preprocess[0] if preprocess is not None else None
    model_path = maybe_download_model(tc.model_path)
    prompt = SamplingParam.from_pretrained(tc.model_path).negative_prompt
    inference_args = make_inference_args(tc, model_path=model_path)
    inference_args.text_encoder_cpu_offload = False
    text_encoder = TextEncoderLoader().load(os.path.join(model_path, 'text_encoder'), inference_args).to('cuda').eval()
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_path, 'tokenizer'))
    refs = [weakref.ref(t) for t in list(text_encoder.parameters()) + list(text_encoder.buffers())]
    module_ref = weakref.ref(text_encoder)
    report['tracked'] = len(refs)
    text = preprocess(prompt) if preprocess is not None else prompt
    with torch.no_grad(), set_forward_context(current_timestep=0, attn_metadata=None):
        text_inputs = tokenizer(text, **dict(ec.tokenizer_kwargs)).to('cuda')
        outputs = text_encoder(input_ids=text_inputs.input_ids, attention_mask=text_inputs.attention_mask)
        outputs.attention_mask = text_inputs['attention_mask']
        embeds = post(outputs).to(device='cuda', dtype=torch.bfloat16)
        mask = text_inputs['attention_mask'].to(device='cuda', dtype=torch.bfloat16)
    if args.disable_gc:
        gc.disable()
    def checkpoint(name):
        torch.cuda.synchronize()
        report['checkpoints'].append({'name': name, 'live': sum(ref() is not None for ref in refs),
          'encoder_live': module_ref() is not None, 'allocated': torch.cuda.memory_allocated(), 'reserved': torch.cuda.memory_reserved()})
        save()
    checkpoint('after_forward')
    del text_encoder, tokenizer, outputs, text_inputs
    checkpoint('after_del')
    if args.debug_cycles:
        gc.set_debug(gc.DEBUG_SAVEALL)
    report['gc_collected'] = gc.collect()
    checkpoint('after_gc')
    if args.debug_cycles:
        encoder = module_ref()
        if encoder is not None:
            ids = {id(x) for x in gc.garbage}
            counts = collections.Counter(type(x).__name__ for x in gc.get_referrers(encoder) if id(x) in ids)
            report['garbage_encoder_referrer_types'] = dict(counts)
        del encoder
        gc.set_debug(0)
    if not args.debug_cycles and report['checkpoints'][-1]['live']:
        obj = next(ref() for ref in refs if ref() is not None)
        def info(x):
            if isinstance(x, dict):
                return {'type': 'dict', 'keys_referring_to_target': [str(k)[:80] for k,v in x.items() if v is obj and isinstance(k, str)][:10]}
            return {'type': type(x).__name__}
        first = gc.get_referrers(obj)
        report['referrer_types'] = [info(x) for x in first[:20]]
        report['second_level_types'] = dict(collections.Counter(type(y).__name__ for x in first[:20] for y in gc.get_referrers(x)))
        del obj, first
        del embeds, mask
        gc.collect()
        checkpoint('after_drop_embeds_mask_gc')
    torch.cuda.empty_cache()
    checkpoint('after_empty_cache')
    report['status'] = 'completed'
    save()


if __name__ == '__main__':
    main()
