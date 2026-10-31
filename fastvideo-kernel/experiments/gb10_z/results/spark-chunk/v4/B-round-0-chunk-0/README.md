# B-round-0-chunk-0

Status: **completed**. Tokens: 31200; chunk: 0; requested updates: 2.
See ../README.md for environment, service controls, inputs, interpretation and limitations.

```bash
python fastvideo-kernel/experiments/gb10_z/code/chunk_v4_entry.py --tokens 31200 --steps 2 --chunk 0 --model "$MODEL" --data "$ONE_ROW_DATA" --output "$FRESH_OUTPUT" --lock "$LOCK_A" --lock "$LOCK_B"
```

This is one attempt; no automatic retry. Full outcome and failure metadata are in result.json; numerical observations and phase memory are in observation.json and memory.json.
