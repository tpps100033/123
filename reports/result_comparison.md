# Result comparison — pending local validation

No local predictions have been produced. The runtime and data handoffs are unavailable in this host session, so there is no defensible Colab/local numerical comparison yet.

The first comparison must pair each local `run_manifest.json` with the same source Colab dataset/model/profile run and confirm identical sample IDs, image preprocessing, keypoint mapping, and protocol. Compare end-to-end OKS/PCK, conditional OKS/PCK, coverage, and latency separately. Preserve the v4 rules: batch 1, 10 warm-up images, FP32, and no AMP/compile/quantization/TensorRT. Do not compare imported legacy latency rows with new fair-reference timings.

| Model | Dataset | Colab run ID | Local run ID | Sample IDs match | Score delta | Latency delta | Review |
|---|---|---|---|---|---:|---:|---|
| Pending | COCO SinglePerson val2017 | — | — | — | — | — | Waiting for validated local run |
| Pending | SyRIP Test100 | — | — | — | — | — | Waiting for validated local run |
