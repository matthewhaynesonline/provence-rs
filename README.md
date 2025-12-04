# Provence.rs

Rust _inference_ implementation of the Provence Reranker model. https://huggingface.co/naver/provence-reranker-debertav3-v1

See: https://github.com/huggingface/candle/pull/3191

## Important Provence License Info

**[NOTE: the Provence model weights are CC-BY-NC-ND-4.0](https://huggingface.co/naver/provence-reranker-debertav3-v1/blob/main/Provence_LICENSE.txt)**. **This repository contains only inference code.** It does not include any model weights directly.

If you use this code with the **Provence** model weights from Naver, be aware that those weights are subject to a [CC BY-NC-ND 4.0 license](https://huggingface.co/naver/provence-reranker-debertav3-v1/blob/main/Provence_LICENSE.txt).

This inference code is model agnostic and could be used with other compatible weights. This repository make no representations regarding the licensing of any model weights.

## Example

Note that all examples here use the `metal`. You may need to adjust this to match your environment.

### Single Questing and Context

```bash
cargo run --example provence --release --features=metal

cargo run --example provence --release --features=metal -- -q "What is used to thicken a classic béchamel sauce?" -c "Béchamel sauce. Basics. Béchamel is one of the five mother sauces of French cuisine. It is a simple white sauce made from a roux of butter and flour, to which milk is gradually added while whisking to avoid lumps. The roux acts as the thickening agent, giving the sauce a smooth, creamy texture. Variations. Some chefs add a pinch of nutmeg for flavor. In Italian cuisine, a similar sauce called besciamella is often used in lasagna. Modern adaptations may substitute butter with olive oil or milk with plant-based alternatives, but the thickening principle with flour remains the same." -t="0.35"
```

### Running on CPU

To run the example on CPU, supply the `--cpu` flag.
