"""A tiny random Gemma 4 with a vision tower, and what transformers makes of an image with it.

    python tests/Amql.Tests/fixtures/make_gemma4_vision.py

golden.json: size (the image's), patches (the first and last of what the processor cuts, and how
many across), features (the projected soft tokens), ids (a prompt with -1 where the image's tokens
stand) and logits (after the prompt, and the greedy tokens that follow). picture.png is the image;
photo*.jpg and the small PNGs are others, with what Pillow reads them as.
"""
import io
import json
import os

import numpy as np
import torch
from PIL import Image
from transformers import Gemma4Config, Gemma4ForConditionalGeneration
from transformers.models.gemma4.image_processing_pil_gemma4 import Gemma4ImageProcessorPil

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "gemma4-vision")
IMAGE, BOI, EOI = 120, 121, 122


def picture(w, h, seed):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w]
    base = np.stack([128 + 100 * np.sin(x / 7 + seed), 128 + 90 * np.cos(y / 5), (x * 255 // w + y * 255 // h) / 2], -1)
    return np.clip(base + rng.normal(0, 12, base.shape), 0, 255).astype(np.uint8)


def files():
    """Other files a reader meets, and their pixels as Pillow has them."""
    out = {}
    rgb = Image.fromarray(picture(61, 43, 3))
    rgb.save(os.path.join(OUT, "photo.jpg"), quality=92)                                   # 4:2:0
    rgb.save(os.path.join(OUT, "photo-444.jpg"), quality=85, subsampling=0)
    rgb.save(os.path.join(OUT, "photo-422.jpg"), quality=85, subsampling=1)
    rgb.save(os.path.join(OUT, "photo-progressive.jpg"), quality=80, progressive=True)
    rgb.save(os.path.join(OUT, "photo-restart.jpg"), quality=80, restart_marker_blocks=2)
    rgb.convert("L").save(os.path.join(OUT, "photo-grey.jpg"), quality=90)
    exif = Image.Exif()
    exif[0x0112] = 6
    rgb.save(os.path.join(OUT, "photo-turned.jpg"), quality=95, exif=exif)
    rgb.save(os.path.join(OUT, "rgb.png"))
    rgb.convert("L").save(os.path.join(OUT, "grey.png"))
    rgb.convert("P", palette=Image.Palette.ADAPTIVE, colors=200).save(os.path.join(OUT, "palette.png"))
    rgb.convert("1").save(os.path.join(OUT, "bit.png"))
    rgba = rgb.convert("RGBA")
    alpha = np.tile(np.linspace(0, 255, 61).astype(np.uint8), (43, 1))
    rgba.putalpha(Image.fromarray(alpha))
    rgba.save(os.path.join(OUT, "alpha.png"))
    Image.fromarray(np.asarray(rgb.convert("L")).astype(np.uint16) * 257).save(os.path.join(OUT, "deep.png"))
    from PIL import ImageOps
    for name in sorted(os.listdir(OUT)):
        if name.endswith((".jpg", ".png")) and name != "picture.png":
            im = ImageOps.exif_transpose(Image.open(os.path.join(OUT, name)))
            if im.mode == "RGBA":
                white = Image.new("RGBA", im.size, (255, 255, 255, 255))
                im = Image.alpha_composite(white, im)
            if im.mode.startswith("I"):
                im = Image.fromarray((np.asarray(im) // 256).astype(np.uint8))
            a = np.asarray(im.convert("RGB"))
            out[name] = {"width": a.shape[1], "height": a.shape[0], "pixels": a.reshape(-1).tolist()}
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    zoo = json.load(open(os.path.join(HERE, "arch-zoo", "gemma4-moe", "config.json")))
    text = {k: v for k, v in zoo.items() if k not in ("architectures", "dtype", "transformers_version", "per_layer_config", "head_dim", "num_key_value_heads")}
    text.update(head_dim=16, num_key_value_heads=2, global_head_dim=32, num_global_key_value_heads=1, sliding_window=8,
                use_bidirectional_attention="vision", vocab_size_per_layer_input=128)
    vision = dict(hidden_size=64, intermediate_size=96, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4, head_dim=24, global_head_dim=24,
                  patch_size=4, pooling_kernel_size=3, position_embedding_size=2048, default_output_length=12, standardize=True,
                  use_clipped_linears=False, hidden_activation="gelu_pytorch_tanh", rms_norm_eps=1e-6, rope_parameters={"rope_theta": 100.0, "rope_type": "default"})
    cfg = Gemma4Config(text_config=text, vision_config=vision, audio_config=None, image_token_id=IMAGE, boi_token_id=BOI, eoi_token_id=EOI,
                       video_token_id=123, audio_token_id=124, boa_token_id=125, eoa_token_id=126, tie_word_embeddings=True)
    cfg._attn_implementation = "eager"
    torch.manual_seed(11)
    model = Gemma4ForConditionalGeneration(cfg).float().eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "vision" in name and p.dim() >= 2:
                p.normal_(0, 0.12)
        for name, b in model.named_buffers():
            if name.endswith("std_bias"):
                b.normal_(0, 0.5)
            elif name.endswith("std_scale"):
                b.uniform_(0.5, 1.5)
            elif name.endswith("layer_scalar"):
                b.fill_(0.8)
    model.save_pretrained(OUT)

    Image.fromarray(picture(77, 50, 1)).save(os.path.join(OUT, "picture.png"))
    image = Image.open(os.path.join(OUT, "picture.png")).convert("RGB")
    processor = Gemma4ImageProcessorPil(patch_size=4, max_soft_tokens=280, pooling_kernel_size=3)
    processor.max_soft_tokens = 12
    import transformers.models.gemma4.image_processing_pil_gemma4 as mod
    mod._SUPPORTED_SOFT_TOKENS = tuple(set(mod._SUPPORTED_SOFT_TOKENS) | {12})
    batch = processor(images=[image], return_tensors="pt")
    pixels, positions = batch["pixel_values"], batch["image_position_ids"]
    n = int(batch["num_soft_tokens_per_image"][0])
    real = positions[0, :, 0] >= 0
    with torch.no_grad():
        features = model.model.get_image_features(pixels, positions, return_dict=True).pooler_output[0]
        ids = [1, 7, 42, BOI] + [IMAGE] * n + [EOI, 99, 17, 5]
        types = [0, 0, 0, 0] + [1] * n + [0] * 4
        out = model(input_ids=torch.tensor([ids]), pixel_values=pixels, image_position_ids=positions,
                    mm_token_type_ids=torch.tensor([types]), use_cache=False)
        logits = [out.logits[0, -1].tolist()]
        tokens = []
        for _ in range(6):
            tokens.append(int(np.argmax(logits[-1])))
            full = ids + tokens
            out = model(input_ids=torch.tensor([full]), pixel_values=pixels, image_position_ids=positions,
                        mm_token_type_ids=torch.tensor([types + [0] * len(tokens)]), use_cache=False)
            logits.append(out.logits[0, -1].tolist())
        causal = model(input_ids=torch.tensor([ids]), pixel_values=pixels, image_position_ids=positions, use_cache=False).logits[0, -1]
    print("what seeing whole images changes:", float((causal - torch.tensor(logits[0])).abs().max()))
    patches = pixels[0][real]
    json.dump({
        "size": list(image.size), "tokens_at_most": 12, "across": int(positions[0, :, 0].max()) + 1, "tokens": n,
        "first": patches[0].tolist(), "last": patches[-1].tolist(), "count": int(real.sum()),
        "features": features.tolist(), "ids": [i if i != IMAGE else -1 for i in ids], "tokens_after": tokens, "logits": logits,
        "files": files(),
    }, open(os.path.join(OUT, "golden.json"), "w"))
    print(n, "tokens", features.shape, tokens, os.listdir(OUT))


if __name__ == "__main__":
    main()
