"""SFT patch for s-flm. Each target must occur exactly once, else abort with nothing written."""
import sys, pathlib
REPO = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else '/root/autodl-tmp/s-flm')
PATCHES = {
  'dataloader.py': [
    ("  save_dir = (f'{config.data.cache_dir}/tiny_gsm_bs{config.model.length}'\n"
     "              f'_{mask_tag}{filter_tag}{wrap_tag}{train_on_pad}_{tokenizer_tag}')\n",
     "  # [SFT patch] optional local JSON with fields question/code; cache key includes file hash\n"
     "  sft_path = config.data.get('sft_json_path', None)\n"
     "  sft_tag = ''\n"
     "  if sft_path:\n"
     "    import hashlib\n"
     "    with open(sft_path, 'rb') as f:\n"
     "      sft_tag = '_sft_' + hashlib.md5(f.read()).hexdigest()[:12]\n"
     "  save_dir = (f'{config.data.cache_dir}/tiny_gsm_bs{config.model.length}'\n"
     "              f'_{mask_tag}{filter_tag}{wrap_tag}{train_on_pad}_{tokenizer_tag}{sft_tag}')\n"),
    ("    ds = datasets.load_dataset(\n"
     "      'TinyGSM/TinyGSM', split='train', \n"
     "      cache_dir=config.data.cache_dir)\n",
     "    if sft_path:  # [SFT patch]\n"
     "      LOGGER.info(f'Loading SFT data from local JSON: {sft_path}')\n"
     "      ds = datasets.load_dataset('json', data_files=sft_path, split='train')\n"
     "    else:\n"
     "      ds = datasets.load_dataset(\n"
     "        'TinyGSM/TinyGSM', split='train',\n"
     "        cache_dir=config.data.cache_dir)\n"),
  ],
  'main.py': [
    ("      config.training.finetune_path,\n"
     "      tokenizer=tokenizer,\n"
     "      config=config)\n",
     "      config.training.finetune_path,\n"
     "      tokenizer=tokenizer,\n"
     "      config=config)\n"
     "    # [SFT patch] start from checkpoint EMA weights (= what eval used), then disable EMA\n"
     "    if config.training.get('finetune_init_from_ema', False):\n"
     "      assert model.ema is not None, 'checkpoint EMA not loaded'\n"
     "      model.ema.copy_to(model._get_parameters())\n"
     "      model.ema = None\n"
     "      print('[SFT patch] initialized weights from checkpoint EMA; EMA disabled')\n"),
  ],
}
new_texts = {}
for fname, patches in PATCHES.items():
    path = REPO / fname
    text = path.read_text()
    if '[SFT patch]' in text:
        sys.exit(f'ABORT: {fname} already patched.')
    for old, new in patches:
        n = text.count(old)
        if n != 1:
            sys.exit(f'ABORT: snippet found {n} times in {fname} (expected 1). Nothing written.\n---\n{old}')
        text = text.replace(old, new)
    new_texts[path] = text
for path, text in new_texts.items():
    path.write_text(text); print('patched', path)
