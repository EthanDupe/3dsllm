# Llama3DS

Local AI chat homebrew for Nintendo 3DS / New 2DS XL powered by [`llama2.c`](https://github.com/karpathy/llama2.c) and `citro2d`, styled after the 3DS System Applets and [`3dsDL`](https://github.com/ankushKun/3dsDL).

## Features
- **3DS System Applet / 3dsDL UI:** Top screen displays hardware status (`N3DS 804MHz` / `O3DS 268MHz`), battery, token speed (`tok/s`), and chat bubbles; bottom screen has a 6-tile D-pad and touch grid.
- **Native 3DS Keyboard:** Uses `swkbd` so you can type custom prompts.
- **Supports Float32 & Int8 Quantized Models:** Drop `stories15M.bin` (or `stories260K.bin` / quantized `q8_0` checkpoints) + `tokenizer.bin` (or `tok512.bin`) into `sdmc:/3ds/Llama3DS/`.
- **GitHub Actions Ready:** Compiles `.3dsx` and `.cia` automatically via `.github/workflows/build.yml`.

## SD Card Setup
Copy your model and tokenizer from [`karpathy/llama2.c`](https://github.com/karpathy/llama2.c) to your SD card:
```
sdmc:/3ds/Llama3DS/stories15M.bin
sdmc:/3ds/Llama3DS/tokenizer.bin
```
