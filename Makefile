name: Build 2DS XL CIA, 3DSX & QR Code

on:
  push:
    branches: [ main, master ]
  workflow_dispatch:

permissions:
  contents: write

jobs:
  build:
    runs-on: ubuntu-latest
    container:
      image: devkitpro/devkitarm:latest

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Ensure 3DS libraries and build tools are installed
        run: |
          dkp-pacman -Sy --noconfirm 3ds-dev || true
          apt-get update && apt-get install -y cmake g++ git python3-qrcode python3-pil

      - name: Install makerom and build bannertool in local tools/bin
        run: |
          mkdir -p tools/bin
          curl -sL -o makerom.zip "https://github.com/3DSGuy/Project_CTR/releases/download/makerom-v0.18.4/makerom-v0.18.4-ubuntu_x86_64.zip"
          unzip -o makerom.zip -d tools/bin
          git clone --depth=1 https://github.com/carstene1ns/3ds-bannertool.git bannertool_src
          cmake -S bannertool_src -B bannertool_src/build -DCMAKE_BUILD_TYPE=Release
          cmake --build bannertool_src/build -j$(nproc)
          cp bannertool_src/build/bannertool tools/bin/bannertool
          chmod +x tools/bin/makerom tools/bin/bannertool
          rm -rf makerom.zip bannertool_src

      - name: Build 3DSX and ELF
        run: |
          export PATH="$PWD/tools/bin:$PATH"
          make -j$(nproc)

      - name: Package CIA
        run: |
          export PATH="$PWD/tools/bin:$PATH"
          python3 -c "import wave; w=wave.open('build/silence.wav','wb'); w.setnchannels(1); w.setsampwidth(2); w.setframerate(22050); w.writeframes(b'\x00\x00'*4410); w.close()"
          bannertool makebanner -i resources/banner.png -a build/silence.wav -o build/banner.bnr
          bannertool makesmdh -s "Llama3DS" -l "Local AI Chat (llama2.c)" -p "Eko" -i resources/icon.png -o build/icon.icn
          makerom -f cia -o Llama3DS.cia -elf Llama3DS.elf -rsf meta/app.rsf -banner build/banner.bnr -icon build/icon.icn

      - name: Generate QR code for direct CIA install
        run: |
          CIA_URL="https://github.com/${{ github.repository }}/releases/download/latest/Llama3DS.cia"
          python3 -c "import qrcode; img = qrcode.make('${CIA_URL}'); img.save('Llama3DS_QR.png')"

      - name: Publish GitHub Release with CIA, 3DSX, and QR code
        uses: softprops/action-gh-release@v2
        with:
          tag_name: latest
          name: Latest Build
          body: |
            Scan `Llama3DS_QR.png` with **3dsDL** or **FBI** to install `Llama3DS.cia` over Wi-Fi.
          files: |
            Llama3DS.cia
            Llama3DS.3dsx
            Llama3DS_QR.png

      - name: Upload workflow artifacts
        uses: actions/upload-artifact@v4
        with:
          name: Llama3DS-Build
          path: |
            Llama3DS.3dsx
            Llama3DS.cia
            Llama3DS_QR.png
