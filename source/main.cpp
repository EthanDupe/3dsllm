#include <3ds.h>
#include <citro2d.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>
#include <dirent.h>
#include "llama2.h"

// 3DS System-Applet + 3dsDL Palette
static constexpr u32 COL_BG        = C2D_Color32(236, 241, 247, 255);
static constexpr u32 COL_CARD      = C2D_Color32(255, 255, 255, 255);
static constexpr u32 COL_CARD_ALT  = C2D_Color32(244, 247, 251, 255);
static constexpr u32 COL_BORDER    = C2D_Color32(205, 216, 229, 255);
static constexpr u32 COL_ACCENT    = C2D_Color32(22, 145, 219, 255);
static constexpr u32 COL_ACCENT_DK = C2D_Color32(15, 108, 172, 255);
static constexpr u32 COL_HEADER    = C2D_Color32(24, 42, 68, 255);
static constexpr u32 COL_TEXT      = C2D_Color32(25, 36, 52, 255);
static constexpr u32 COL_MUTED     = C2D_Color32(98, 114, 134, 255);
static constexpr u32 COL_USER_BUB  = C2D_Color32(219, 240, 255, 255);
static constexpr u32 COL_AI_BUB    = C2D_Color32(255, 255, 255, 255);
static constexpr u32 COL_GREEN     = C2D_Color32(34, 160, 96, 255);
static constexpr u32 COL_AMBER     = C2D_Color32(218, 142, 20, 255);

static C2D_TextBuf g_textBuf = nullptr;

static void drawRoundedRect(float x, float y, float w, float h, float r, u32 color) {
    if (r * 2 > w) r = w / 2.0f;
    if (r * 2 > h) r = h / 2.0f;
    C2D_DrawRectSolid(x + r, y, 0.5f, w - 2 * r, h, color);
    C2D_DrawRectSolid(x, y + r, 0.5f, r, h - 2 * r, color);
    C2D_DrawRectSolid(x + w - r, y + r, 0.5f, r, h - 2 * r, color);
    C2D_DrawCircleSolid(x + r, y + r, 0.5f, r, color);
    C2D_DrawCircleSolid(x + w - r, y + r, 0.5f, r, color);
    C2D_DrawCircleSolid(x + r, y + h - r, 0.5f, r, color);
    C2D_DrawCircleSolid(x + w - r, y + h - r, 0.5f, r, color);
}

static void drawRoundedCard(float x, float y, float w, float h, float r, u32 fill, u32 border, bool selected) {
    if (selected) {
        drawRoundedRect(x - 2, y - 2, w + 4, h + 4, r + 2, COL_ACCENT);
    } else {
        drawRoundedRect(x - 1, y - 1, w + 2, h + 2, r + 1, border);
    }
    drawRoundedRect(x, y, w, h, r, fill);
}

static void drawText(const char* str, float x, float y, float scale, u32 color, float maxW = 0.0f) {
    C2D_Text t;
    C2D_TextParse(&t, g_textBuf, str);
    C2D_TextOptimize(&t);
    if (maxW > 0.0f) {
        C2D_DrawText(&t, C2D_WithColor | C2D_WordWrap, x, y, 0.5f, scale, scale, color, maxW);
    } else {
        C2D_DrawText(&t, C2D_WithColor, x, y, 0.5f, scale, scale, color);
    }
}

struct AppState {
    bool isNew3DS = false;
    u8 battery = 5;
    std::vector<std::string> modelFiles;
    int selectedModelIdx = 0;
    bool modelLoaded = false;
    std::string statusMsg = "Select a model or tap Prompt";

    std::string lastPrompt = "Once upon a time";
    std::string outputText = "Place .bin models and tokenizer.bin in sdmc:/3ds/Llama3DS/ then press A or tap Load Model.";
    float temperature = 0.8f;
    int maxSteps = 96;
    float lastTokPerSec = 0.0f;

    Transformer transformer{};
    Tokenizer tokenizer{};
    Sampler sampler{};

    int cursorIdx = 0; // 0: Type Prompt, 1: Run Preset, 2: Switch Model, 3: Temp, 4: Steps, 5: Clear
    bool generating = false;
};

static void scanModels(AppState& app) {
    app.modelFiles.clear();
    mkdir("sdmc:/3ds", 0777);
    mkdir("sdmc:/3ds/Llama3DS", 0777);
    DIR* dir = opendir("sdmc:/3ds/Llama3DS");
    if (!dir) return;
    struct dirent* ent;
    while ((ent = readdir(dir)) != nullptr) {
        std::string name = ent->d_name;
        if (name == "tokenizer.bin" || name == "tok512.bin") continue;
        if (name.size() > 4 && name.substr(name.size() - 4) == ".bin") {
            app.modelFiles.push_back(name);
        }
    }
    closedir(dir);
}

static void unloadModel(AppState& app) {
    if (app.modelLoaded) {
        free_sampler(&app.sampler);
        free_tokenizer(&app.tokenizer);
        free_transformer(&app.transformer);
        app.modelLoaded = false;
    }
}

static bool loadSelectedModel(AppState& app) {
    unloadModel(app);
    if (app.modelFiles.empty()) {
        app.statusMsg = "No .bin model in sdmc:/3ds/Llama3DS/";
        return false;
    }
    std::string modelPath = "sdmc:/3ds/Llama3DS/" + app.modelFiles[app.selectedModelIdx];
    app.statusMsg = "Loading " + app.modelFiles[app.selectedModelIdx] + "...";

    if (build_transformer(&app.transformer, modelPath.c_str()) != 0) {
        app.statusMsg = "Failed to load model (check RAM / file)";
        return false;
    }

    const char* tokPath = (app.transformer.config.vocab_size <= 1024)
        ? "sdmc:/3ds/Llama3DS/tok512.bin"
        : "sdmc:/3ds/Llama3DS/tokenizer.bin";

    if (build_tokenizer(&app.tokenizer, tokPath, app.transformer.config.vocab_size) != 0) {
        // Fallback try tokenizer.bin
        if (build_tokenizer(&app.tokenizer, "sdmc:/3ds/Llama3DS/tokenizer.bin", app.transformer.config.vocab_size) != 0) {
            free_transformer(&app.transformer);
            app.statusMsg = "Missing tokenizer.bin or tok512.bin";
            return false;
        }
    }

    build_sampler(&app.sampler, app.transformer.config.vocab_size, app.temperature, 0.9f, (unsigned long long)osGetTime());
    app.modelLoaded = true;
    app.statusMsg = "Loaded: " + app.modelFiles[app.selectedModelIdx];
    app.outputText = "Model ready (" + std::to_string(app.transformer.config.dim) + "d, " +
                     std::to_string(app.transformer.config.n_layers) + "L" +
                     (app.transformer.quantized ? ", Q8)" : ", F32)") + ". Tap Type Prompt to chat.";
    return true;
}

static void renderScreens(C3D_RenderTarget* top, C3D_RenderTarget* bot, AppState& app) {
    C3D_FrameBegin(C3D_FRAME_SYNCDRAW);
    C2D_TextBufClear(g_textBuf);

    // ================= TOP SCREEN (400 x 240) =================
    C2D_TargetClear(top, COL_BG);
    C2D_SceneBegin(top);

    // Header bar (3dsDL / System Applet style)
    C2D_DrawRectSolid(0, 0, 0.5f, 400, 26, COL_HEADER);
    drawText("Llama3DS", 10, 5, 0.52f, C2D_Color32(255, 255, 255, 255));
    char hwBadge[64];
    snprintf(hwBadge, sizeof(hwBadge), "%s | Bat %d/5 | %.1f tok/s",
             app.isNew3DS ? "N3DS 804MHz" : "O3DS 268MHz",
             app.battery, app.lastTokPerSec);
    drawText(hwBadge, 145, 7, 0.40f, C2D_Color32(170, 210, 245, 255));

    // User Prompt Bubble
    drawRoundedCard(10, 34, 380, 36, 8, COL_USER_BUB, COL_BORDER, false);
    std::string promptLine = "You: " + app.lastPrompt;
    drawText(promptLine.c_str(), 18, 42, 0.44f, COL_TEXT, 364.0f);

    // AI Output Card
    drawRoundedCard(10, 76, 380, 136, 10, COL_AI_BUB, COL_BORDER, false);
    std::string trimmed = app.outputText;
    if (trimmed.size() > 420) {
        trimmed = "..." + trimmed.substr(trimmed.size() - 417);
    }
    drawText(trimmed.c_str(), 18, 84, 0.43f, COL_TEXT, 364.0f);

    // Bottom Shoulder Bar (3dsDL style)
    C2D_DrawRectSolid(0, 218, 0.5f, 400, 22, C2D_Color32(223, 231, 241, 255));
    drawText("L: Reload Model", 10, 222, 0.40f, COL_MUTED);
    drawText(app.statusMsg.c_str(), 125, 222, 0.40f, app.modelLoaded ? COL_GREEN : COL_AMBER);
    drawText("START: Exit", 322, 222, 0.40f, COL_MUTED);

    // ================= BOTTOM SCREEN (320 x 240) =================
    C2D_TargetClear(bot, COL_BG);
    C2D_SceneBegin(bot);

    // Top mini bar
    C2D_DrawRectSolid(0, 0, 0.5f, 320, 24, COL_CARD);
    C2D_DrawRectSolid(0, 23, 0.5f, 320, 1, COL_BORDER);
    std::string mdl = app.modelFiles.empty() ? "No .bin found" : app.modelFiles[app.selectedModelIdx];
    drawText(("Model: " + mdl).c_str(), 10, 5, 0.42f, COL_TEXT);

    // 6-Tile 3dsDL / HOME Menu Grid (2 columns x 3 rows)
    struct Tile { const char* title; const char* sub; u32 accent; };
    char tempBuf[32], stepBuf[32];
    snprintf(tempBuf, sizeof(tempBuf), "Value: %.1f", app.temperature);
    snprintf(stepBuf, sizeof(stepBuf), "Max: %d tok", app.maxSteps);

    Tile tiles[6] = {
        { "Type Prompt",   "Open 3DS Keyboard", COL_ACCENT },
        { "Quick Story",   "Run sample prompt", COL_GREEN },
        { "Next Model",    "Cycle SD models",   COL_ACCENT_DK },
        { "Temperature",   tempBuf,             COL_AMBER },
        { "Token Limit",   stepBuf,             COL_MUTED },
        { "Clear Chat",    "Reset output box",  C2D_Color32(190, 65, 65, 255) }
    };

    for (int i = 0; i < 6; i++) {
        int col = i % 2;
        int row = i / 2;
        float x = 12.0f + col * 152.0f;
        float y = 32.0f + row * 64.0f;
        float w = 144.0f;
        float h = 56.0f;
        drawRoundedCard(x, y, w, h, 10, COL_CARD, COL_BORDER, app.cursorIdx == i);
        C2D_DrawRectSolid(x + 8, y + 10, 0.6f, 4, h - 20, tiles[i].accent);
        drawText(tiles[i].title, x + 18, y + 11, 0.46f, COL_TEXT);
        drawText(tiles[i].sub,   x + 18, y + 32, 0.37f, COL_MUTED);
    }

    drawText("D-Pad / Touch: Select   |   A: Activate", 34, 224, 0.38f, COL_MUTED);

    C3D_FrameEnd(0);
}

struct StreamCtx {
    C3D_RenderTarget* top;
    C3D_RenderTarget* bot;
    AppState* app;
    u64 startTick;
};

static void onTokenGenerated(const char* piece, int step, void* userdata) {
    StreamCtx* ctx = (StreamCtx*)userdata;
    ctx->app->outputText += piece;
    u64 elapsedMs = (osGetTime() - ctx->startTick);
    if (elapsedMs > 0) {
        ctx->app->lastTokPerSec = (step * 1000.0f) / (float)elapsedMs;
    }
    if (step % 3 == 0) {
        renderScreens(ctx->top, ctx->bot, *ctx->app);
    }
}

static void runGeneration(C3D_RenderTarget* top, C3D_RenderTarget* bot, AppState& app, const std::string& prompt) {
    if (!app.modelLoaded) {
        if (!loadSelectedModel(app)) return;
    }
    app.lastPrompt = prompt;
    app.outputText = prompt;
    app.statusMsg = "Generating...";
    app.sampler.temperature = app.temperature;
    renderScreens(top, bot, app);

    StreamCtx ctx{ top, bot, &app, osGetTime() };
    generate_tokens(&app.transformer, &app.tokenizer, &app.sampler, prompt.c_str(), app.maxSteps, onTokenGenerated, &ctx);
    app.statusMsg = "Done (" + std::to_string((int)app.lastTokPerSec) + " tok/s)";
}

int main() {
    gfxInitDefault();
    osSetSpeedupEnable(true);
    ptmuInit();

    C3D_Init(C3D_DEFAULT_CMDBUF_SIZE);
    C2D_Init(C2D_DEFAULT_MAX_OBJECTS);
    C2D_Prepare();
    g_textBuf = C2D_TextBufNew(4096);

    C3D_RenderTarget* top = C2D_CreateScreenTarget(GFX_TOP, GFX_LEFT);
    C3D_RenderTarget* bot = C2D_CreateScreenTarget(GFX_BOTTOM, GFX_LEFT);

    AppState app;
    APT_CheckNew3DS(&app.isNew3DS);
    PTMU_GetBatteryLevel(&app.battery);
    scanModels(app);
    if (!app.modelFiles.empty()) {
        loadSelectedModel(app);
    }

    while (aptMainLoop()) {
        hidScanInput();
        u32 kDown = hidKeysDown();
        if (kDown & KEY_START) break;

        if (kDown & KEY_L) {
            scanModels(app);
            loadSelectedModel(app);
        }

        if (kDown & KEY_RIGHT) app.cursorIdx = (app.cursorIdx + 1) % 6;
        if (kDown & KEY_LEFT)  app.cursorIdx = (app.cursorIdx + 5) % 6;
        if (kDown & KEY_DOWN)  app.cursorIdx = (app.cursorIdx + 2) % 6;
        if (kDown & KEY_UP)    app.cursorIdx = (app.cursorIdx + 4) % 6;

        bool activate = (kDown & KEY_A) != 0;

        if (kDown & KEY_TOUCH) {
            touchPosition touch;
            hidTouchRead(&touch);
            for (int i = 0; i < 6; i++) {
                int col = i % 2;
                int row = i / 2;
                float x = 12.0f + col * 152.0f;
                float y = 32.0f + row * 64.0f;
                if (touch.px >= x && touch.px <= x + 144.0f && touch.py >= y && touch.py <= y + 56.0f) {
                    app.cursorIdx = i;
                    activate = true;
                    break;
                }
            }
        }

        if (activate) {
            if (app.cursorIdx == 0) {
                SwkbdState swkbd;
                char inputBuf[256] = {0};
                swkbdInit(&swkbd, SWKBD_TYPE_NORMAL, 2, -1);
                swkbdSetHintText(&swkbd, "Enter a prompt for local AI...");
                SwkbdButton btn = swkbdInputText(&swkbd, inputBuf, sizeof(inputBuf));
                if (btn == SWKBD_BUTTON_RIGHT && inputBuf[0] != '\0') {
                    runGeneration(top, bot, app, std::string(inputBuf));
                }
            } else if (app.cursorIdx == 1) {
                runGeneration(top, bot, app, "Lily was a little girl who found a glowing blue");
            } else if (app.cursorIdx == 2) {
                if (!app.modelFiles.empty()) {
                    app.selectedModelIdx = (app.selectedModelIdx + 1) % (int)app.modelFiles.size();
                    loadSelectedModel(app);
                } else {
                    scanModels(app);
                }
            } else if (app.cursorIdx == 3) {
                app.temperature += 0.2f;
                if (app.temperature > 1.25f) app.temperature = 0.0f;
            } else if (app.cursorIdx == 4) {
                app.maxSteps += 32;
                if (app.maxSteps > 192) app.maxSteps = 32;
            } else if (app.cursorIdx == 5) {
                app.outputText = "Chat cleared. Tap Type Prompt to start.";
            }
        }

        renderScreens(top, bot, app);
    }

    unloadModel(app);
    C2D_TextBufDelete(g_textBuf);
    C2D_Fini();
    C3D_Fini();
    ptmuExit();
    gfxExit();
    return 0;
}
