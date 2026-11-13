#include <3ds.h>
#include <citro2d.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <string>
#include <vector>
#include <dirent.h>
#include "llama2.h"

// 3DS System Settings Palette (inspired by Nintendo 3DS System Settings)
static constexpr u32 COL_BG         = C2D_Color32(246, 240, 184, 255); // Warm cream yellow
static constexpr u32 COL_BG_STRIPE  = C2D_Color32(240, 234, 174, 255); // Subtle stripe
static constexpr u32 COL_PANEL      = C2D_Color32(223, 216, 156, 255); // Recessed olive/tan box
static constexpr u32 COL_PANEL_BDR  = C2D_Color32(198, 191, 134, 255);
static constexpr u32 COL_DOTS       = C2D_Color32(145, 138, 96, 255);
static constexpr u32 COL_TOP_BAR    = C2D_Color32(246, 240, 184, 255);
static constexpr u32 COL_PILL_BG    = C2D_Color32(248, 248, 246, 255);
static constexpr u32 COL_PILL_BDR   = C2D_Color32(165, 165, 155, 255);
static constexpr u32 COL_BTN_FACE   = C2D_Color32(252, 252, 249, 255);
static constexpr u32 COL_BTN_SHADOW = C2D_Color32(182, 175, 122, 255);
static constexpr u32 COL_BTN_SEL    = C2D_Color32(46, 158, 232, 255);  // Active blue highlight
static constexpr u32 COL_FOOTER     = C2D_Color32(106, 106, 106, 255); // Dark gray bottom bar
static constexpr u32 COL_FOOTER_TOP = C2D_Color32(142, 142, 142, 255);
static constexpr u32 COL_TEXT       = C2D_Color32(58, 55, 46, 255);
static constexpr u32 COL_MUTED      = C2D_Color32(115, 110, 88, 255);
static constexpr u32 COL_ORANGE     = C2D_Color32(224, 124, 24, 255);
static constexpr u32 COL_BLUE       = C2D_Color32(42, 145, 224, 255);
static constexpr u32 COL_PINK       = C2D_Color32(222, 82, 132, 255);
static constexpr u32 COL_GREEN      = C2D_Color32(48, 170, 102, 255);

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

static void drawDottedLine(float x1, float x2, float y) {
    for (float x = x1; x < x2; x += 6.0f) {
        C2D_DrawRectSolid(x, y, 0.5f, 2.0f, 2.0f, COL_DOTS);
    }
}

static void drawSystemButton(float x, float y, float w, float h, float r, bool selected) {
    // 3D drop shadow underneath
    drawRoundedRect(x, y + 3, w, h, r, COL_BTN_SHADOW);
    if (selected) {
        drawRoundedRect(x - 2, y - 2, w + 4, h + 4, r + 2, COL_BTN_SEL);
    } else {
        drawRoundedRect(x - 1, y - 1, w + 2, h + 2, r + 1, COL_PILL_BDR);
    }
    drawRoundedRect(x, y, w, h, r, COL_BTN_FACE);
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

static void drawTextCentered(const char* str, float cx, float y, float scale, u32 color) {
    C2D_Text t;
    C2D_TextParse(&t, g_textBuf, str);
    C2D_TextOptimize(&t);
    float w = 0, h = 0;
    C2D_TextGetDimensions(&t, scale, scale, &w, &h);
    C2D_DrawText(&t, C2D_WithColor, cx - w * 0.5f, y, 0.5f, scale, scale, color);
}

struct AppState {
    bool isNew3DS = false;
    u8 battery = 5;
    std::vector<std::string> modelFiles;
    int selectedModelIdx = 0;
    bool modelLoaded = false;
    std::string statusMsg = "Ready";

    std::string lastPrompt = "Once upon a time";
    std::string outputText = "Place .bin model + tokenizer.bin in sdmc:/3ds/Llama3DS/ and tap Type Prompt.";
    float temperature = 0.8f;
    int maxSteps = 96;
    float lastTokPerSec = 0.0f;

    Transformer transformer{};
    Tokenizer tokenizer{};
    Sampler sampler{};

    int page = 0;      // 0 = 2x2 main tiles, 1 = pill list settings
    int cursorIdx = 0; // 0..3 on page 0, 0..2 on page 1
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
        app.statusMsg = "No .bin found";
        app.outputText = "No model .bin found in sdmc:/3ds/Llama3DS/.";
        return false;
    }
    std::string modelName = app.modelFiles[app.selectedModelIdx];
    std::string modelPath = "sdmc:/3ds/Llama3DS/" + modelName;

    int rc = build_transformer(&app.transformer, modelPath.c_str());
    if (rc != 0) {
        app.statusMsg = "Load err " + std::to_string(rc);
        if (rc == -1) app.outputText = "Err -1: Could not open " + modelPath;
        else if (rc == -2) app.outputText = "Err -2: File too small (" + modelName + ")";
        else if (rc == -3) app.outputText = "Err -3: Out of RAM loading " + modelName + ". Try stories260K.bin or CIA with 124MB mode.";
        else app.outputText = "Err -4: Out of RAM for KV cache (" + modelName + ")";
        return false;
    }

    const char* tokPath = (app.transformer.config.vocab_size <= 1024)
        ? "sdmc:/3ds/Llama3DS/tok512.bin"
        : "sdmc:/3ds/Llama3DS/tokenizer.bin";

    int trc = build_tokenizer(&app.tokenizer, tokPath, app.transformer.config.vocab_size);
    if (trc != 0) {
        trc = build_tokenizer(&app.tokenizer, "sdmc:/3ds/Llama3DS/tokenizer.bin", app.transformer.config.vocab_size);
    }
    if (trc != 0) {
        free_transformer(&app.transformer);
        app.statusMsg = "Tok err " + std::to_string(trc);
        app.outputText = "Missing or invalid tokenizer (" + std::string(tokPath) + ")";
        return false;
    }

    build_sampler(&app.sampler, app.transformer.config.vocab_size, app.temperature, 0.9f, (unsigned long long)osGetTime());
    app.modelLoaded = true;
    app.statusMsg = "Loaded";
    app.outputText = modelName + " ready (" + std::to_string(app.transformer.config.dim) + "d, " +
                     std::to_string(app.transformer.config.n_layers) + "L). Tap Type Prompt to start.";
    return true;
}

static void drawBackground(float w) {
    C2D_DrawRectSolid(0, 0, 0.1f, w, 240, COL_BG);
    for (int y = 0; y < 240; y += 4) {
        C2D_DrawRectSolid(0, (float)y, 0.2f, w, 2, COL_BG_STRIPE);
    }
}

static void renderScreens(C3D_RenderTarget* top, C3D_RenderTarget* bot, AppState& app) {
    C3D_FrameBegin(C3D_FRAME_SYNCDRAW);
    C2D_TextBufClear(g_textBuf);

    // ================= TOP SCREEN (400 x 240) =================
    C2D_TargetClear(top, COL_BG);
    C2D_SceneBegin(top);
    drawBackground(400.0f);

    // System Settings style status bar
    drawRoundedRect(24, 5, 62, 15, 3, COL_PILL_BG);
    drawText(app.modelLoaded ? "Loaded" : "No Model", 30, 6, 0.36f, app.modelLoaded ? COL_GREEN : COL_ORANGE);

    time_t now = time(NULL);
    struct tm* tm_info = localtime(&now);
    char timeBuf[40];
    snprintf(timeBuf, sizeof(timeBuf), "%02d/%02d %02d:%02d",
             tm_info->tm_mon + 1, tm_info->tm_mday, tm_info->tm_hour, tm_info->tm_min);
    drawText(timeBuf, 290, 6, 0.38f, COL_TEXT);

    // Battery indicator
    C2D_DrawRectSolid(368, 7, 0.5f, 20, 10, COL_TEXT);
    C2D_DrawRectSolid(370, 9, 0.6f, (app.battery * 16.0f) / 5.0f, 6, COL_BLUE);

    // Header title + dotted dividers
    drawTextCentered("Llama3DS - Local AI", 200, 26, 0.48f, COL_ORANGE);
    drawDottedLine(20, 380, 44);

    // Recessed olive/tan output panel (like System Settings top box)
    drawRoundedRect(20, 50, 360, 156, 8, COL_PANEL_BDR);
    drawRoundedRect(21, 51, 358, 154, 7, COL_PANEL);

    std::string promptLine = "Prompt: " + app.lastPrompt;
    drawText(promptLine.c_str(), 30, 57, 0.40f, COL_ORANGE, 340.0f);

    std::string trimmed = app.outputText;
    if (trimmed.size() > 380) {
        trimmed = "..." + trimmed.substr(trimmed.size() - 377);
    }
    drawText(trimmed.c_str(), 30, 82, 0.42f, COL_TEXT, 340.0f);

    drawDottedLine(20, 380, 212);

    // Bottom status (like Ver. 0.0.0-0E)
    std::string mdl = app.modelFiles.empty() ? "None" : app.modelFiles[app.selectedModelIdx];
    drawText(("Model: " + mdl).c_str(), 22, 219, 0.38f, COL_MUTED);
    char verBuf[48];
    snprintf(verBuf, sizeof(verBuf), "%s | %.1f tok/s", app.isNew3DS ? "N3DS" : "O3DS", app.lastTokPerSec);
    drawText(verBuf, 275, 219, 0.38f, COL_TEXT);

    // ================= BOTTOM SCREEN (320 x 240) =================
    C2D_TargetClear(bot, COL_BG);
    C2D_SceneBegin(bot);
    drawBackground(320.0f);

    // Page indicator pills at top (like 1 2)
    drawRoundedRect(132, 6, 22, 16, 8, app.page == 0 ? COL_ORANGE : COL_PILL_BG);
    drawTextCentered("1", 143, 8, 0.38f, app.page == 0 ? COL_BTN_FACE : COL_TEXT);
    drawRoundedRect(166, 6, 22, 16, 8, app.page == 1 ? COL_ORANGE : COL_PILL_BG);
    drawTextCentered("2", 177, 8, 0.38f, app.page == 1 ? COL_BTN_FACE : COL_TEXT);

    if (app.page == 0) {
        // 2x2 chunky white rounded buttons (exact System Settings layout)
        struct BigTile { const char* l1; const char* l2; u32 col; };
        BigTile tiles[4] = {
            { "Type",   "Prompt",     COL_BLUE },
            { "Quick",  "Story",      COL_PINK },
            { "Switch", "Model",      COL_GREEN },
            { "Other",  "Settings",   COL_ORANGE }
        };
        for (int i = 0; i < 4; i++) {
            int col = i % 2;
            int row = i / 2;
            float x = 18.0f + col * 146.0f;
            float y = 30.0f + row * 86.0f;
            drawSystemButton(x, y, 138.0f, 76.0f, 10.0f, app.cursorIdx == i);
            drawRoundedRect(x + 53, y + 10, 32, 24, 6, tiles[i].col);
            drawTextCentered(tiles[i].l1, x + 69, y + 40, 0.44f, COL_TEXT);
            drawTextCentered(tiles[i].l2, x + 69, y + 55, 0.44f, COL_TEXT);
        }
    } else {
        // Pill-shaped list rows (exact Other Settings layout)
        char tempBuf[32], stepBuf[32];
        snprintf(tempBuf, sizeof(tempBuf), "Temperature: %.1f", app.temperature);
        snprintf(stepBuf, sizeof(stepBuf), "Token Limit: %d", app.maxSteps);
        const char* rows[3] = { tempBuf, stepBuf, "Clear Output" };
        u32 cols[3] = { COL_ORANGE, COL_BLUE, COL_PINK };

        for (int i = 0; i < 3; i++) {
            float x = 28.0f;
            float y = 36.0f + i * 54.0f;
            drawSystemButton(x, y, 264.0f, 42.0f, 21.0f, app.cursorIdx == i);
            C2D_DrawCircleSolid(x + 24, y + 21, 0.6f, 11.0f, cols[i]);
            drawText(rows[i], x + 46, y + 13, 0.46f, COL_TEXT);
        }
    }

    // Dark gray bottom Close / Back bar (exact System Settings footer)
    C2D_DrawRectSolid(0, 206, 0.5f, 320, 34, COL_FOOTER_TOP);
    C2D_DrawRectSolid(0, 208, 0.6f, 320, 32, COL_FOOTER);
    drawTextCentered(app.page == 0 ? "START: Close" : "B: Back   |   START: Close", 160, 216, 0.45f, COL_BTN_FACE);

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
    app.statusMsg = "Thinking...";
    app.sampler.temperature = app.temperature;
    renderScreens(top, bot, app);

    StreamCtx ctx{ top, bot, &app, osGetTime() };
    generate_tokens(&app.transformer, &app.tokenizer, &app.sampler, prompt.c_str(), app.maxSteps, onTokenGenerated, &ctx);
    app.statusMsg = "Done";
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

        if ((kDown & KEY_B) && app.page == 1) {
            app.page = 0;
            app.cursorIdx = 3;
        }
        if (kDown & KEY_L) { app.page = 0; app.cursorIdx = 0; }
        if (kDown & KEY_R) { app.page = 1; app.cursorIdx = 0; }

        if (app.page == 0) {
            if (kDown & KEY_RIGHT) app.cursorIdx = (app.cursorIdx + 1) % 4;
            if (kDown & KEY_LEFT)  app.cursorIdx = (app.cursorIdx + 3) % 4;
            if (kDown & KEY_DOWN)  app.cursorIdx = (app.cursorIdx + 2) % 4;
            if (kDown & KEY_UP)    app.cursorIdx = (app.cursorIdx + 2) % 4;
        } else {
            if (kDown & KEY_DOWN) app.cursorIdx = (app.cursorIdx + 1) % 3;
            if (kDown & KEY_UP)   app.cursorIdx = (app.cursorIdx + 2) % 3;
        }

        bool activate = (kDown & KEY_A) != 0;

        if (kDown & KEY_TOUCH) {
            touchPosition touch;
            hidTouchRead(&touch);
            // Footer bar tap
            if (touch.py >= 206) {
                if (app.page == 1) { app.page = 0; app.cursorIdx = 3; }
                else break;
            }
            // Top page dots tap
            if (touch.py <= 24) {
                if (touch.px >= 125 && touch.px <= 156) { app.page = 0; app.cursorIdx = 0; }
                if (touch.px >= 162 && touch.px <= 194) { app.page = 1; app.cursorIdx = 0; }
            }
            if (app.page == 0) {
                for (int i = 0; i < 4; i++) {
                    float x = 18.0f + (i % 2) * 146.0f;
                    float y = 30.0f + (i / 2) * 86.0f;
                    if (touch.px >= x && touch.px <= x + 138.0f && touch.py >= y && touch.py <= y + 76.0f) {
                        app.cursorIdx = i;
                        activate = true;
                        break;
                    }
                }
            } else {
                for (int i = 0; i < 3; i++) {
                    float x = 28.0f;
                    float y = 36.0f + i * 54.0f;
                    if (touch.px >= x && touch.px <= x + 264.0f && touch.py >= y && touch.py <= y + 42.0f) {
                        app.cursorIdx = i;
                        activate = true;
                        break;
                    }
                }
            }
        }

        if (activate) {
            if (app.page == 0) {
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
                        loadSelectedModel(app);
                    }
                } else if (app.cursorIdx == 3) {
                    app.page = 1;
                    app.cursorIdx = 0;
                }
            } else {
                if (app.cursorIdx == 0) {
                    app.temperature += 0.2f;
                    if (app.temperature > 1.25f) app.temperature = 0.0f;
                } else if (app.cursorIdx == 1) {
                    app.maxSteps += 32;
                    if (app.maxSteps > 192) app.maxSteps = 32;
                } else if (app.cursorIdx == 2) {
                    app.outputText = "Output cleared.";
                    app.page = 0;
                    app.cursorIdx = 0;
                }
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
