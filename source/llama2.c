#include "llama2.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>

static int malloc_run_state(RunState* s, Config* p, int quantized, int group_size) {
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    s->x = (float*)calloc(p->dim, sizeof(float));
    s->xb = (float*)calloc(p->dim, sizeof(float));
    s->xb2 = (float*)calloc(p->dim, sizeof(float));
    s->hb = (float*)calloc(p->hidden_dim, sizeof(float));
    s->hb2 = (float*)calloc(p->hidden_dim, sizeof(float));
    s->q = (float*)calloc(p->dim, sizeof(float));
    s->key_cache = (float*)calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->value_cache = (float*)calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->att = (float*)calloc(p->n_heads * p->seq_len, sizeof(float));
    s->logits = (float*)calloc(p->vocab_size, sizeof(float));

    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q ||
        !s->key_cache || !s->value_cache || !s->att || !s->logits) {
        return -1;
    }

    if (quantized) {
        int gs = group_size > 0 ? group_size : 64;
        s->xq.q = (int8_t*)calloc(p->dim, sizeof(int8_t));
        s->xq.s = (float*)calloc(p->dim / gs + 1, sizeof(float));
        s->hq.q = (int8_t*)calloc(p->hidden_dim, sizeof(int8_t));
        s->hq.s = (float*)calloc(p->hidden_dim / gs + 1, sizeof(float));
        if (!s->xq.q || !s->xq.s || !s->hq.q || !s->hq.s) return -1;
    } else {
        s->xq.q = NULL; s->xq.s = NULL;
        s->hq.q = NULL; s->hq.s = NULL;
    }
    return 0;
}

static void free_run_state(RunState* s) {
    free(s->x); free(s->xb); free(s->xb2);
    free(s->hb); free(s->hb2); free(s->q);
    free(s->key_cache); free(s->value_cache);
    free(s->att); free(s->logits);
    free(s->xq.q); free(s->xq.s);
    free(s->hq.q); free(s->hq.s);
}

static void memory_map_weights_f32(TransformerWeights* w, Config* p, float* ptr, int shared_weights) {
    int head_size = p->dim / p->n_heads;
    unsigned long long n_layers = p->n_layers;
    w->token_embedding_table = ptr;
    ptr += (unsigned long long)p->vocab_size * p->dim;
    w->rms_att_weight = ptr;
    ptr += n_layers * p->dim;
    w->wq = ptr;
    ptr += n_layers * p->dim * (p->n_heads * head_size);
    w->wk = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wv = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wo = ptr;
    ptr += n_layers * (p->n_heads * head_size) * p->dim;
    w->rms_ffn_weight = ptr;
    ptr += n_layers * p->dim;
    w->w1 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->w2 = ptr;
    ptr += n_layers * p->hidden_dim * p->dim;
    w->w3 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->rms_final_weight = ptr;
    ptr += p->dim;
    ptr += p->seq_len * head_size / 2;
    ptr += p->seq_len * head_size / 2;
    w->wcls = shared_weights ? w->token_embedding_table : ptr;
}

static QuantizedTensor* init_quantized_tensors(void** ptr, int n, int size_each, int group_size) {
    void* p = *ptr;
    QuantizedTensor* res = (QuantizedTensor*)malloc(n * sizeof(QuantizedTensor));
    for (int i = 0; i < n; i++) {
        res[i].q = (int8_t*)p;
        p = (int8_t*)p + size_each;
        res[i].s = (float*)p;
        p = (float*)p + size_each / group_size;
    }
    *ptr = p;
    return res;
}

static void memory_map_weights_q8(TransformerWeights* w, Config* p, void* ptr, uint8_t shared_classifier, int group_size) {
    int head_size = p->dim / p->n_heads;
    float* fptr = (float*)ptr;
    w->rms_att_weight = fptr; fptr += p->n_layers * p->dim;
    w->rms_ffn_weight = fptr; fptr += p->n_layers * p->dim;
    w->rms_final_weight = fptr; fptr += p->dim;

    ptr = (void*)fptr;
    w->q_tokens = init_quantized_tensors(&ptr, 1, p->vocab_size * p->dim, group_size);
    w->token_embedding_table = (float*)malloc(p->vocab_size * p->dim * sizeof(float));
    for (int i = 0; i < p->vocab_size * p->dim; i++) {
        w->token_embedding_table[i] = w->q_tokens[0].q[i] * w->q_tokens[0].s[i / group_size];
    }

    w->wq_q = init_quantized_tensors(&ptr, p->n_layers, p->dim * (p->n_heads * head_size), group_size);
    w->wk_q = init_quantized_tensors(&ptr, p->n_layers, p->dim * (p->n_kv_heads * head_size), group_size);
    w->wv_q = init_quantized_tensors(&ptr, p->n_layers, p->dim * (p->n_kv_heads * head_size), group_size);
    w->wo_q = init_quantized_tensors(&ptr, p->n_layers, (p->n_heads * head_size) * p->dim, group_size);

    w->w1_q = init_quantized_tensors(&ptr, p->n_layers, p->dim * p->hidden_dim, group_size);
    w->w2_q = init_quantized_tensors(&ptr, p->n_layers, p->hidden_dim * p->dim, group_size);
    w->w3_q = init_quantized_tensors(&ptr, p->n_layers, p->dim * p->hidden_dim, group_size);

    w->wcls_q = shared_classifier ? w->q_tokens : init_quantized_tensors(&ptr, 1, p->dim * p->vocab_size, group_size);
}

int build_transformer(Transformer* t, const char* checkpoint_path) {
    memset(t, 0, sizeof(Transformer));
    FILE* file = fopen(checkpoint_path, "rb");
    if (!file) return -1;

    fseek(file, 0, SEEK_END);
    t->file_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    if (t->file_size < 256) { fclose(file); return -2; }

    uint32_t magic = 0;
    fread(&magic, sizeof(uint32_t), 1, file);
    fseek(file, 0, SEEK_SET);

    if (magic == 0x616b3432) {
        // llama2.c v2 quantized format
        int version = 0;
        fread(&magic, sizeof(uint32_t), 1, file);
        fread(&version, sizeof(int), 1, file);
        fread(&t->config, sizeof(Config), 1, file);
        uint8_t shared_classifier = 0;
        int group_size = 64;
        fread(&shared_classifier, sizeof(uint8_t), 1, file);
        fread(&group_size, sizeof(int), 1, file);
        t->quantized = 1;
        t->shared_weights = shared_classifier;
        t->weights.group_size = group_size;

        long header_size = 256;
        long data_size = t->file_size - header_size;
        t->raw_buffer = malloc(data_size);
        if (!t->raw_buffer) { fclose(file); return -3; }
        fseek(file, header_size, SEEK_SET);
        fread(t->raw_buffer, 1, data_size, file);
        fclose(file);

        memory_map_weights_q8(&t->weights, &t->config, t->raw_buffer, shared_classifier, group_size);
        if (malloc_run_state(&t->state, &t->config, 1, group_size) != 0) return -4;
    } else {
        // llama2.c v1 float32 format
        fread(&t->config, sizeof(Config), 1, file);
        t->quantized = 0;
        t->shared_weights = t->config.vocab_size > 0 ? 1 : 0;
        t->config.vocab_size = abs(t->config.vocab_size);
        t->weights.group_size = 0;

        long header_size = sizeof(Config);
        long data_size = t->file_size - header_size;
        t->raw_buffer = malloc(data_size);
        if (!t->raw_buffer) { fclose(file); return -3; }
        fread(t->raw_buffer, 1, data_size, file);
        fclose(file);

        memory_map_weights_f32(&t->weights, &t->config, (float*)t->raw_buffer, t->shared_weights);
        if (malloc_run_state(&t->state, &t->config, 0, 0) != 0) return -4;
    }
    return 0;
}

void free_transformer(Transformer* t) {
    if (t->quantized) {
        free(t->weights.token_embedding_table);
        free(t->weights.q_tokens);
        free(t->weights.wq_q); free(t->weights.wk_q); free(t->weights.wv_q); free(t->weights.wo_q);
        free(t->weights.w1_q); free(t->weights.w2_q); free(t->weights.w3_q);
        if (!t->shared_weights) free(t->weights.wcls_q);
    }
    free(t->raw_buffer);
    free_run_state(&t->state);
    memset(t, 0, sizeof(Transformer));
}

static void rmsnorm(float* o, float* x, float* weight, int size) {
    float ss = 0.0f;
    for (int j = 0; j < size; j++) ss += x[j] * x[j];
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    for (int j = 0; j < size; j++) o[j] = weight[j] * (ss * x[j]);
}

static void softmax(float* x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < size; i++) x[i] /= sum;
}

static void matmul_f32(float* xout, float* x, float* w, int n, int d) {
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        float* wi = w + i * n;
        for (int j = 0; j < n; j++) val += wi[j] * x[j];
        xout[i] = val;
    }
}

static void quantize(QuantizedTensor* qx, float* x, int n, int GS) {
    int num_groups = n / GS;
    float Q_MAX = 127.0f;
    for (int group = 0; group < num_groups; group++) {
        float wmax = 0.0f;
        for (int i = 0; i < GS; i++) {
            float val = fabsf(x[group * GS + i]);
            if (val > wmax) wmax = val;
        }
        float scale = wmax / Q_MAX;
        qx->s[group] = scale;
        float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;
        for (int i = 0; i < GS; i++) {
            float quant_value = x[group * GS + i] * inv_scale;
            qx->q[group * GS + i] = (int8_t)roundf(quant_value);
        }
    }
}

static void matmul_q8(float* xout, QuantizedTensor* x, QuantizedTensor* w, int n, int d, int GS) {
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        int in = i * n;
        for (int j = 0; j <= n - GS; j += GS) {
            int32_t ival = 0;
            for (int k = 0; k < GS; k++) {
                ival += ((int32_t)x->q[j + k]) * ((int32_t)w->q[in + j + k]);
            }
            val += ((float)ival) * w->s[(in + j) / GS] * x->s[j / GS];
        }
        xout[i] = val;
    }
}

static float* forward(Transformer* transformer, int token, int pos) {
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    float* x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads;
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;
    int GS = w->group_size;

    memcpy(x, w->token_embedding_table + token * dim, dim * sizeof(float));

    for (int l = 0; l < p->n_layers; l++) {
        rmsnorm(s->xb, x, w->rms_att_weight + l * dim, dim);

        int loff = l * p->seq_len * kv_dim;
        s->k = s->key_cache + loff + pos * kv_dim;
        s->v = s->value_cache + loff + pos * kv_dim;

        if (transformer->quantized) {
            quantize(&s->xq, s->xb, dim, GS);
            matmul_q8(s->q, &s->xq, w->wq_q + l, dim, dim, GS);
            matmul_q8(s->k, &s->xq, w->wk_q + l, dim, kv_dim, GS);
            matmul_q8(s->v, &s->xq, w->wv_q + l, dim, kv_dim, GS);
        } else {
            matmul_f32(s->q, s->xb, w->wq + l * dim * dim, dim, dim);
            matmul_f32(s->k, s->xb, w->wk + l * dim * kv_dim, dim, kv_dim);
            matmul_f32(s->v, s->xb, w->wv + l * dim * kv_dim, dim, kv_dim);
        }

        for (int i = 0; i < dim; i += 2) {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            int rotn = i < kv_dim ? 2 : 1;
            for (int v = 0; v < rotn; v++) {
                float* vec = v == 0 ? s->q : s->k;
                float v0 = vec[i];
                float v1 = vec[i + 1];
                vec[i] = v0 * fcr - v1 * fci;
                vec[i + 1] = v0 * fci + v1 * fcr;
            }
        }

        for (int h = 0; h < p->n_heads; h++) {
            float* q = s->q + h * head_size;
            float* att = s->att + h * p->seq_len;
            for (int t = 0; t <= pos; t++) {
                float* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) score += q[i] * k[i];
                att[t] = score / sqrtf((float)head_size);
            }
            softmax(att, pos + 1);

            float* xb = s->xb + h * head_size;
            memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                float* v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                float a = att[t];
                for (int i = 0; i < head_size; i++) xb[i] += a * v[i];
            }
        }

        if (transformer->quantized) {
            quantize(&s->xq, s->xb, dim, GS);
            matmul_q8(s->xb2, &s->xq, w->wo_q + l, dim, dim, GS);
        } else {
            matmul_f32(s->xb2, s->xb, w->wo + l * dim * dim, dim, dim);
        }
        for (int i = 0; i < dim; i++) x[i] += s->xb2[i];

        rmsnorm(s->xb, x, w->rms_ffn_weight + l * dim, dim);

        if (transformer->quantized) {
            quantize(&s->xq, s->xb, dim, GS);
            matmul_q8(s->hb, &s->xq, w->w1_q + l, dim, hidden_dim, GS);
            matmul_q8(s->hb2, &s->xq, w->w3_q + l, dim, hidden_dim, GS);
        } else {
            matmul_f32(s->hb, s->xb, w->w1 + l * dim * hidden_dim, dim, hidden_dim);
            matmul_f32(s->hb2, s->xb, w->w3 + l * dim * hidden_dim, dim, hidden_dim);
        }

        for (int i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            val *= (1.0f / (1.0f + expf(-val)));
            val *= s->hb2[i];
            s->hb[i] = val;
        }

        if (transformer->quantized) {
            quantize(&s->hq, s->hb, hidden_dim, GS);
            matmul_q8(s->xb, &s->hq, w->w2_q + l, hidden_dim, dim, GS);
        } else {
            matmul_f32(s->xb, s->hb, w->w2 + l * dim * hidden_dim, hidden_dim, dim);
        }
        for (int i = 0; i < dim; i++) x[i] += s->xb[i];
    }

    rmsnorm(x, x, w->rms_final_weight, dim);
    if (transformer->quantized) {
        quantize(&s->xq, x, dim, GS);
        matmul_q8(s->logits, &s->xq, w->wcls_q, dim, p->vocab_size, GS);
    } else {
        matmul_f32(s->logits, x, w->wcls, dim, p->vocab_size);
    }
    return s->logits;
}

static int compare_tokens(const void* a, const void* b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

int build_tokenizer(Tokenizer* t, const char* tokenizer_path, int vocab_size) {
    memset(t, 0, sizeof(Tokenizer));
    t->vocab_size = vocab_size;
    t->vocab = (char**)malloc(vocab_size * sizeof(char*));
    t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));
    t->sorted_vocab = NULL;
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }
    FILE* file = fopen(tokenizer_path, "rb");
    if (!file) return -1;
    if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) { fclose(file); return -2; }
    int len;
    for (int i = 0; i < vocab_size; i++) {
        if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1) { fclose(file); return -3; }
        if (fread(&len, sizeof(int), 1, file) != 1) { fclose(file); return -4; }
        t->vocab[i] = (char*)malloc(len + 1);
        if (fread(t->vocab[i], len, 1, file) != 1) { fclose(file); return -5; }
        t->vocab[i][len] = '\0';
    }
    fclose(file);
    return 0;
}

void free_tokenizer(Tokenizer* t) {
    if (t->vocab) {
        for (int i = 0; i < t->vocab_size; i++) free(t->vocab[i]);
        free(t->vocab);
    }
    free(t->vocab_scores);
    free(t->sorted_vocab);
    memset(t, 0, sizeof(Tokenizer));
}

static char* decode(Tokenizer* t, int prev_token, int token) {
    char* piece = t->vocab[token];
    if (prev_token == 1 && piece[0] == ' ') piece++;
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = (char*)t->byte_pieces + byte_val * 2;
    }
    return piece;
}

static int str_lookup(char* str, TokenIndex* sorted_vocab, int vocab_size) {
    TokenIndex tok = { .str = str, .id = 0 };
    TokenIndex* res = (TokenIndex*)bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

static void encode(Tokenizer* t, const char* text, int8_t bos, int8_t eos, int* tokens, int* n_tokens) {
    if (t->sorted_vocab == NULL) {
        t->sorted_vocab = (TokenIndex*)malloc(t->vocab_size * sizeof(TokenIndex));
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }

    char* str_buffer = (char*)malloc((t->max_token_length * 2 + 3) * sizeof(char));
    size_t str_len = 0;
    *n_tokens = 0;
    if (bos) tokens[(*n_tokens)++] = 1;
    if (text[0] != '\0') {
        int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
        if (dummy_prefix != -1) tokens[(*n_tokens)++] = dummy_prefix;
    }

    for (const char* c = text; *c != '\0'; c++) {
        if ((*c & 0xC0) != 0x80) str_len = 0;
        str_buffer[str_len++] = *c;
        str_buffer[str_len] = '\0';
        if ((*(c + 1) & 0xC0) == 0x80 && str_len < 4) continue;
        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
        if (id != -1) {
            tokens[(*n_tokens)++] = id;
        } else {
            for (size_t i = 0; i < str_len; i++) {
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0;
    }

    while (1) {
        float best_score = -1e10f;
        int best_id = -1;
        int best_idx = -1;
        for (int i = 0; i < (*n_tokens - 1); i++) {
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i + 1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }
        if (best_idx == -1) break;
        tokens[best_idx] = best_id;
        for (int i = best_idx + 1; i < (*n_tokens - 1); i++) tokens[i] = tokens[i + 1];
        (*n_tokens)--;
    }

    if (eos) tokens[(*n_tokens)++] = 2;
    free(str_buffer);
}

static unsigned int random_u32(unsigned long long* state) {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1DULL) >> 32;
}

static float random_f32(unsigned long long* state) {
    return (random_u32(state) >> 8) / 16777216.0f;
}

static int sample_argmax(float* probabilities, int n) {
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) { max_i = i; max_p = probabilities[i]; }
    }
    return max_i;
}

static int sample_mult(float* probabilities, int n, float coin) {
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) return i;
    }
    return n - 1;
}

static int compare_prob(const void* a, const void* b) {
    ProbIndex* a_ = (ProbIndex*)a;
    ProbIndex* b_ = (ProbIndex*)b;
    if (a_->prob > b_->prob) return -1;
    if (a_->prob < b_->prob) return 1;
    return 0;
}

static int sample_topp(float* probabilities, int n, float topp, ProbIndex* probindex, float coin) {
    int n0 = 0;
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    qsort(probindex, n0, sizeof(ProbIndex), compare_prob);
    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1;
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) { last_idx = i; break; }
    }
    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) return probindex[i].index;
    }
    return probindex[last_idx].index;
}

static int sample(Sampler* s, float* logits) {
    if (s->temperature == 0.0f) return sample_argmax(logits, s->vocab_size);
    for (int q = 0; q < s->vocab_size; q++) logits[q] /= s->temperature;
    softmax(logits, s->vocab_size);
    float coin = random_f32(&s->rng_state);
    if (s->topp <= 0.0f || s->topp >= 1.0f) return sample_mult(logits, s->vocab_size, coin);
    return sample_topp(logits, s->vocab_size, s->topp, s->probindex, coin);
}

void build_sampler(Sampler* s, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
    s->vocab_size = vocab_size;
    s->temperature = temperature;
    s->topp = topp;
    s->rng_state = rng_seed;
    s->probindex = (ProbIndex*)malloc(vocab_size * sizeof(ProbIndex));
}

void free_sampler(Sampler* s) {
    free(s->probindex);
    s->probindex = NULL;
}

int generate_tokens(Transformer* t, Tokenizer* tok, Sampler* s, const char* prompt, int steps, TokenCallback cb, void* userdata) {
    const char* p = prompt ? prompt : "";
    int* prompt_tokens = (int*)malloc((strlen(p) + 4) * sizeof(int));
    int num_prompt_tokens = 0;
    encode(tok, p, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) { free(prompt_tokens); return -1; }

    int next = 0;
    int token = prompt_tokens[0];
    int pos = 0;
    int generated = 0;
    if (steps > t->config.seq_len) steps = t->config.seq_len;

    while (pos < steps) {
        float* logits = forward(t, token, pos);
        if (pos < num_prompt_tokens - 1) {
            next = prompt_tokens[pos + 1];
        } else {
            next = sample(s, logits);
        }
        pos++;
        if (next == 1) break;
        const char* piece = decode(tok, token, next);
        if (cb) cb(piece, pos, userdata);
        generated++;
        token = next;
    }
    free(prompt_tokens);
    return generated;
}
