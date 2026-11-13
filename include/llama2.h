#ifndef LLAMA2_H
#define LLAMA2_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int seq_len;
} Config;

typedef struct {
    int8_t* q;
    float* s;
} QuantizedTensor;

typedef struct {
    // float32 weights (when quantized == 0)
    float* token_embedding_table;
    float* rms_att_weight;
    float* rms_ffn_weight;
    float* wq;
    float* wk;
    float* wv;
    float* wo;
    float* w1;
    float* w2;
    float* w3;
    float* rms_final_weight;
    float* wcls;

    // int8 quantized weights (when quantized == 1)
    QuantizedTensor* q_tokens;
    QuantizedTensor* wq_q;
    QuantizedTensor* wk_q;
    QuantizedTensor* wv_q;
    QuantizedTensor* wo_q;
    QuantizedTensor* w1_q;
    QuantizedTensor* w2_q;
    QuantizedTensor* w3_q;
    QuantizedTensor* wcls_q;
    int group_size;
} TransformerWeights;

typedef struct {
    float* x;
    float* xb;
    float* xb2;
    float* hb;
    float* hb2;
    float* q;
    float* k;
    float* v;
    float* att;
    float* logits;
    float* key_cache;
    float* value_cache;
    QuantizedTensor xq;
    QuantizedTensor hq;
} RunState;

typedef struct {
    Config config;
    TransformerWeights weights;
    RunState state;
    int quantized;
    int shared_weights;
    void* raw_buffer;
    long file_size;
} Transformer;

typedef struct {
    char* str;
    int id;
} TokenIndex;

typedef struct {
    char** vocab;
    float* vocab_scores;
    TokenIndex* sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512];
} Tokenizer;

typedef struct {
    float prob;
    int index;
} ProbIndex;

typedef struct {
    int vocab_size;
    ProbIndex* probindex;
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

typedef void (*TokenCallback)(const char* piece, int step, void* userdata);

int build_transformer(Transformer* t, const char* checkpoint_path);
void free_transformer(Transformer* t);
int build_tokenizer(Tokenizer* t, const char* tokenizer_path, int vocab_size);
void free_tokenizer(Tokenizer* t);
void build_sampler(Sampler* s, int vocab_size, float temperature, float topp, unsigned long long rng_seed);
void free_sampler(Sampler* s);
int generate_tokens(Transformer* t, Tokenizer* tok, Sampler* s, const char* prompt, int steps, TokenCallback cb, void* userdata);

#ifdef __cplusplus
}
#endif

#endif
