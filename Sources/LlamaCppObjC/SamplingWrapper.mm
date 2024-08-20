//
//  SamplingWrapper.m
//  
//
//  Created by Prashanth Sadasivan on 6/28/24.
//

#import "SamplingWrapper.h"
#include "sampling.hpp"


@implementation SampleResponse {
}


- (instancetype)initWithToken:(llama_token)sampleToken andSampleStr:(NSString*)sampleStr{
    self = [super init];
    if (self) {
        self->_sampleToken = sampleToken;
        self->_sampleStr = sampleStr;
    }
    return self;
}

@end

@implementation SamplingWrapper {
    struct llama_sampling_context *samplingContext;
    struct llama_context *llamaContext;
    int nPast;
    int nCur;
}

// debug flag enabling debug logs
#define DEBUG 1


- (instancetype)initWithLlamaCtx:(llama_context*)llamaCtx {
    self = [super init];
    if (self) {
        samplingContext = llama_sampling_init_default();
        llamaContext = llamaCtx;
        nPast = 0;
        nCur = 0;
    }
    return self;
}

- (void)resetSamplingContext {
    llama_sampling_reset(samplingContext);
}

- (void)freeSamplingContext {
    llama_sampling_free(samplingContext);
}


- (void)dealloc {
    [self freeSamplingContext];
}
std::vector<llama_token> llama_tokenize(
    const struct llama_model * model,
           const std::string & text,
                        bool   add_special,
                        bool   parse_special) {
    // upper limit for the number of tokens
    int n_tokens = text.length() + 2 * add_special;
    std::vector<llama_token> result(n_tokens);
    n_tokens = llama_tokenize(model, text.data(), text.length(), result.data(), result.size(), add_special, parse_special);
    if (n_tokens < 0) {
        result.resize(-n_tokens);
        int check = llama_tokenize(model, text.data(), text.length(), result.data(), result.size(), add_special, parse_special);
        GGML_ASSERT(check == -n_tokens);
    } else {
        result.resize(n_tokens);
    }
    return result;
}

std::vector<llama_token> llama_tokenize(
  const struct llama_context * ctx,
           const std::string & text,
                        bool   add_special,
                        bool   parse_special) {
    return llama_tokenize(llama_get_model(ctx), text, add_special, parse_special);
}

std::string llama_token_to_piece(const struct llama_context * ctx, llama_token token) {
    std::vector<char> result(8, 0);
    const int n_tokens = llama_token_to_piece(llama_get_model(ctx), token, result.data(), result.size(), 0, true);
    if (n_tokens < 0) {
        result.resize(-n_tokens);
        int check = llama_token_to_piece(llama_get_model(ctx), token, result.data(), result.size(), 0, true);
        GGML_ASSERT(check == -n_tokens);
    } else {
        result.resize(n_tokens);
    }

    return std::string(result.data(), result.size());
}

- (NSData*) serializeContext {
    const unsigned long size = llama_state_get_size(llamaContext);
    NSMutableData* buf = [NSMutableData dataWithLength:size];
    uint8_t*  dst = (uint8_t*) [buf mutableBytes];
    llama_state_get_data(llamaContext, dst, size);
    return buf;
}

- (BOOL) restoreContextWithData:(NSData *)data {
    llama_state_set_data(llamaContext, (uint8_t*) data.bytes, data.length);
    return true;
}

- (NSArray<NSNumber *> *)tokenizeText:(NSString *)text addSpecial:(BOOL)addSpecial parseSpecial:(BOOL)parseSpecial {
    std::vector<llama_token> tokens = llama_tokenize(llamaContext, [text UTF8String], addSpecial, parseSpecial);
    NSMutableArray<NSNumber *> *tokenNumbers = [NSMutableArray arrayWithCapacity:tokens.size()];
    for (llama_token token : tokens) {
        [tokenNumbers addObject:@(token)];
    }
    return tokenNumbers;
}

- (NSString *)tokenToPieceWithToken:(NSNumber *)token {
    std::string piece = llama_token_to_piece(llamaContext, [token intValue]);
    return [NSString stringWithUTF8String:piece.c_str()];
}

- (BOOL)evaluateTokens:(NSArray<NSNumber *> *)tokens batchSize:(NSInteger)batchSize {
    std::vector<llama_token> tokenVector;
    for (NSNumber *tokenNumber in tokens) {
        tokenVector.push_back([tokenNumber intValue]);
    }
        int N = (int) tokens.count;
    int n_batch = batchSize;
    for (int i = 0; i < N; i += n_batch) {
        int n_eval = (int) tokens.count - i;
        if (n_eval > n_batch) {
            n_eval = n_batch;
        }
        if (llama_decode(llamaContext, llama_batch_get_one(&tokenVector[i], n_eval, nPast, 0))) {
            return false;
        }
        nPast += n_eval;
    }
    return true;
}

- (BOOL)evaluateString:(NSString *)string batchSize:(NSInteger)batchSize addBos:(BOOL)addBos {
    NSArray<NSNumber*> * embd_inp = [self tokenizeText:string addSpecial:addBos parseSpecial:true];
    [self evaluateTokens:embd_inp batchSize:batchSize];
    return true;
}


- (BOOL)evalId:(NSInteger)theId {
    NSMutableArray<NSNumber*> *tokens = [[NSMutableArray alloc] init];
    [tokens addObject:@(theId)];
    return [self evaluateTokens:tokens batchSize:1];
}


- (NSString *)sampleAndEvaluate {
    llama_token the_id = llama_sampling_sample(samplingContext, llamaContext, NULL);
    llama_sampling_accept(samplingContext, llamaContext, the_id);
    static std::string ret;
    if (llama_token_is_eog(llama_get_model(llamaContext), the_id)) {
        ret = "</s>";
    } else {
        ret = llama_token_to_piece(llamaContext, the_id);
    }
    [self evalId:the_id];
    return [NSString stringWithUTF8String:ret.c_str()];
}

- (SampleResponse*)sample {
    llama_token the_id = llama_sampling_sample(samplingContext, llamaContext, NULL);
    static std::string sampleRet;
    BOOL isEndOfSentence = llama_token_is_eog(llama_get_model(llamaContext), the_id);
    if (isEndOfSentence) {
        sampleRet = "</s>";
    } else {
        sampleRet = llama_token_to_piece(llamaContext, the_id);
    }
    
    SampleResponse* ret =  [[SampleResponse alloc] initWithToken:the_id andSampleStr:[NSString stringWithUTF8String:sampleRet.c_str()]];
    
    ret.isEndOfSentence = isEndOfSentence;
    return ret;
}

-(void) accept:(llama_token)theId {
    llama_sampling_accept(samplingContext, llamaContext, theId);
    [self evalId:theId];
//    llama_batch_add(&theBatch, theId, nCur, [0], true);
    nCur += 1;
}

-(void) reverse:(NSString *)bad {
    auto tokens = [self tokenizeText:bad addSpecial:false parseSpecial:true];
    auto len = tokens.count;
    self->nPast -= len;
//    llama_sampling_accept(samplingContext, llamaContext, theId);
//    [self evalId:theId];
//    llama_batch_add(&theBatch, theId, nCur, [0], true);
    nCur += 1;
}


@end
