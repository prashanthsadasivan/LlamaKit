//
//  Header.h
//  
//
//  Created by Prashanth Sadasivan on 6/28/24.
//
#import <Foundation/Foundation.h>
#include "llama.h"
@interface SamplingWrapper : NSObject

// Initializers
- (instancetype)initWithLlamaCtx:(struct llama_context*)llamaCtx;
- (void)resetSamplingContext;
- (void)freeSamplingContext;


// Additional Functionalities
- (NSArray<NSNumber *> *)tokenizeText:(NSString *)text addSpecial:(BOOL)addSpecial parseSpecial:(BOOL)parseSpecial;
- (NSString *)tokenToPieceWithToken:(NSNumber *)token;
- (BOOL)evaluateTokens:(NSArray<NSNumber *> *)tokens batchSize:(NSInteger)batchSize;
- (BOOL)evaluateString:(NSString *)string batchSize:(NSInteger)batchSize addBos:(BOOL)addBos;
- (NSString *)sampleAndEvaluate;
- (NSString *)sample;
- (void)accept:(llama_token)theId;
- (void)reverse:(NSString *)bad;

@end
