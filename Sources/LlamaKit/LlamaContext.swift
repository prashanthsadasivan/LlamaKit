//
//  File.swift
//
//
//  Created by Prashanth Sadasivan on 6/28/24.
//

import Foundation
import llama
import LlamaCppObjC


func llama_batch_clear(_ batch: inout llama_batch) {
    batch.n_tokens = 0
}

func llama_batch_add(_ batch: inout llama_batch, _ id: llama_token, _ pos: llama_pos, _ seq_ids: [llama_seq_id], _ logits: Bool) {
    batch.token   [Int(batch.n_tokens)] = id
    batch.pos     [Int(batch.n_tokens)] = pos
    batch.n_seq_id[Int(batch.n_tokens)] = Int32(seq_ids.count)
    for i in 0..<seq_ids.count {
        batch.seq_id[Int(batch.n_tokens)]![Int(i)] = seq_ids[i]
    }
    batch.logits  [Int(batch.n_tokens)] = logits ? 1 : 0
    
    batch.n_tokens += 1
}

private actor SamplingWrapperActor {
    private var samplingWrapper: SamplingWrapper
    init(samplingWrapper: SamplingWrapper) {
        self.samplingWrapper = samplingWrapper
    }
    
    func accept(token: llama_token) {
        samplingWrapper.accept(token)
    }
    
    func sample() -> SampleResponse {
        return samplingWrapper.sample()!
    }
    
    func evaluateString(prompt: String) {
        samplingWrapper.evaluateString(prompt, batchSize: 1, addBos: false)
    }
    
    func evaluateTokens(tokens: [llama_token]) {
        samplingWrapper.evaluateTokens(tokens.map { t in
            NSNumber(value: t)
        }, batchSize: 1)
    }
    
    func serializeContext(includeSampler: Bool) -> Data {
        return samplingWrapper.serializeContextIncludeSampler(includeSampler);
    }
    
    func restoreContext(data: Data) {
        samplingWrapper.restoreContext(with: data);
    }
    
    func reverse(bad: String) {
        samplingWrapper.reverse(bad)
    }
    
    func clear() {
        samplingWrapper.clear()
    }
    
}

public actor LlamaContext {
    private var llamaModel: OpaquePointer
    private var context: OpaquePointer
    private var nCur: Int32 = 0
    private var wrapper: SamplingWrapperActor?
    
    init(llamaModel: OpaquePointer, context: OpaquePointer) {
        self.llamaModel = llamaModel
        self.context = context
        self.wrapper = SamplingWrapperActor(samplingWrapper: SamplingWrapper(llamaCtx: context))
        // TODO - use batching?
    }
    
    deinit {
        llama_free_model(self.llamaModel)
        llama_free(self.context)
        llama_backend_free()
    }
    
    public static func createContext(path:String, params: LlamaModelParams) throws -> LlamaContext {
        llama_backend_init()
        let modelParams = llama_model_default_params()
        let model = llama_load_model_from_file(path, modelParams)
        
        guard let model else {
            throw LlamaError.modelLoadFailed
        }
        
        
        let nThreads = max(1, min(8, ProcessInfo.processInfo.processorCount - 2))
        var contextParams = llama_context_default_params()
        contextParams.seed = UInt32.random(in: 0..<UINT32_MAX)
        contextParams.n_ctx = params.contextLength
        contextParams.n_threads = UInt32(nThreads)
        contextParams.n_threads_batch = UInt32(nThreads)
        
        
        let context = llama_new_context_with_model(model, contextParams)
        
        guard let context else {
            throw LlamaError.couldNotInitializeContext
        }
        
        return LlamaContext(llamaModel: model, context: context)
    }
    
    
    private func tokenize(text: String, add_bos: Bool) -> [llama_token] {
        let utf8Count = text.utf8.count
        let n_tokens = utf8Count + (add_bos ? 1 : 0) + 1
        let tokens = UnsafeMutablePointer<llama_token>.allocate(capacity: n_tokens)
        let tokenCount = llama_tokenize(self.llamaModel, text, Int32(utf8Count), tokens, Int32(n_tokens), add_bos, false)
        
        var swiftTokens: [llama_token] = []
        
        for i in 0..<tokenCount {
            swiftTokens.append(tokens[Int(i)])
        }
        
        tokens.deallocate()
        
        return swiftTokens
    }
    
    private func chatifyAndTokenizeQuery(query: String) throws -> [llama_token] {
        var cRole: UnsafePointer<CChar>?
        var cPrompt: UnsafePointer<CChar>?
        "user".withCString { role in
            query.withCString { prompt in
                cRole = UnsafePointer(strdup(role))
                cPrompt = UnsafePointer(strdup(prompt))
            }
        }
        let user_chat = llama_chat_message(role: cRole, content: cPrompt)
        
        
        let chats = [user_chat]
        let maxLen = 1200 + query.lengthOfBytes(using: .utf8)
        let formatted = UnsafeMutablePointer<CChar>.allocate(capacity: maxLen)
        let formatResult = llama_chat_apply_template(self.llamaModel, UnsafePointer(bitPattern: 0), chats  , 1, true, formatted, Int32(maxLen))
        if formatResult < 0 {
            throw LlamaError.sampleInitFailed
        }
        let prompt = String(cString: formatted)
        let promptTokens = tokenize(text: prompt, add_bos: true)
        let modelContextLen = llama_n_ctx(self.context)
        let nLen = 64 // TODO - what is this.
        let nKVReq = promptTokens.count + (Int(nLen) - promptTokens.count)
        
        
        if nKVReq > modelContextLen {
            print("error: n_kv_req > n_ctx, the required KV cache size is not big enough")
            print("\n n_len = \(nLen), n_ctx = \(modelContextLen), n_kv_req = \(nKVReq)")
        }
        return promptTokens
    }
    
    public func savePromptState(prompt: String, includeSampler: Bool) async throws -> Data {
        guard let wrapper else {
            throw LlamaError.modelLoadFailed
        }
        let promptTokens = try chatifyAndTokenizeQuery(query: prompt)
        await wrapper.evaluateTokens(tokens: promptTokens)
        return await wrapper.serializeContext(includeSampler: includeSampler);
    }
    
    public func restorePromptState(context: Data) async throws {
        guard let wrapper else {
            throw LlamaError.modelLoadFailed
        }
        
        await wrapper.restoreContext(data: context)
    }
    
    
    public func clear() async throws {
        
        guard let wrapper else {
            throw LlamaError.modelLoadFailed
        }
        
        await wrapper.clear()
    }
    
    public func prompt(query: String, callback : (LlamaSampledValue?) -> LlamaKitSamplingReturn) async throws -> String {
        
        guard let wrapper else {
            throw LlamaError.modelLoadFailed
        }
        
        let promptTokens = try chatifyAndTokenizeQuery(query: query);
        await wrapper.evaluateTokens(tokens: promptTokens)
        var callbackResult = LlamaKitSamplingReturn.start
        var ret = ""
        var strings = Array<String>()
        while true {
            if case .complete = callbackResult {
                break
            }
            var sample: LlamaSampledValue? = nil
            switch callbackResult {
            case .accept(let sample):
                ret += sample.token
                await wrapper.accept(token: sample.tokenValue)
            case .force(let str):
                let tokens = tokenize(text: str, add_bos: false)
                for t in tokens {
                    await wrapper.accept(token: t)
                }
                ret += " \(str)"
                strings.append(str)
            case .acceptAndAvoid(let str, let avoid):
                let tokens = tokenize(text: str, add_bos: false)
                for t in tokens {
                    await wrapper.accept(token: t)
                }
                //TODO: avoid
                ret += str
                strings.append(str)
            case .acceptAndForce(let str, let forced):
                let tokens = tokenize(text: str, add_bos: false)
                for t in tokens {
                    await wrapper.accept(token: t)
                }
                ret += str
                strings.append(str)
                let next = tokenize(text: forced, add_bos: false)
                for t in next {
                    await wrapper.accept(token: t)
                }
                ret += forced
                strings.append(forced)
            case .reverseAndForce(let bad, let forced):
                await wrapper.reverse(bad: bad)
                ret = ret.replacingOccurrences(of: bad, with: forced)
                
                let next = tokenize(text: forced, add_bos: false)
                for t in next {
                    await wrapper.accept(token: t)
                }
                
            case .start: break
                // do nothing, sample as normal
                
            case .complete:
                return ret
            }
            let r = await wrapper.sample()
            sample = LlamaSampledValue(token: r.sampleStr, fullResponse: ret, tokenValue: Int32(r.sampleToken), isEoS: r.isEndOfSentence)
            
            callbackResult = callback(sample)
        }
        return ret
    }
}
