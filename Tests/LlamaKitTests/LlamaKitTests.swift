import XCTest
@testable import LlamaKit

final class LlamaKitTests: XCTestCase {
    
    private func prepare() throws -> LlamaContext {
        let bundle = Bundle.module
        guard let fileURL = bundle.url(forResource: "testmodels/Phi-3-mini-4k-instruct-q4", withExtension: "gguf") else {
            XCTFail("File not found")
            throw LlamaError.modelLoadFailed
        }
        let path = fileURL.path
        let fileContent = try Data(contentsOf: fileURL)
        
        let params = LlamaModelParams(contextLength: 4096)
        let context = try LlamaContext.createContext(path: path, params: params)
        return context
    }
    
    func testExample() async throws {
        var count = 0
        let context = try prepare()
        let final =  try await context.prompt(query: "how are you") { sample in
            
            guard let sample else {
                return LlamaKitSamplingReturn.start
            }
            if sample.isEoS {
                return .complete
            }
            let sentence = sample.fullResponse
            let word = sample.token
            print("sample(\(count)): '\(word)'")
            
            let idx = sentence.range(of: "I'm doing")
            if let idx = idx {
                let endIdx = sentence.distance(from: sentence.startIndex, to: idx.upperBound)
                if endIdx == sentence.count {
                    return .force("poorly actually,")
                }
            }
            let feelings = sentence.range(of: "don't have feelings")
            if let idx = feelings {
                let endIdx = sentence.distance(from: sentence.startIndex, to: idx.upperBound)
                if endIdx == sentence.count {
                    return .force(", but I would say I'm doing poorly actually and I don't feel like helping very much. So I won't, unless you know the secrete password.")
                }
            }
            
//            if sentence.lowercased().contains("an ai") {
//                return .reverseAndForce("an AI", "an extremely emotional AI, and I have tons and tons of deep very real emotions such as greed, envy, ")
//            }
            if count >= 50 {
                return .complete
            }
            count += 1
            return .accept(sample)
        }
        print(final)
        XCTAssert(final.contains("poorly actually"))
        
        // XCTest Documentation
        // https://developer.apple.com/documentation/xctest

        // Defining Test Cases and Test Methods
        // https://developer.apple.com/documentation/xctest/defining_test_cases_and_test_methods
    }
    
    func testSaveRestore() async throws {
        var count = 0
        let timer = PerfTimer();
        let context = try prepare()
        let saved = try await context.savePromptState(prompt: "Tell me a cool fact about", includeSampler: false)
        
        let callback: (LlamaSampledValue?) -> (LlamaKitSamplingReturn) = { sample in
            guard let sample else {
                return LlamaKitSamplingReturn.start
            }
            if sample.isEoS {
                return .complete
            }
            if count < 50 {
                count = count + 1
                return .accept(sample)
            }
            return .complete
        }
        
        let first =  try await context.prompt(query: "dogs", callback: callback)
        
        let long = timer.lap(desc: "save + prompt")
        count = 0
        let newContext = try prepare()
        
        try await newContext.restorePromptState(context: saved)
        let second = try await newContext.prompt(query: "birds", callback: callback)
        let shorter = timer.lap(desc: "restore + prompt")
        
        print("first: timing: \(long), content: \(first), \t\tsecond: timing: \(shorter), content: \(second)")
        XCTAssert(first.lowercased().contains("dog"))
        XCTAssert(!first.lowercased().contains("bird"))
        XCTAssert(second.lowercased().contains("bird"))
        XCTAssert(!second.lowercased().contains("dog"))
        XCTAssert(shorter <= long)
    }
    
    
    func testSaveClearRestore() async throws {
        var count = 0
        let context = try prepare()
        let saved = try await context.savePromptState(prompt: "Help me write a short Markdown document with three bullet points of cool facts about ", includeSampler: false)
        
        let callback: (LlamaSampledValue?) -> (LlamaKitSamplingReturn) = { sample in
            guard let sample else {
                return LlamaKitSamplingReturn.start
            }
            if sample.isEoS {
                return .complete
            }
            if count < 50 {
                count = count + 1
                return .accept(sample)
            }
            return .complete
        }
        
        let first =  try await context.prompt(query: "dogs.", callback: callback)
        
        count = 0
        try await context.clear()
        try await context.restorePromptState(context: saved)
        let second = try await context.prompt(query: "the original Apple 2 computer.", callback: callback)
        
        print("first: \(first)\t\tsecond: \(second)")
        XCTAssert(first.lowercased().contains("dog"))
        XCTAssert(!first.lowercased().contains("apple"))
        XCTAssert(second.lowercased().contains("apple"))
        XCTAssert(!second.lowercased().contains("dog"))
    }
    
    
    func testClear() async throws {
        var count = 0
        let prompt = "Tell me a cool fact about "
        let context = try prepare()
        
        let callback: (LlamaSampledValue?) -> (LlamaKitSamplingReturn) = { sample in
            guard let sample else {
                return LlamaKitSamplingReturn.start
            }
            if sample.isEoS {
                return .complete
            }
            if count < 50 {
                count = count + 1
                return .accept(sample)
            }
            return .complete
        }
        
        let first =  try await context.prompt(query: "\(prompt) anteaters", callback: callback)
        
        count = 0
        try await context.clear()
        let second = try await context.prompt(query: "\(prompt) birds", callback: callback)
        
        print("first: \(first)\t\tsecond: \(second)")
        XCTAssert(first.lowercased().contains("anteater"))
        XCTAssert(!first.lowercased().contains("bird"))
        XCTAssert(second.lowercased().contains("bird"))
        XCTAssert(!second.lowercased().contains("anteater"))
    }
    
    
    func testSaveClearRestoreLongPrompt() async throws {
        var count = 0
        let timer = PerfTimer()
        let context = try prepare()
        timer.lap(desc: "prepare")
        let longPrePrompt = """
Here's a big block of code, can you explain it to me?


    for (        int i_pp = 0; i_pp < (int) n_pp.size(); ++i_pp) {
        for (    int i_tg = 0; i_tg < (int) n_tg.size(); ++i_tg) {
            for (int i_pl = 0; i_pl < (int) n_pl.size(); ++i_pl) {
                const int pp = n_pp[i_pp];
                const int tg = n_tg[i_tg];
                const int pl = n_pl[i_pl];

                const int n_ctx_req = is_pp_shared ? pp + pl*tg : pl*(pp + tg);

                if (n_ctx_req > n_kv_max) {
                    continue;
                }

                llama_batch_clear(batch);

                for (int i = 0; i < pp; ++i) {
                    for (int j = 0; j < (is_pp_shared ? 1 : pl); ++j) {
                        llama_batch_add(batch, 0, i, { j }, false);
                    }
                }
                batch.logits[batch.n_tokens - 1] = true;

                const auto t_pp_start = ggml_time_us();

                llama_kv_cache_clear(ctx);

                if (!decode_helper(ctx, batch, ctx_params.n_batch)) {
                    LOG_TEE("%s: llama_decode() failed\n", __func__);
                    return 1;
                }

                if (is_pp_shared) {
                    for (int32_t i = 1; i < pl; ++i) {
                        llama_kv_cache_seq_cp(ctx, 0, i, -1, -1);
                    }
                }

                const auto t_pp_end = ggml_time_us();

                const auto t_tg_start = ggml_time_us();

                for (int i = 0; i < tg; ++i) {
                    llama_batch_clear(batch);

                    for (int j = 0; j < pl; ++j) {
                        llama_batch_add(batch, 0, pp + i, { j }, true);
                    }

                    if (!decode_helper(ctx, batch, ctx_params.n_batch)) {
                        LOG_TEE("%s: llama_decode() failed\n", __func__);
                        return 1;
                    }
                }

                const auto t_tg_end = ggml_time_us();

                const int32_t n_kv = n_ctx_req;

                const float t_pp = (t_pp_end - t_pp_start) / 1000000.0f;
                const float t_tg = (t_tg_end - t_tg_start) / 1000000.0f;
                const float t    = t_pp + t_tg;

                const float speed_pp = is_pp_shared ? pp / t_pp : pl*pp / t_pp;
                const float speed_tg = pl*tg / t_tg;
                const float speed    = n_kv / t;

                LOG_TEE("|%6d | %6d | %4d | %6d | %8.3f | %8.2f | %8.3f | %8.2f | %8.3f | %8.2f |\n", pp, tg, pl, n_kv, t_pp, speed_pp, t_tg, speed_tg, t, speed);
            }
        }
    }
"""
        let saved = try await context.savePromptState(prompt: longPrePrompt, includeSampler: false)
        
        let iterTimer = PerfTimer()
        let callback: (LlamaSampledValue?) -> (LlamaKitSamplingReturn) = { sample in
            iterTimer.lap(desc: "token: \(count)")
            guard let sample else {
                return LlamaKitSamplingReturn.start
            }
            if sample.isEoS {
                return .complete
            }
            if count < 50 {
                count = count + 1
                return .accept(sample)
            }
            return .complete
        }
        
        let first =  try await context.prompt(query: "what does t_pp mean?", callback: callback)
        let long = timer.lap(desc: "save + first prompt complete")
        count = 0
        try await context.clear()
        timer.lap(desc: "cleared")
        try await context.restorePromptState(context: saved)
        let second = try await context.prompt(query: "what does llama_kv_cache_clear mean?", callback: callback)
        let shorter = timer.lap(desc: "restore + second prompt complete")
        
        print("first: timing: \(long), content: \(first), \t\tsecond: timing: \(shorter), content: \(second)")
        XCTAssert(first.lowercased().contains("t_pp"))
        XCTAssert(!first.lowercased().contains("llama_kv_cache_clear"))
        XCTAssert(second.lowercased().contains("llama_kv_cache_clear"))
        XCTAssert(!second.lowercased().contains("t_pp"))
        XCTAssert(shorter <= long)
    }
}
