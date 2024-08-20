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
        
        print("first: timing: \(long), content: \(first), \t\tsecond: timing: \(long), content: \(second)")
        XCTAssert(first.lowercased().contains("dogs"))
        XCTAssert(!first.lowercased().contains("birds"))
        XCTAssert(second.lowercased().contains("birds"))
        XCTAssert(!second.lowercased().contains("dogs"))
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
        let second = try await context.prompt(query: "\(prompt) birds and the previous animal i asked you about", callback: callback)
        
        print("first: \(first)\t\tsecond: \(second)")
        XCTAssert(first.lowercased().contains("anteater"))
        XCTAssert(!first.lowercased().contains("bird"))
        XCTAssert(second.lowercased().contains("bird"))
        XCTAssert(!second.lowercased().contains("anteater"))
    }
}
