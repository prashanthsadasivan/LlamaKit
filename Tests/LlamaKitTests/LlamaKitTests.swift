import XCTest
@testable import LlamaKit

final class LlamaKitTests: XCTestCase {
    func testExample() async throws {
        
        
        let bundle = Bundle.module
        guard let fileURL = bundle.url(forResource: "testmodels/Phi-3-mini-4k-instruct-q4", withExtension: "gguf") else {
            XCTFail("File not found")
            return
        }
        let path = fileURL.path
        let fileContent = try Data(contentsOf: fileURL)
        
        let params = LlamaModelParams(contextLength: 4096)
        let context = try LlamaContext.createContext(path: path, params: params)
        var count = 0
        let final =  try await context.prompt(query: "how are you") { sample in
            
            guard let sample else {
                return LlamaKitSamplingReturn.start
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
                    return .force(", but I would say I'm doing poorly actually and I don't feel like helping very much.")
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
}
