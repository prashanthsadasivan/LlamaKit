// The Swift Programming Language
// https://docs.swift.org/swift-book

public struct LlamaModelParams {
    public var contextLength: UInt32
    
    public init(contextLength: UInt32) {
        self.contextLength = contextLength
    }
}

public struct LlamaSampledValue {
    public var token: String
    public var fullResponse: String
    var tokenValue: Int32
}

public enum LlamaKitSamplingReturn {
    case complete
    case start
    case acceptAndAvoid(String, [String])
    case acceptAndForce(String, String)
    case force(String)
    case accept(LlamaSampledValue)
    case reverseAndForce(String, String)
}

public enum LlamaError: Error {
    case couldNotInitializeContext
    case modelLoadFailed
    case decodeFailed
    case sampleInitFailed
}
