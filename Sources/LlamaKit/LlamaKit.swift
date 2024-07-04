// The Swift Programming Language
// https://docs.swift.org/swift-book


public struct LlamaModelParams {
    public var contextLength: UInt32
}
public enum LlamaKitSamplingReturn {
    case complete
    case start
    case acceptAndAvoid(String, [String])
    case acceptAndForce(String, String)
    case accept(String)
    case reverseAndForce(String, String)
}

public enum LlamaError: Error {
    case couldNotInitializeContext
    case modelLoadFailed
    case decodeFailed
    case sampleInitFailed
}
