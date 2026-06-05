import Foundation
import Vision
import ImageIO
import CoreGraphics

struct OCRLine: Codable {
    let text: String
    let bbox: [Int]
}

struct OCRPayload: Codable {
    let text: String
    let quality: String
    let layout: String
    let lines: [OCRLine]
    let warnings: [String]
}

enum VisionOCRError: Error {
    case missingImagePath
    case failedToLoadImage
}

func loadImage(at path: String) throws -> CGImage {
    let url = URL(fileURLWithPath: path) as CFURL
    guard
        let source = CGImageSourceCreateWithURL(url, nil),
        let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
    else {
        throw VisionOCRError.failedToLoadImage
    }
    return image
}

func normalizeText(_ text: String) -> String {
    text
        .replacingOccurrences(of: "\r\n", with: "\n")
        .replacingOccurrences(of: "\r", with: "\n")
        .trimmingCharacters(in: .whitespacesAndNewlines)
}

func makePayload(from observations: [VNRecognizedTextObservation]) -> OCRPayload {
    typealias LineBox = (text: String, bbox: [Int], confidence: Float)
    var lineBoxes: [LineBox] = []

    for observation in observations {
        guard let candidate = observation.topCandidates(1).first else {
            continue
        }

        let text = normalizeText(candidate.string)
        if text.isEmpty {
            continue
        }

        let box = observation.boundingBox
        let left = Int((box.minX * 1000).rounded())
        let top = Int(((1 - box.maxY) * 1000).rounded())
        let right = Int((box.maxX * 1000).rounded())
        let bottom = Int(((1 - box.minY) * 1000).rounded())
        lineBoxes.append((text: text, bbox: [left, top, right, bottom], confidence: candidate.confidence))
    }

    lineBoxes.sort { lhs, rhs in
        if lhs.bbox[1] != rhs.bbox[1] {
            return lhs.bbox[1] < rhs.bbox[1]
        }
        return lhs.bbox[0] < rhs.bbox[0]
    }

    let joinedText = lineBoxes.map(\.text).joined(separator: "\n")
    let averageConfidence: Float
    if lineBoxes.isEmpty {
        averageConfidence = 0
    } else {
        averageConfidence = lineBoxes.reduce(0) { $0 + $1.confidence } / Float(lineBoxes.count)
    }

    let quality: String
    if lineBoxes.isEmpty || averageConfidence < 0.45 {
        quality = "low"
    } else if averageConfidence < 0.78 {
        quality = "medium"
    } else {
        quality = "high"
    }

    return OCRPayload(
        text: joinedText,
        quality: quality,
        layout: "plain_text",
        lines: lineBoxes.map { OCRLine(text: $0.text, bbox: $0.bbox) },
        warnings: lineBoxes.isEmpty ? ["no_text_detected"] : []
    )
}

func performOCR(imagePath: String) throws -> OCRPayload {
    let image = try loadImage(at: imagePath)
    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate
    request.recognitionLanguages = ["zh-Hans", "en-US"]
    request.usesLanguageCorrection = false
    if #available(macOS 13.0, *) {
        request.automaticallyDetectsLanguage = false
    }

    let handler = VNImageRequestHandler(cgImage: image, options: [:])
    try handler.perform([request])
    return makePayload(from: request.results ?? [])
}

do {
    guard CommandLine.arguments.count >= 2 else {
        throw VisionOCRError.missingImagePath
    }

    let payload = try performOCR(imagePath: CommandLine.arguments[1])
    let data = try JSONEncoder().encode(payload)
    FileHandle.standardOutput.write(data)
} catch {
    let message: String
    switch error {
    case VisionOCRError.missingImagePath:
        message = "missing_image_path"
    case VisionOCRError.failedToLoadImage:
        message = "failed_to_load_image"
    default:
        message = String(describing: error)
    }
    fputs(message + "\n", stderr)
    exit(1)
}
