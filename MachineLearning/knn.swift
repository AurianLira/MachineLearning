//
//  knn.swift
//  TestingML
//
//  Created by Aurian Moura de Lira on 12/09/25.
//

import Foundation

struct Dataset {
    let parameters: [Double]
    let label: String
}

struct KNN {

    static func loadDataset(from filePath: String, delimiter: Character = ",", labelColumnIndex: Int = -1) -> [Dataset] {
        do {
            let content = try String(contentsOfFile: filePath, encoding: .utf8)
            let lines = content.components(separatedBy: .newlines).filter { !$0.isEmpty }
            
            guard lines.count > 1 else { return [] }
            let dataLines = lines.dropFirst()
            
            var dataset: [Dataset] = []
            
            for line in dataLines {
                let columns = line.split(separator: delimiter).map { String($0) }
                guard labelColumnIndex >= 0 && labelColumnIndex < columns.count else { continue }
                
                let label = columns[labelColumnIndex]
                let features = columns.enumerated()
                    .filter { $0.offset != labelColumnIndex }
                    .compactMap { Double($0.element) }
                
                guard features.count == columns.count - 1 else { continue }
                
                dataset.append(Dataset(parameters: features, label: label))
            }
            
            return dataset
            
        } catch {
            print("Error reading CSV file: \(error)")
            return []
        }
    }
    
    static func normalizeDataset(_ data: [Dataset]) -> [Dataset] {
        guard !data.isEmpty else { return [] }
        let featureCount = data.first!.parameters.count
        var mins = Array(repeating: Double.greatestFiniteMagnitude, count: featureCount)
        var maxs = Array(repeating: -Double.greatestFiniteMagnitude, count: featureCount)
        
        for sample in data {
            for (i, value) in sample.parameters.enumerated() {
                mins[i] = min(mins[i], value)
                maxs[i] = max(maxs[i], value)
            }
        }
        
        return data.map { sample in
            let normalizedFeatures = sample.parameters.enumerated().map { (i, value) -> Double in
                let range = maxs[i] - mins[i]
                return range == 0 ? 0.0 : (value - mins[i]) / range
            }
            return Dataset(parameters: normalizedFeatures, label: sample.label)
        }
    }
    
    static func splitDataset(_ data: [Dataset], splitRatio: Double) -> (train: [Dataset], test: [Dataset]) {
        let shuffled = data.shuffled()
        let splitIndex = Int(Double(shuffled.count) * splitRatio)
        return (Array(shuffled[..<splitIndex]), Array(shuffled[splitIndex...]))
    }
    
    static func manhattanDistance(_ a: [Double], _ b: [Double]) -> Double {
        precondition(a.count == b.count, "Error: Vectors of different sizes")
        
        var distance: Double = 0.0
        for i in 0..<a.count {
            distance += abs(a[i] - b[i])
        }
        
        return distance
    }


    private static func calculateDistances(from testSample: Dataset, to trainSet: [Dataset], distanceMetric: ([Double],[Double])->Double) -> [(dist: Double, label: String)] {
        return trainSet.map { sample in
            (dist: distanceMetric(testSample.parameters, sample.parameters), label: sample.label)
        }
    }

    private static func getNearestNeighbors(from distances: [(dist: Double,label: String)], k: Int) -> [(dist: Double,label: String)] {
        return Array(distances.sorted { $0.dist < $1.dist }.prefix(k))
    }
    
    static func predict(testSample: Dataset, trainSet: [Dataset], k: Int, distanceMetric: ([Double],[Double])->Double) -> String {
        let distances = calculateDistances(from: testSample, to: trainSet, distanceMetric: distanceMetric)
        let kNearest = getNearestNeighbors(from: distances, k: k)
        let labelCounts = Dictionary(grouping: kNearest, by: { $0.label }).mapValues { $0.count }
        return labelCounts.max(by: { $0.value < $1.value })?.key ?? "N/A"
    }
    
    static func calculateAccuracy(trainSet: [Dataset], testSet: [Dataset], k: Int, distanceMetric: ([Double],[Double])->Double) -> Double {
        guard !testSet.isEmpty else { return 0.0 }
        let correct = testSet.filter { predict(testSample: $0, trainSet: trainSet, k: k, distanceMetric: distanceMetric) == $0.label }
        return Double(correct.count)/Double(testSet.count)
    }
}
