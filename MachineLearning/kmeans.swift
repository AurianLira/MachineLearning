//
//  kmeans.swift
//  MachineLearning
//
//  Created by Aurian Moura de Lira on 26/09/25.
//

import Foundation

struct Kmeans {
    var clusters: Int
    
    func loadDataset(from filePath: String) -> [DataPoint] {
        var dataset: [DataPoint] = []
        if let content = try? String(contentsOf: URL(fileURLWithPath: filePath), encoding: .utf8) {
            let lines = content.components(separatedBy: .newlines).filter { !$0.isEmpty }
            for line in lines {
                let values = line.split(separator: ";")
                if let x = Double(values[0]),
                   let y = Double(values[1]),
                   let label = Int(values[2]) {
                    dataset.append(DataPoint(x: x, y: y, label: label))
                } else {
                    print("conversion error for line: \(line)")
                }
            }
        } else {
            print("error reading file")
        }
        return dataset
    }
    
    private func splitDataset(dataset: [DataPoint]) -> [[DataPoint]] {
        var labelGroups: [Int: [DataPoint]] = [:]
        
        for point in dataset {
            labelGroups[point.label, default: []].append(point)
        }
        
        let sortedLabels = labelGroups.keys.sorted()
        
        var groups: [[DataPoint]] = []
        for label in sortedLabels {
            groups.append(labelGroups[label]!)
        }
        while groups.count < clusters {
            groups.append([])
        }
        
        return groups
    }
    
    
    func calculateMeans(groups: [[DataPoint]]) -> [(meanX: Double, meanY: Double)] {
        var means: [(meanX: Double, meanY: Double)] = []
        for group in groups {
            guard !group.isEmpty else {
                means.append((meanX: 0, meanY: 0))
                continue
            }
            let sumX = group.reduce(0) { $0 + $1.x }
            let sumY = group.reduce(0) { $0 + $1.y }
            let count = Double(group.count)
            means.append((meanX: sumX / count, meanY: sumY / count))
        }
        return means
    }
    
    func calculateManhattanDistance(points: [DataPoint], means: [(meanX: Double, meanY: Double)]) -> [[Double]] {
        return points.map { point in
            means.map { mean in
                abs(point.x - mean.meanX) + abs(point.y - mean.meanY)
            }
        }
    }
    
    private func redistributePoints(dataset: [DataPoint], means: [(meanX: Double, meanY: Double)]) -> [[DataPoint]] {
        let distances = calculateManhattanDistance(points: dataset, means: means)
        var newGroups: [[DataPoint]] = Array(repeating: [], count: clusters)
        for (i, point) in dataset.enumerated() {
            if let minIndex = distances[i].enumerated().min(by: { $0.element < $1.element })?.offset {
                newGroups[minIndex].append(point)
            }
        }
        return newGroups
    }
    
    private func groupsChanged(oldGroups: [[DataPoint]], newGroups: [[DataPoint]]) -> Bool {
        for i in 0..<clusters {
            if oldGroups[i].map({ $0.label }) != newGroups[i].map({ $0.label }) {
                return true
            }
        }
        return false
    }
    
    func distributeNewGroupsWithIterations(dataset: [DataPoint]) -> ([[DataPoint]], Int) {
        var groups = splitDataset(dataset: dataset)
        var changed = true
        var iterations = 0
        
        while changed {
            let means = calculateMeans(groups: groups)
            let newGroups = redistributePoints(dataset: dataset, means: means)
            changed = groupsChanged(oldGroups: groups, newGroups: newGroups)
            groups = newGroups
            iterations += 1
        }
        
        return (groups, iterations)
    }
    
}
