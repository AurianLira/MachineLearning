//
//  perceptron.swift
//  MachineLearning
//
//  Created by Aurian Moura de Lira on 19/09/25.
//

import Foundation

struct Perceptron {
    var weights: [Double]
    var bias: Double
    var featureCount: Int
    var learningRate: Double
    let labelMin: Int
    let labelMax: Int

    init(featureCount: Int = 2, learningRate: Double = 0.1, labelMin: Int = 1, labelMax: Int = 2) {
        self.featureCount = featureCount
        self.learningRate = learningRate
        self.labelMin = labelMin
        self.labelMax = labelMax
        self.weights = Array(repeating: 0.1, count: featureCount)
        self.bias = 1.0
    }

    private func activation(_ x: Double) -> Int {
        return x >= 0 ? labelMax : labelMin
    }

    func calculateoutput(_ features: [Double]) -> Int {
        var sum = bias
        for i in 0..<weights.count {
            sum += weights[i] * features[i]
        }
        return activation(sum)
    }

    mutating func applyperceptron(features: [Double], desiredvalue: Int) {
        let y = calculateoutput(features)
        let erro = desiredvalue - y
        for i in 0..<weights.count {
            weights[i] += learningRate * Double(erro) * features[i]
        }
        bias += learningRate * Double(erro)
    }

    mutating func trainDataset(dataset: [(features: [Double], label: Int)], epochs: Int) {
        for _ in 0..<epochs {
            for sample in dataset {
                applyperceptron(features: sample.features, desiredvalue: sample.label)
            }
        }
    }

    func calculateaccuracy(dataset: [(features: [Double], label: Int)]) -> Double {
        guard !dataset.isEmpty else { return 0.0 }
        let correct = dataset.filter { calculateoutput($0.features) == $0.label }
        return Double(correct.count) / Double(dataset.count)
    }
}
