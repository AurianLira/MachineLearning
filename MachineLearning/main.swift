//
//  main.swift
//  TestingML
//
//  Created by Aurian Moura de Lira on 05/09/25.
//

import Foundation

struct DatasetConfig {
    let name: String
    let filePath: String
    let delimiter: Character
    let labelColumnIndex: Int
}

let datasets = [
    DatasetConfig(name: "Wine Dataset",
                  filePath: "/Users/auri/Documents/Projetos/MachineLearning/MachineLearning/winequality-red.csv",
                  delimiter: ";",
                  labelColumnIndex: 11),
                  
    DatasetConfig(name: "Breast Cancer Dataset",
                  filePath: "/Users/auri/Documents/Projetos/MachineLearning/MachineLearning/data.csv",
                  delimiter: ",",
                  labelColumnIndex: 1)
]

let trainTestSplitRatio = 0.7
let kValues = [1, 3, 5, 7, 9, 11]

let fileManager = FileManager.default

for dataset in datasets {

    print("Analyzed dataset: \(dataset.name)")

    let filePath = dataset.filePath
    
    guard fileManager.fileExists(atPath: filePath) else {
    
        print("File not found: \(filePath)")
        continue
    }


    print("Reading the dataset...")
    
    let fullDataset = KNN.loadDataset(from: filePath, delimiter: dataset.delimiter, labelColumnIndex: dataset.labelColumnIndex)
    guard !fullDataset.isEmpty else {
        print("Unable to load data")
        continue
    }
    let normalizedDataset = KNN.normalizeDataset(fullDataset)

    let (trainSet, testSet) = KNN.splitDataset(normalizedDataset, splitRatio: trainTestSplitRatio)

    for k in kValues {
        let accuracy = KNN.calculateAccuracy(trainSet: trainSet, testSet: testSet, k: k, distanceMetric: KNN.manhattanDistance)
        let accuracyPercent = String(format: "%.2f%%", accuracy * 100)
        print("Accuracy for k=\(k): \(accuracyPercent)")
    }

}
