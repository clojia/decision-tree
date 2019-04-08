import os
import argparse

from printer import TreePrinter, RuleSetPrinter
from learner import Node, DecisionTree
from predictor import Predictor

def changeLabel(data, label):
    new_index = label.index(data[-1]) - 1
    data[-1] = label[new_index]

def addNoise(data, rate, label):
    num = int(rate/100 * len(data))
    for i in range(num):
        changeLabel(data[i], label)
#        print(data[i])
    return data

def testTennis(trainDataFile, testDataFile, attrDataFile):
    tennisTree = DecisionTree(trainDataFile, testDataFile, attrDataFile)
    tennisTree.loadData()
    tennisTree.preprocessData()
    trainData = tennisTree.getTrainData()
    testData = tennisTree.getTestData()
    validationData = tennisTree.getValidationData()
    tennisTree.generateTree(trainData)
    tree = tennisTree.getTree()
    attributes = tennisTree.getAttributes()
    attrValues = tennisTree.getAttrValues()
    tennisTreePrinter = TreePrinter(tree, attrValues)
    
    print("----------Tree----------")
    tennisTreePrinter.printTree()

    tennisRulePrinter = RuleSetPrinter(tree, attrValues)
    print("--------Rule Set--------")
    tennisRulePrinter.printRuleSet()
    rules = tennisRulePrinter.getRules()
 
    trainPred = Predictor(trainData, validationData, attributes, attrValues, tree, rules)
    trainAcc = trainPred.calculateAccuracy(trainData, tree)
    print("Training Accuracy (%): " + str(trainAcc))
    
    testPred = Predictor(testData, validationData, attributes, attrValues, tree, rules) 
    testAcc = testPred.calculateAccuracy(testData, tree)
    print("Test Accuracy (%): " + str(testAcc))

def testIris(trainDataFile, testDataFile, attrDataFile):
    irisTree = DecisionTree(trainDataFile, testDataFile, attrDataFile)
    irisTree.loadData()
    irisTree.preprocessData()
    irisTree.splitDataSet(irisTree.getTrainData())
    trainData = irisTree.getTrainData()
    testData = irisTree.getTestData()
    validationData = irisTree.getValidationData()
    irisTree.generateTree(trainData)
    tree = irisTree.getTree()
    attributes = irisTree.getAttributes()
    attrValues = irisTree.getAttrValues()
    irisTreePrinter = TreePrinter(tree, attrValues)
    print("----------Tree----------")
    irisTreePrinter.printTree()

    irisRulePrinter = RuleSetPrinter(tree, attrValues)
    print("--------Rule Set--------")
    irisRulePrinter.printRuleSet()
    rules = irisRulePrinter.getRules()

    trainPred = Predictor(trainData, validationData, attributes, attrValues, tree, rules)
    trainAcc = trainPred.calculateRuleAccuracy(trainData, rules)
    print("Training Accuracy (%): " + str(trainAcc))
    
    testPred = Predictor(testData, validationData, attributes, attrValues, tree, rules) 
    testAcc = testPred.calculateRuleAccuracy(testData, rules)
    print("Test Accuracy (%): " + str(testAcc))
 
    print("----Post-pruning Rules----")
    pruneRules = testPred.postPruning()
    print(os.linesep.join(pruneRules))
    trainPred = Predictor(trainData, validationData, attributes, attrValues, tree, pruneRules)
    trainAcc = trainPred.calculateRuleAccuracy(trainData, pruneRules)
    print("Training Accuracy (%): " + str(trainAcc))
    
    testPred = Predictor(testData, validationData, attributes, attrValues, tree, pruneRules) 
    testAcc = testPred.calculateRuleAccuracy(testData, pruneRules)
    print("Test Accuracy (%): " + str(testAcc))


def testIrisNoisy(trainDataFile, testDataFile, attrDataFile):
    irisTree = DecisionTree(trainDataFile, testDataFile, attrDataFile)
    irisTree.loadData()
    irisTree.preprocessData()
    irisTree.splitDataSet(irisTree.getTrainData())
    trainData = irisTree.getTrainData()
    testData = irisTree.getTestData()
    attributes = irisTree.getAttributes()
    attrValues = irisTree.getAttrValues() 
    validationData = irisTree.getValidationData()
    print(validationData)
    for rate in range(0, 21, 2): 
        noisyData = addNoise(trainData, rate, irisTree.getClasses()) 
        irisTree.generateTree(noisyData)
        tree = irisTree.getTree()
        irisRulePrinter = RuleSetPrinter(tree, attrValues)
        print("\nNoise Rate (%): " + str(rate))
        print("--------Rule Set--------")
        irisRulePrinter.printRuleSet()
        rules = irisRulePrinter.getRules()
        testPred = Predictor(testData, validationData, attributes, attrValues, tree, rules) 
    #    testAcc = testPred.calculateRuleAccuracy(validationData, rules)
        testAcc = testPred.calculateRuleAccuracy(testData, rules)
        print("Test Accuracy (%): " + str(testAcc))
        print("----Post-pruning Rules----")
        pruneRules = testPred.postPruning()
        print(os.linesep.join(pruneRules))
        testPred = Predictor(testData, validationData, attributes, attrValues, tree, pruneRules) 
        testAcc = testPred.calculateRuleAccuracy(testData, pruneRules)
        print("Test Accuracy (%): " + str(testAcc))

def main():
    parser = argparse.ArgumentParser(description="Decision Tree")
    parser.add_argument("-e", "--experiment", required=True, dest="experiment", 
            choices=["testTennis", "testIris", "testIrisNoisy"], help='experiment name.')
    args = parser.parse_args()
    print("Experiment: " + args.experiment)
    if args.experiment == "testTennis":
        testTennis("data/tennis/tennis-train.txt", "data/tennis/tennis-test.txt", "data/tennis/tennis-attr.txt")
    elif args.experiment == "testIris":
        testIris("data/iris/iris-train.txt", "data/iris/iris-test.txt", "data/iris/iris-attr.txt")
    elif args.experiment == "testIrisNoisy":
        testIrisNoisy("data/iris/iris-train.txt", "data/iris/iris-test.txt", "data/iris/iris-attr.txt")

if __name__ == '__main__':
    main()
