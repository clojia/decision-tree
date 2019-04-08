import math

class Node:
    def __init__(self, isLeaf, label, threshold, instances = 0):
        self.label = label
        self.threshold = threshold
        self.isLeaf = isLeaf
        self.children = []
        self.instances = instances

class DecisionTree:
    """
    load attributes, 
    training and test files,
    split training data for validation (in post-pruning),
    build and output decision tree

    """
    def __init__(self, trainDataFile, testDataFile, attrFile):
        self.trainDataFile = trainDataFile
        self.testDataFile = testDataFile
        self.attrFile = attrFile
        self.trainData = []
        self.validationData = []
        self.testData = []
        self.numAttributes = -1
        self.label = {}
        self.classes = []
        self.attrValues = {}
        self.attributes = []
        self.tree = None

    def getTrainData(self):
        return self.trainData

    def getValidationData(self):
        return self.validationData

    def getTestData(self):
        return self.testData

    def getClasses(self):
        return self.classes

    def getAttributes(self):
        return self.attributes

    def getAttrValues(self):
        return self.attrValues

    def getTree(self):
        return self.tree
    
    def loadData(self):
        with open(self.attrFile, "r") as file: #load attribute file
            for line in file:
                if line.strip() == '':
                    line = next(file)
                    [label, values] = line.split(" ", 1)
                    self.classes = [x.strip() for x in values.split(" ")]
                    self.label[label] = self.classes
                    break
                [attribute, values] = [x.strip() for x in line.split(" ", 1)]
                values = [x.strip() for x in values.split(" ")]
                self.attrValues[attribute] = values
        self.numAttributes = len(self.attrValues.keys())
        self.attributes = list(self.attrValues.keys())
        with open(self.trainDataFile, "r") as file: #load training data
            for line in file:
                row = [x.strip() for x in line.split(" ")]
                if row != [] or row != [""]:
                    self.trainData.append(row)
        with open(self.testDataFile, "r") as file:
            for line in file:
                row = [x.strip() for x in line.split(" ")]
                if row!= [] or row != [""]:
                    self.testData.append(row)

    def preprocessData(self):
        self.convertContinuousData(self.trainData)
        self.convertContinuousData(self.testData)
    
    def splitDataSet(self, data, splitRatio=0.85): 
        #split data for validation, default ration 0.15
        splitLength = int(len(data) * splitRatio)
        self.validationData = data[splitLength:]
        self.trainData = data[:splitLength]

    def convertContinuousData(self, data): #convert continuous data str to float 
        for index, row in enumerate(data):
            for attr_index in range(self.numAttributes):
                if self.isAttrContinuous(self.attributes[attr_index]): 
                    data[index][attr_index] = float(data[index][attr_index])

    def generateTree(self, data): 
        self.tree = self.id3(data, self.attributes)

    def id3(self, trainData, attributes): #generate trees according to information gain
        allSame = self.allSameClass(trainData)
        if len(trainData) == 0:
            return Node(True, "Fail", None)
        elif allSame is not False:
         #   print(trainData, len(trainData))
            return Node(True, allSame, None, len(trainData))
        elif len(attributes) == 0:
            majClass = self.getMajClass(trainData) 
            return Node(True, majClass, None, len(trainData))
        else: 
            (best_attribute, best_threshold, splitted) = self.splitAttribute(trainData, attributes)
          #  print(best_attribute)
            remainingAttributes = attributes[:]
            if best_attribute != -1:
                remainingAttributes.remove(best_attribute)
            node = Node(False, best_attribute, best_threshold)
            for subset in splitted:
                if not subset:
                    node.children.append(Node(True, self.getMajClass(trainData), None, len(trainData)))
                else:
                    node.children.append(self.id3(subset, remainingAttributes))
            return node

    def allSameClass(self, trainData):
        for row in trainData:
            if row[-1] != trainData[0][-1]:
                return False
         #   print(trainData)
        return trainData[0][-1]  
    
    def getMajClass(self, trainData):
        freq = [0] * len(self.classes)
        for row in trainData:
            index = self.classes.index(row[-1])
            freq[index] += 1
        maxInd =freq.index(max(freq))
        return self.classes[maxInd]
 
    def isAttrContinuous(self, attribute):
        if attribute not in self.attributes:
            raise ValueError("Attribute not listed")
        elif self.attrValues[attribute][0] == "continuous":
            return True
        else:
            return False

    def splitAttribute(self, trainData, attributes): 
        splitted = []
        maxInfoGain = -1 * float("inf")
        best_attribute = -1
        best_threshold = None
        for attribute in attributes:
            indexOfAttribute = self.attributes.index(attribute)
            if self.isAttrContinuous(attribute): #continuous attribute
                trainData.sort(key = lambda x: x[indexOfAttribute])
                for j in range(0, len(trainData) - 1):
                    if trainData[j][indexOfAttribute] != trainData[j+1][indexOfAttribute]:
                        threshold = (trainData[j][indexOfAttribute] + trainData[j+1][indexOfAttribute]) / 2 #caculate split threshold first
                        less = []
                        greater = []
                        for row in trainData:
                            if(row[indexOfAttribute] > threshold):
                                greater.append(row)
                            else:
                                less.append(row)
                        ig = self.getInfoGain(trainData, [less, greater])
                        if ig >= maxInfoGain:
                            splitted = [less, greater]
                            
                            maxInfoGain = ig
                            best_attribute = attribute
                            best_threshold = threshold 
            else: #categorical attribute
                valuesForAttribute = self.attrValues[attribute]
                subsets = [[] for a in valuesForAttribute]
                for row in trainData:
                    for index in range(len(valuesForAttribute)):
                        if row[indexOfAttribute] == valuesForAttribute[index]:
                            subsets[index].append(row)
                            break
                ig = self.getInfoGain(trainData, subsets)
                if ig > maxInfoGain:
                    maxInfoGain = ig
                    splitted = subsets
                    best_attribute = attribute
                    best_threshold = None
        return (best_attribute, best_threshold, splitted)

    def getInfoGain(self, unionSet, subsets):
        S= len(unionSet)
        impurityBeforeSplit = self.entropy(unionSet)
        weights = [len(subset)/S for subset in subsets]
        impurityAfterSplit = 0
        for i in range(len(subsets)):
            impurityAfterSplit += weights[i] * self.entropy(subsets[i])
        totalGain = impurityBeforeSplit - impurityAfterSplit
        return totalGain

    def entropy(self, trainDataSet):
        S = len(trainDataSet)
        if S == 0:
            return 0
        num_classes = [0 for i in self.classes]
        for row in trainDataSet:
            classIndex = list(self.classes).index(row[-1])
            num_classes[classIndex] += 1
        num_classes = [x/S for x in num_classes]
        entropy = 0
        for num in num_classes:
            entropy += num * self.log(num)
        return entropy * (-1)

    def log(self, x):
        if x == 0: 
            return 0
        else:
            return math.log(x, 2)
