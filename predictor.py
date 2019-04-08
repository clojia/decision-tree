class Predictor:
    """
    predict labels, output accuracy, post-prune rules
    """
    def __init__(self,
            testData,  
            validationData,
            attributes, 
            attrValues, 
            tree, 
            rules):
        self.testData = testData
        self.validationData = validationData
        self.attributes = attributes
        self.attrValues = attrValues
        self.tree = tree             
        self.rules = rules

    def calculateAccuracy(self, data, tree): #calculate accuracy based on tree structure
        correctCount = 0
        for row in data:
            pred = self.predictLabel(tree, row)
            if pred == row[-1]:
                correctCount = correctCount + 1
        return correctCount / len(data) * 100

    def predictLabel(self, node, data): 
        if node.isLeaf:
            return node.label  
        indexOfAttribute = self.attributes.index(node.label)
        if node.threshold is None:
            for index, child in enumerate(node.children):
                if data[indexOfAttribute] == self.attrValues[node.label][index]:
                    return self.predictLabel(child, data)
        else:
            leftChild = node.children[0]
            rightChild = node.children[1]
            if data[indexOfAttribute] <= node.threshold:
                return self.predictLabel(leftChild, data)
            else:
                return self.predictLabel(rightChild, data)

    def matchRule(self, data, rule): 
        [conditionComb, label, distribution] = rule.replace("  (instances: ", " -> ").split(" -> ")
        conditions = conditionComb.split(" ^ ")
        for condition in conditions:
            if " <= " in condition:
                [attribute, threshold] = condition.split(" <= ")
                indexOfAttribute = self.attributes.index(attribute)
                if data[indexOfAttribute] > float(threshold):
                    return 0
            elif " > " in condition:
                [attribute, threshold] = condition.split(" > ")
                indexOfAttribute = self.attributes.index(attribute)
                if data[indexOfAttribute] <= float(threshold):
                    return 0
            elif " = " in condition: #categorical
                [attribute, value] = condition.split(" = ")
                indexOfAttribute = self.attributes.index(attribute)
                if data[indexOfAttribute] != value:
                    return 0          
        return label

    def matchRuleSet(self, data, rules):
        for rule in rules:
            if self.matchRule(data, rule):
                return self.matchRule(data, rule)
        return 0
    
    def calculateRuleAccuracy(self, data, rules): #calculate accuracy based on rule structure, mainly for post-pruning
        correctCount = 0
        for row in data:
            pred = self.matchRuleSet(row, rules)  
            if pred == row[-1]:
                correctCount = correctCount + 1
        return correctCount / len(data) * 100
 
    def pruneRules(self, rule): 
        maxAcc = self.calculateRuleAccuracy(self.validationData, [rule])  
        [conditionComb, label, distribution] = rule.replace("  (instances: ", " -> ").split(" -> ")
        conditions = conditionComb.split(" ^ ")
        for i in range(len(conditions))[::-1]:  #prune rule by removing preconditions
            newRule = (" ^ ").join(conditions[:i] + conditions[i+1:]) + " -> " + label + "  (instances: " + distribution
            newAcc = self.calculateRuleAccuracy(self.validationData, [newRule])
            if newAcc > maxAcc:  #return the one has greatest improvement
                maxAcc = newAcc
                rule = newRule
                self.pruneRules(rule)
        return rule, maxAcc
    
    def postPruning(self):
        ruleSet = {}
        for rule in self.rules: 
            (newRule, acc) = self.pruneRules(rule)
            ruleSet[newRule] = acc 
        return sorted(ruleSet, key=ruleSet.get, reverse=True) #sort pruned rules based on their estimated accuracy

