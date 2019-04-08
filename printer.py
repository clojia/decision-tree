import os

#pre-order traversal,  tree printer and rule set printer

class TreePrinter:
    """
    recursively print nodes
    """
    
    def __init__(self, tree, attrValues):
        self.tree = tree
        self.attrValues = attrValues

    def printTree(self):
        self.printNode(self.tree, self.attrValues)

    def printNode(self, node, attrValues, indent=""):
        if not node.isLeaf:
            if node.threshold is None: #categorical nodes
                for index, child in enumerate(node.children):
                    if child.isLeaf:
                        print(indent + node.label + " = " + attrValues[node.label][index] + " : " + child.label + "  (instances: " + str(child.instances) + ")") #instances as distributions
                    else:
                        print(indent + node.label + " = " + attrValues[node.label][index] + " : ")
                        self.printNode(child, attrValues, indent + "      ")
            else:                    #continuous nodes
                leftChild = node.children[0]
                rightChild = node.children[1] 
                if leftChild.isLeaf:
                    print(indent + node.label + " <= " + str(node.threshold) + " : " + leftChild.label + "  (instances: " + str(leftChild.instances) + ")")
                else:
                    print(indent + node.label + " <= " + str(node.threshold) + " : ")
                    self.printNode(leftChild, attrValues, indent + "      ")

                if rightChild.isLeaf:
                    print(indent + node.label + " > " + str(node.threshold) + " : " + rightChild.label + "  (instances: " + str(rightChild.instances) + ")")

                else:
                    print(indent + node.label + " > " + str(node.threshold) + " : ")
                    self.printNode(rightChild, attrValues, indent + "      ")


class RuleSetPrinter:
    """
    convert tree to rules, recursively print rules
    """
    def __init__(self, tree, attrValues):
        self.tree = tree
        self.attrValues = attrValues
        self.stack = []
        self.rules = set()

    def getRules(self):
        return self.rules

    def printRuleSet(self):
        self.convertRules(self.tree, self.attrValues, self.stack, self.rules)
        print(os.linesep.join(self.rules))
    
    def convertRules(self, node, attrValues, stack, rules):
        if not node.isLeaf:
            if node.threshold is None: #categorical nodes
                for index, child in enumerate(node.children):
                    if child.isLeaf:
                        stack.append(node.label + " = " + attrValues[node.label][index] + " -> " + child.label + "  (instances: " + str(child.instances) + ")")
                        rules.add(" ^ ".join(stack))
                        stack.pop()
                    else: 
                        stack.append( node.label + " = " + attrValues[node.label][index])
                        self.convertRules(child, attrValues, stack, rules)
                        stack.pop()
            else:                    #continuouse nodes
                leftChild = node.children[0]
                rightChild = node.children[1] 
                if leftChild.isLeaf:
                    stack.append(node.label + " <= " + str(node.threshold) + " -> " + leftChild.label + "  (instances: " + str(leftChild.instances) + ")")
                    rules.add(" ^ ".join(stack))
                    stack.pop() 
                else:
                    stack.append(node.label + " <= " + str(node.threshold))
                    self.convertRules(leftChild, attrValues, stack, rules)
                    stack.pop()

                if rightChild.isLeaf:
                    stack.append(node.label + " > " + str(node.threshold) + " -> " + rightChild.label + "  (instances: " + str(rightChild.instances) + ")")
                    rules.add(" ^ ".join(stack))
                    stack.pop() 
                else:
                    stack.append(node.label + " > " + str(node.threshold))
                    self.convertRules(rightChild, attrValues, stack, rules)
                    stack.pop() 
