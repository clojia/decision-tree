############ files ###########
- decision_tree.py
main function: select experiments: "testTennis", "testIris" and "testIrisNoisy".

- learner.py
Node class and DecisionTree class: generate training, validation and test data, output tree

- predictor.py
Predictor class: predict labels based on tree/rule set, output accuracy, post-prune rules

- printer.py
TreePrinter class and RuleSetPrinter class: pre-order traversal, print tree, convert tree to rule set, print instances distributions.

- /data/tennis/*.txt
tennis data files for experiment "testTennis"

- /data/iris/*.txt
iris data files for experiments "testIris" and "testIrisNoisy"


############# Usage ###########
python3 decision_tree.py -e <experiment>

e.g.
"""
python3 decision_tree.py -e testTennis
"""
