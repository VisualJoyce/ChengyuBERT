import nltk
from chengyubert.utils.tree import TreePrettyPrinter

x = '(X (X (X 小船) (X 大海)) (X 天边)) '
t = nltk.Tree.fromstring(x)

print(TreePrettyPrinter(t).text())
