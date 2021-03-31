import monkdata as m
import dtree as d

t = d.buildTree(m.monk1, m.attributes)
print(d.check(t, m.monk1))
print(round(1-d.check(t, m.monk1), 3))
print(d.check(t, m.monk1test))
print(round(1-d.check(t, m.monk1test), 3))

t = d.buildTree(m.monk2, m.attributes)
print(d.check(t, m.monk2))
print(round(1-d.check(t, m.monk2), 3))
print(d.check(t, m.monk2test))
print(round(1-d.check(t, m.monk2test), 3))

t = d.buildTree(m.monk3, m.attributes)
print(d.check(t, m.monk3))
print(round(1-d.check(t, m.monk3), 3))
print(d.check(t, m.monk3test))
print(round(1-d.check(t, m.monk3test), 3))
