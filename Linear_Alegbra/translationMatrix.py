import numpy as np
def translate_object(points, tx, ty):
	for p in points:
		p[0] = p[0]+tx
		p[1] =  p[1]+ty
	
	return points

points = [[0, 0], [1, 0], [0.5, 1]]
tx, ty = 2, 3

print(translate_object(points,tx,ty))

