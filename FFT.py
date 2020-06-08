from math import cos, sin, pi

class complex():
    def __init__(self, real, image):
        self.real = real
        self.image = image

    def __mul__(self, other):
        return complex(self.real * other.real - self.image * other.image, self.real * other.image + self.image * other.real)

    def __add__(self, other):
        return complex(self.real + other.real, self.image + other.image)

    def __sub__(self, other):
        return complex(self.real - other.real, self.image - other.image)

P = list(map(int, input().split()))
n = len(P)
w = complex(cos(pi / n), sin(pi / n))

def FFT(poly, x):
    if len(poly) == 1:
        return poly[0] * pow(x, 2)
    P_even = filter(lambda x: poly.index(x) % 2 != 0, poly)
    P_odd = filter(lambda x: poly.index(x) % 2 == 0, poly)
    P_value = FFT(P_even, x * x) + x * FFT(P_odd, x * x)
    return P_value

print(FFT(P, w))