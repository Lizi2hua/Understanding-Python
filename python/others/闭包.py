def outter(out):
    x=out
    def inner():
        print(x)
    return inner
a=1
x=outter(1)
x()