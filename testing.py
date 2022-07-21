

class A:

    def __init__(self,test = 30,**kwargs ) -> None:
        self.a = 1000
        self.test = test
        self.pars = kwargs
        try:
            self.z = self.pars["u"]
            print("yep")
        except KeyError:
            print("Nope")
        return
        self.a = 100


ff = A(a = 20, b = 100, c = 3874,u=20)

print(ff.a)


for _ in range(0):
    print("hey")