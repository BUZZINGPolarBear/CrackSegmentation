import random
class DUMMY_model:
    def __init__(self):
        print("===========================")
        print("어쩌구 저쩌구 설정들")
    def forward(self, data_loader):

        for i in range (0, len(data_loader)):
            print("conv 넣었다 늘리고 줄이고 우당탕탕... ")
        return random.randrange(0, 10)