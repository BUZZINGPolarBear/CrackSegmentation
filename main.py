import argparse
import torch
import os

from torch import optim
from data import dataset, data
# from models import FCHardNet
from utils import utils
from models import DUMMYmodel

def parse_args():
    parser = argparse.ArgumentParser(description='train, resume, test arguments')
    parser.add_argument('--train', '-t', action='store_true', default=True)
    parser.add_argument('--resume', '-r', action='store_true', default=False)
    parser.add_argument('--evaluate', '-e', action='store_true', default=False)
    parser.add_argument('--pretrained', '-pre', action='store_true', default=False)
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.01)
    parser.add_argument('--batch_size', '-b', type=int, default=16)
    parser.add_argument('--epochs', '-epoch', type=int, default=100)
    parser.add_argument('--workers', '-w', type=int, default= 4*4)
    parser.add_argument('--project_name', '-n', type=str, default="default")

    return parser.parse_args()
def main():
    args = parse_args()

    #! 환경을 세팅 (random seed 고정) torch 연산, numpy 연산 random 연산
    start_epoch = 0
    end_epoch = 300
    learning_rate = 0.01
    # Sample model parameter
    W = torch.zeros(1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    optimizer = optim.SGD([W,b], learning_rate)



    #! DataLoader - (img, label)
    current_path = os.getcwd();
    train_dataset = dataset.CrackDataSet(current_path, "train")
    print("1. DATA Set클래스를 정의. 현재 경로의 data set을 가져오도록 설정했습니다.")
    #print(train_dataset.__getitem__(0))

    train_data_loader, valid_data_loader = data.initialize_data_loader(16, 2)

    #! 모델 선언
    print("ㅠㅠㅠ 쿠다를 써야해서 모델이 안돌아가네요 대략적인 그림만 그려놓겠습니다!!")

    #! Img , label

    for epoch in range(start_epoch, end_epoch):
        DUMMY_JUNI_model = DUMMYmodel.DUMMY_model()
        utils.adjust_learing_rate(optimizer, epoch, args.learning_rate)

        cost = DUMMY_JUNI_model.train(train_data_loader)
        if epoch%10 == 0:
            print("################################")
            print("################################")
            print("################################")
            print(f'Epoch {epoch} Cost: {cost}')
            print("################################")
            print("################################")
            print("################################")
        # optimizer.zero_grad()
        # cost.backward()
        # optimizer.step()

    #! for 문 (몇 에폭까지 ?)
        #! for 문 (1epoch) Dataloader가 가지고 있는 거 다 내놔

        # train(model) - gradient 계산 모델 업데이트 학습 o

        # metric = validate(model) - gradient 계산 x - 학습 x

        # metric 이 좋아졌네?
        # 모델을 저장

    #! 모든 에폭이 종료
    # eval()
    #! 최종 점수 계산 및 최고 점 찍어줘
        
    

if __name__ == "__main__":
    main()