import argparse


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


    #! DataLoader - (img, label)
    #! Img(source Img for Training) - 정답지(Label)

    #! 모델 선언

    #! Img , label
    #! for 문 (몇 에폭까지 ?)
        #! for 문 (1epoch) Dataloader가 가지고 있는 거 다 내놔

        train(model) - gradient 계산 모델 업데이트 학습 o

        metric = validate(model) - gradient 계산 x - 학습 x

        metric 이 좋아졌네?
        모델을 저장

    #! 모든 에폭이 종료
    eval()
    #! 최종 점수 계산 및 최고 점 찍어줘

        
    

if __name__ == "__main__":
    main()