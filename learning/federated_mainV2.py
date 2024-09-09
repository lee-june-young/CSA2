import copy, time
import numpy as np

import torch

from .options import args_parser
from .update import LocalUpdate, test_inference
from .models import CNNMnist
from .utils import get_mnist_train, get_mnist_test, get_users_data, average_weights, exp_details, average_weight
import learning.models_helper as mhelper
import logging
from common import writeToExcel, readWeightsFromFile

# 전역변수 선언
train_loss, train_accuracy, train_dataset = [], [], []
run_data = []
weights_data = [["epoch", "user_idx", "weight[0]", "weight[0] (가중치 적용 후)", "weight[-1]", "weight[-1] (가중치 적용 후)"]]
user_dataset_num = []
args = args_parser()
user_groups = {} #0807
isFirst=True

# [호츌] : 서버, 클라이언트
# [리턴] : global_model 
# 처음 시작할 때만 호출도는 setup 함수.
def setup(iidMode):
    global args, train_dataset

    if iidMode:
        args = args_parser(1, 0)
    else:
        args = args_parser(0, 1)
    # exp_details(args)

    if args.gpu:
        torch.cuda.set_device(int(args.gpu))
    device = 'cuda' if args.gpu else 'cpu'

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    
    # get train data
    train_dataset = get_mnist_train()
    #print("Train Dataset 배열의 크기:", len(train_dataset))
    #print("Train Dataset[0]:", train_dataset[0])
    
    return global_model

# [호츌] : 서버
# [인자] : num_users(사용자 수), isCluster(클러스터 기반인지), cluster(클러스터 정보), num_items(클러스터별로 할당할 data set 사이즈)
# [리턴] : user_groups[dict[int, Any]]
def get_user_dataset(client_num, index, isCluster=False, cluster=0, num_items=0):
    global args, train_dataset, user_groups, isFirst
    #user_groups = get_users_data(args, num_users, train_dataset, isCluster, cluster, num_items)
    if index==0 and isFirst :
        user_groups = get_users_data(args, client_num, train_dataset, isCluster, cluster, num_items)
        isFirst=False
    
    # 여기 user_num에 준만큼 데이터가 커짐



    #user_dataset = []
    #user_dataset.extend(user_groups[indexes[0]])
    #user_dataset.extend(user_groups[indexes[1]])
    #print(user_groups[108])

    #print("user_groups = " , user_groups)
    user_dataset=user_groups[index]
    print(index, "'s userdataset: ", len(user_dataset))
    print(len(user_dataset_num))
    print(f"{user_dataset[0]}~{user_dataset[-1]}")
    if len(user_dataset_num) == 0:
        
        for i in range(client_num):
            user_dataset_num.append(len(user_groups[i]))
        print("유저 데이터 셋 갯수 설정됨!!!")
        print(f"user_dataset_num={user_dataset_num}")

        run_data.append([])
        run_data.append(["사용자별 데이터 갯수"])
        run_data.append(user_dataset_num)
        run_data.append([])
        run_data.append(["round", "accuracy"])


    return user_dataset

# 서버는 user_groups[idx] 를 클라이언트로 전달
# ex) 0번 클라이언트에게 user_groups[0], 1번 클라이언트에게 user_groups[1] 전달

# [호츌] : 클라이언트
# [인자] : global_model, user_group(서버에게 받은 데이터셋), epoch(몇 번째 학습인지 저장해놓은 변수)
# [리턴] : local_model, local_weight, local_loss
# localupdate를 수행한 후 local_weight와 local_loss를 update하여 리턴
def local_update(client_num, global_model, idx, epoch):
    global args, train_dataset
    
    # print("get dataset")
    print(">> 파이썬 get_user_dataset!!!!!!!!!!!")
    user_dataset = get_user_dataset(client_num, idx)
    global_model.train()

    print(">> 파이썬 local update 시작")


    #print("train_dataset: ", len(train_dataset))
    #print("user_dataset: ", len(user_dataset))
    #print(user_dataset)

    weights_info, before_weights=mhelper.flatten_tensor(mhelper.get_model_weights(global_model))
    print("weight[0](before): ", before_weights[0])
    #print("weight[n-1](before): ", before_weights[len(before_weights)-1])

    with open("weight.txt", "w") as f:
        f.write("weights before localupdate:\n")
        for w in before_weights:
            f.write(str(w) + " ")
        f.write("\n\n")

    #print("args:", args)

    local_model = LocalUpdate(args=args, dataset=train_dataset,idxs=user_dataset) #dataset: 전체데이터, idxs: userdata
    w, loss = local_model.update_weights(
        model=copy.deepcopy(global_model), global_round=epoch)
    local_weight = copy.deepcopy(w)
    local_loss = copy.deepcopy(loss)

    weights_info, weight = mhelper.flatten_tensor(local_weight)

    print("weight[0](after): ", weight[0])
    weights_data.append([epoch, idx, weight[0], weight[0]*(user_dataset_num[idx]/sum(user_dataset_num)), weight[len(weight)-1], weight[len(weight)-1]*(user_dataset_num[idx]/sum(user_dataset_num)), user_dataset_num[idx]/sum(user_dataset_num)])
    print(f"가중치 적용 전 weight[0] {weight[0]}")
    print(f"user {idx}의 가중치 = {user_dataset_num[idx]}/{sum(user_dataset_num)} =  {(user_dataset_num[idx]/sum(user_dataset_num))}")
    
    
    for i in range(len(weight)):
        weight[i] = weight[i] * (user_dataset_num[idx]/sum(user_dataset_num)) #흠!!! 가중치 = 유저가 가진데이터/전체데이터
    
    print(f"가중치 적용 후 weight[0] {weight[0]}")
    
    print(len(user_dataset))
    #print(user_dataset)
    #user_input = input(f"user {idx} 넘어가시겠습니까??: (데이터수= {len(user_dataset)})")
    
    # print("weight 길이: ",len(weight))
    # print("weight[0](after): ", weight[0])
    # print("weight[n-1](after): ", weight[len(weight)-1])

    with open("weight.txt", "w") as f:
        f.write("weights after localupdate:\n")
        for w in weight:
            f.write(str(w) + " ")
        f.write("\n\n")

    print(">> 파이썬 local update 종료")
    #return local_model, local_weight, local_loss
    return weight



# 클라언트는 local_model은 리턴받아 저장
# 서버로 local_weight, local_loss를 전달

# [호츌] : 서버
# [인자] : global_model, local_weight_sum (local_weight들의 합)
# [리턴] : global_model (업데이트된 global_model) 
# local train이 끝나고 서버는 해당 결과를 모아서 global_model을 업데이트 
def update_globalmodel(global_model, local_weight_sum):
    global train_loss

    #local weight의 합 평균 내기
    average_weight = average_weights(local_weight_sum)

    print("Local weight sum:", local_weight_sum)
    print("Average weight:", average_weight)

    global_model.load_state_dict(average_weight) # update

    # loss
    # loss_avg = sum(local_losses) / len(local_losses)
    # train_loss.append(loss_avg)
    return global_model

# 서버는 전달받은 update된 global model을 클라이언트들에게 전송

# [호츌] : 클라이언트
# [인자] : local_model (바로 이전에 클라이언트가 학습한 결과), global_model (서버로부터 전달받은 update된 global_model) 
# [리턴] : acc (정확도)
# 매 epoch마다의 검증과 모든 학습 후 정확성 출력을 위해 새롭게 업데이트 된 global_model과 이전에 학습해서 나온 local_model을 비교
def test_accuracy(local_model, global_model):
    global_model.eval()
    acc, loss = local_model.inference(model=global_model)
    return acc

# 클라이언트는 리턴된 acc를 서버로 전달하고 local_train 시작

# [호츌] : 서버
# [인자] : list_acc (클라이언트들로부터 전달받은 acc들을 저장해놓은 배열), epoch(몇 번째 학습인지 저장해놓은 변수)
# [리턴] : X
# 클라이언트들이 보낸 acc 값들로 해당 학습의 정확도를 저장하고 epoch 매 2회마다 train loss 와 train accuracy를 출력
def add_accuracy(list_acc, epoch):
    # list_acc.append(acc)
    global train_accuracy
    train_accuracy.append(sum(list_acc)/len(list_acc))
    print_every = 1
    global train_loss
    if (epoch+1) % print_every == 0:
        print(f' \nAvg Training Stats after {epoch+1} global rounds:')
        print(f'Training Loss : {np.mean(np.array(train_loss))}')
        print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))

# [호츌] : 서버
# [인자] : global_model
# [리턴] : test accuracy
# # 모든 학습이 끝난후 출력 
def test_model(global_model):
    global args
    test_dataset = get_mnist_test()
    test_acc, test_loss = test_inference(args, global_model, test_dataset)
    
    #print(f' \n Results after {args.epochs} global rounds of training:')
    #print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    #print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))
    #print("|---- Test Loss: {:.2f}%".format(100*test_loss))

    # Saving the objects train_loss and train_accuracy:
    # file_name = '../save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
    #     format(args.dataset, args.model, args.epochs, args.frac, args.iid,
    #            args.local_ep, args.local_bs)

    # with open(file_name, 'wb') as f:
    #     pickle.dump([train_loss, train_accuracy], f)

    # 로그를 저장할 파일 경로

    log_file_path = 'accuracy_file.txt'
    logging.basicConfig(
        level=logging.INFO,  # 기록할 로그의 수준 설정
        format='%(asctime)s - %(levelname)s - %(message)s',  # 로그 메시지의 형식
        handlers=[
           logging.FileHandler(log_file_path, mode='a'),  # 파일을 추가 모드로 열기
        ]
    )
    logging.info(f"accuracy: {100 * test_acc}%")

    print("accuracy: ",  100*test_acc, "%")
    return 100*test_acc

def setInitalWeight(model, epoch):
    # do final aggregation: aggregate all IS value
    print(f"hi I calling setInitalWeight now!!! epoch = {epoch}")
    base_weights = mhelper.restore_weights_tensor(mhelper.default_weights_info, readWeightsFromFile())
    model.load_state_dict(base_weights)

    # 6. test model
    accuracy = round(test_model(model), 4)
    #weights_data.append([epoch, "서버", base_weights[0], base_weights[len(base_weights)-1], accuracy])
    if len(run_data)==0: # 한번만 기록하기 위함
        run_data.append(["setInitalWeight"])
        run_data.append(["round","accuracy"])
        run_data.append([epoch,accuracy])
    #print(run_data)
    return model

def finalAggregationV2(model, sum_weights, epoch, index):
    # do final aggregation: aggregate all IS value
    print(f"hi I calling final aggregation now!!! epoch = {epoch}")

    print(len(sum_weights))

    # 5. update global model
    new_weight = mhelper.restore_weights_tensor(mhelper.default_weights_info, sum_weights)
    model.load_state_dict(new_weight)
    #print("model: ", model)

    # 6. test model
    log_file_path = 'accuracy_file.txt'
    logging.basicConfig(
        level=logging.INFO,  # 기록할 로그의 수준 설정
        format='%(asctime)s - %(levelname)s - %(message)s',  # 로그 메시지의 형식
        handlers=[
           logging.FileHandler(log_file_path, mode='a'),  # 파일을 추가 모드로 열기
        ]
    )
    logging.info(f"sum_weights[0]: {sum_weights[0]}")
    logging.info(f"sum_weights[n-1]: {sum_weights[21839]}")
    # 6. test model
    if index==0:
        accuracy = round(test_model(model), 4)
        weights_data.append([])
        weights_data.append(["epoch", "주체", "웨이트(0)", "웨이트(n-1)", "정확도"])
        weights_data.append([epoch, "server", sum_weights[0], sum_weights[len(sum_weights)-1], accuracy]) #라운드, 서버, 웨이트(0), 웨이트(n-1), 정확도
        weights_data.append([])
        run_data.append([epoch, accuracy])
        
    ############
    #print(run_data)
    
    #self.totalTime = round(time.time() - self.start, 4)ls
    #self.allTime = round(self.allTime + self.totalTime, 4)
    print("sum_weights[0]: ", sum_weights[0])
    print("sum_weights[n-1]: ", sum_weights[21839])
    #print("my_model:", mhelper.get_model_weights(model))``
    return model

def write_result():
    print("write_result")
    print(run_data)
    writeToExcel('/home/lab/results/csa.xlsx', run_data)
    writeToExcel('/home/lab/results/weights.xlsx', weights_data)