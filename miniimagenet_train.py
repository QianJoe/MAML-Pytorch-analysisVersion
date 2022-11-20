import  torch, os
import  numpy as np
from    MiniImagenet import MiniImagenet
import  scipy.stats
from    torch.utils.data import DataLoader
from    torch.optim import lr_scheduler
import  random, sys, pickle
import  argparse

from meta import Meta


def mean_confidence_interval(accs, confidence=0.95):
    n = accs.shape[0]
    m, se = np.mean(accs), scipy.stats.sem(accs)
    h = se * scipy.stats.t._ppf((1 + confidence) / 2, n - 1)
    return m, h


def main():

    # 设置CPU生成的随机数种子，方便重新运行时，生成的随机数保持一致
    torch.manual_seed(222)
    # 设置CGPU生成的随机数种子，方便重新运行时，生成的随机数保持一致
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)
    # 打印一下这些参数，是否解析成功
    print(args)

    # 定义网络结构，写到数组config中
    # 好处：1.可以实现灵活搭建网络
    # 2.MAML的模型参数需要内部更新和外部更新，所以传统的搭建方式不容易实现
    config = [
        ('conv2d', [32, 3, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 1, 0]),
        ('flatten', []),
        ('linear', [args.n_way, 32 * 5 * 5])
    ]

    # 用device分配GPU
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device('cuda')
    # 实例化网络训练类
    maml = Meta(args, config).to(device)
    # filter用法：filter(function, iterable)
    # function – 判断函数。
    # iterable – 可迭代对象
    # 返回的是迭代器
    # 这个函数是筛选网络net中 requires_grad==True 的网络层
    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    # np.prod 这里是按行乘，对每个x
    # 算出训练数据的总数
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)

    # batchsz here means total episode number
    # 这里就是加载的训练数据和测试数据
    mini = MiniImagenet('miniimagenet/', mode='train', n_way=args.n_way, k_shot=args.k_spt,
                        k_query=args.k_qry,
                        batchsz=10000, resize=args.imgsz)
    mini_test = MiniImagenet('miniimagenet/', mode='test', n_way=args.n_way, k_shot=args.k_spt,
                             k_query=args.k_qry,
                             batchsz=100, resize=args.imgsz)

    # args.epoch 默认给的是60000，这里变成6
    for epoch in range(args.epoch//10000):
        # fetch meta_batchsz num of episode each time
        # 加载训练数据-DataLodader创建迭代器
        # task的数量为4，这里相当于采样了4次任务
        # 随机shuffle
        # 子线程num_workers，负责将batch加载到RAM，
        # 设置大，好处就是速度快，坏处就是内存开销大，加重CPU负担，一般单跑一个任务，直接设置为CPU的核心数
        # 锁页内存pin_memory，设置为True，直接将内存的张量转义到GPU的显存速度会快，如果显存爆炸，就设置为False
        db = DataLoader(mini, args.task_num, shuffle=True, num_workers=1, pin_memory=True)
        # 遍历迭代器，将训练数据送入到网络训练类，训练网络，输出准确率
        for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db):
            # 指定GPU来处理
            x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)
            # 输出准确率
            accs = maml(x_spt, y_spt, x_qry, y_qry)

            if step % 30 == 0:  # 每30步输出一次a准确率
                print('step:', step, '\ttraining acc:', accs)

            if step % 500 == 0:  # evaluation  每500步评估一次
                db_test = DataLoader(mini_test, 1, shuffle=True, num_workers=1, pin_memory=True)
                accs_all_test = []
                 # 遍历迭代器，将测试数据送入到网络训练类，训测试网络，输出准确率
                # squeeze是删除第一维，维度必须为1
                for x_spt, y_spt, x_qry, y_qry in db_test:
                    x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
                                                 x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)
                    # 微调阶段
                    accs = maml.finetunning(x_spt, y_spt, x_qry, y_qry)
                    accs_all_test.append(accs)

                # [b, update_step+1]
                accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
                print('Test acc:', accs)


if __name__ == '__main__':

    # 实例化一个参数解析器
    argparser = argparse.ArgumentParser()
    # 默认epoch为60000
    argparser.add_argument('--epoch', type=int, help='epoch number', default=60000)
    # 5W 1K
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    # support set 为每个类别1个已标注的样本
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    # query set 为每个类别15个已标注的样本
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    # 样本为三通道图像  84*84*3
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=84)
    argparser.add_argument('--imgc', type=int, help='imgc', default=3)
    # 一个batch包括4个任务
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=4)
    # 第二次梯度更新的学习率
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    # 第一次梯度更新的学习率
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    # 这里用到的是query set 5Way * 15 = 75个样本
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    # 面对新task 微调阶段
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    # 解析出所有参数
    args = argparser.parse_args()

    main()
