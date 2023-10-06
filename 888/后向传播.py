import torch
import matplotlib.pyplot as plt


def sigmoid(z):
    a = 1 / (1 + torch.exp(-z))
    return a


def forward_propagate(x1, x2):
    in_h1 = w1 * x1 + w3 * x2
    out_h1 = sigmoid(in_h1)  # out_h1 = torch.sigmoid(in_h1)
    in_h2 = w2 * x1 + w4 * x2
    out_h2 = sigmoid(in_h2)  # out_h2 = torch.sigmoid(in_h2)

    in_o1 = w5 * out_h1 + w7 * out_h2
    out_o1 = sigmoid(in_o1)  # out_o1 = torch.sigmoid(in_o1)
    in_o2 = w6 * out_h1 + w8 * out_h2
    out_o2 = sigmoid(in_o2)  # out_o2 = torch.sigmoid(in_o2)

    print("正向计算：", out_o1.data, out_o2.data)
    return out_o1, out_o2


def loss_fuction(y1_pred, y2_pred, y1, y2):  # 损失函数
    # print(y1_pred, y2_pred, y1, y2)
    loss = (1 / 2) * (y1_pred - y1) ** 2 + (1 / 2) * (y2_pred - y2) ** 2  # 考虑 ： t.nn.MSELoss()
    print("损失函数：", loss)
    return loss


def update_w(w1, w2, w3, w4, w5, w6, w7, w8):
    # 步长
    step = 1
    w1.data = w1.data - step * w1.grad.data
    w2.data = w2.data - step * w2.grad.data
    w3.data = w3.data - step * w3.grad.data
    w4.data = w4.data - step * w4.grad.data
    w5.data = w5.data - step * w5.grad.data
    w6.data = w6.data - step * w6.grad.data
    w7.data = w7.data - step * w7.grad.data
    w8.data = w8.data - step * w8.grad.data
    w1.grad.data.zero_()  # 注意：将w中所有梯度清零
    w2.grad.data.zero_()
    w3.grad.data.zero_()
    w4.grad.data.zero_()
    w5.grad.data.zero_()
    w6.grad.data.zero_()
    w7.grad.data.zero_()
    w8.grad.data.zero_()
    return w1, w2, w3, w4, w5, w6, w7, w8


if __name__ == "__main__":
    x1, x2 = torch.Tensor([0.5]), torch.Tensor([0.3])
    y1, y2 = torch.Tensor([0.23]), torch.Tensor([-0.07])
    print("=====输入值：x1, x2；真实输出值：y1, y2=====")
    print(x1, x2, y1, y2)
    w1, w2, w3, w4, w5, w6, w7, w8 = torch.Tensor([0.2]), torch.Tensor([-0.4]), torch.Tensor([0.5]), torch.Tensor(
        [0.6]), torch.Tensor([0.1]), torch.Tensor([-0.5]), torch.Tensor([-0.3]), torch.Tensor([0.8])  # 权重初始值
    w1.requires_grad = True
    w2.requires_grad = True
    w3.requires_grad = True
    w4.requires_grad = True
    w5.requires_grad = True
    w6.requires_grad = True
    w7.requires_grad = True
    w8.requires_grad = True
    # print("=====更新前的权值=====")
    # print(w1.data, w2.data, w3.data, w4.data, w5.data, w6.data, w7.data, w8.data)
    eli = []
    lli = []
    for i in range(10):
        print("=====第" + str(i) + "轮=====")
        y1_pred, y2_pred = forward_propagate(x1, x2)  # 前向传播
        L = loss_fuction(y1_pred, y2_pred, y1, y2)  # 前向传播，求 Loss，构建计算图
        L.backward()  # 自动求梯度，不需要人工编程实现。反向传播，求出计算图中所有梯度存入w中
        # print("\tgrad W: ", round(w1.grad.item(), 2), round(w2.grad.item(), 2), round(w3.grad.item(), 2),
        #       round(w4.grad.item(), 2), round(w5.grad.item(), 2), round(w6.grad.item(), 2), round(w7.grad.item(), 2),
        #       round(w8.grad.item(), 2))
        w1, w2, w3, w4, w5, w6, w7, w8 = update_w(w1, w2, w3, w4, w5, w6, w7, w8)
        eli.append(i)
        lli.append(L.data.numpy())

    # print("更新后的权值")
    # print(w1.data, w2.data, w3.data, w4.data, w5.data, w6.data, w7.data, w8.data)

    plt.plot(eli, lli)
    plt.ylabel('Loss')
    plt.xlabel('w')
    plt.show()