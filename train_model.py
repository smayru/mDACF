#!/usr/bin/env python

import numpy as np
import chainer
from chainer import optimizers
import chainer.functions as F
import chainer.links as L
import codecs
import pandas as pd
import argparse
import pickle
import random

parser = argparse.ArgumentParser(description='Chainer example: MNIST')
parser.add_argument('--alpha', type=float, default="0.7",
                    help='The value of alpha')
parser.add_argument('--beta', type=float, default='0.004',
                    help='The beta of beta')
parser.add_argument('--lambda_val', type=float, default='0.2',
                    help='The value of lambda')
parser.add_argument('--corrupt_ratio', type=float, default='0.002',
                    help='The ratio for mSDA')
parser.add_argument('--Epoch', type=int, default='100',
                    help='The value of Epoch')
args = parser.parse_args()

############################
# data loading
############################
col_name1 = ['user_id', 'item_id', 'rating', 'timestamp']
col_name2 = ['user id', 'age', 'gender', 'occupation', 'zip code']
df_train = pd.read_csv('./ml-100K/u1.base', sep='\t', names=col_name1, index_col=False)
df_user = pd.read_csv('./ml-100K/u.user', sep='|', names=col_name2, index_col=False)

col_name3 = {0: "item_id", 1: "item_title", 5: "unknown", 6: "Action",
                7: "Adventure", 8: "Animation", 9: "Children", 10: "Comedy",
                11: "Crime", 12: "Documentary", 13: "Drama", 14: "Fantasy",
                15: "Film-Noir", 16: "Horror", 17: "Musical", 18: "Mystery",
                19: "Romance", 20: "Sci-Fi", 21: "Thriller", 22: "War",
                23: "Western"}
with codecs.open('./ml-100K/u.item', 'r', 'utf-8', errors='ignore') as f:
    df_item = pd.read_table(f, delimiter='|', header=None).ix[:, :]
    df_item.rename(columns=col_name3, inplace=True)

############################
# preprocessing
############################


def preprocessing(df, df_user, df_item):
    """preprocessing for the matrix R."""
    target_name = ["Action", "Adventure", "Animation",
                   "Children", "Comedy", "Crime", "Documentary", "Drama",
                   "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
                   "Romance", "Sci-Fi", "Thriller", "War", "Western"]

    item_detail = df_item[target_name]  # information about item
    item_detail.index = df_item["item_id"]

    occupation_list = sorted(list(set(df_user["occupation"])))
    occupation_dict = {}
    for i in range(len(occupation_list)):
        occupation_dict[occupation_list[i]] = i

    num_user = df_user.shape[0]
    num_occupation = len(occupation_list)

    user_detail = pd.DataFrame(np.zeros([num_user, num_occupation]), columns=occupation_list)# information abount user
    for i in range(num_user):
        user_detail.iloc[i, occupation_dict[df_user["occupation"].iloc[i]]] = 1
    user_detail.index = df_user["user id"]
    user_detail["gender"] = np.where(df_user["gender"] == "M", 1, 0)

    rating_detail = pd.DataFrame(np.empty([num_user, max(df_train["item_id"])]))# rating matrix
    mask_detail = pd.DataFrame(np.zeros([num_user, max(df_train["item_id"])]))# mask matrix
    rating_detail.index = df_user["user id"]
    rating_detail.columns = df_item["item_id"]
    mask_detail.index = df_user["user id"]
    mask_detail.columns = df_item["item_id"]
    for i in range(df.shape[0]):
        rating_detail.loc[df["user_id"].iloc[i], df["item_id"].iloc[i]] = df["rating"].iloc[i]
        mask_detail.loc[df["user_id"].iloc[i], df["item_id"].iloc[i]] = 1

    return [item_detail, user_detail, rating_detail, mask_detail]


[item_detail_train, user_detail_train, rating_detail_train, mask_detail_train] = preprocessing(df_train,df_user,df_item)


############################
# Define parameters
############################
random.seed(1)
training_ratio = 0.5
num_item = df_train.shape[0]
[m, p] = user_detail_train.shape
[n, q] = item_detail_train.shape
d = 10
R = rating_detail_train
X = np.transpose(user_detail_train)
Y = np.transpose(item_detail_train)
X = np.asarray(X, dtype=np.float32)
Y = np.asarray(Y, dtype=np.float32)

tmp = list(range(num_item))
random.shuffle(tmp)
train_mask = tmp[0:int(training_ratio * len(tmp))]
test_mask = tmp[int(training_ratio * len(tmp))::]
non_zero_element = np.where(mask_detail_train == 1)

A = np.zeros([m, n])
A[non_zero_element[0][train_mask], non_zero_element[1][train_mask]] = 1

A_test = np.zeros([m, n])  # mask matrix for evaluation
A_test[non_zero_element[0][test_mask], non_zero_element[1][test_mask]] = 1


# normalization
R_mean = np.mean(np.asarray(R)[np.where(A == 1)])
R_std = np.std(np.asarray(R)[np.where(A == 1)])
R = (R - R_mean) / R_std
R = np.asarray(R, dtype=np.float32)

alpha = args.alpha
beta = args.beta
lambda_val = args.lambda_val
corrupt_ratio = args.corrupt_ratio  # the ratio for mSDA
Epoch = args.Epoch

np.random.seed(100)
W1 = np.random.rand(p, p).astype(np.float32)
P1 = np.random.rand(p, d).astype(np.float32)
W2 = np.random.rand(q, q).astype(np.float32)
P2 = np.random.rand(q, d).astype(np.float32)

############################
# Update rules
############################

def update_P1(W_1, X, U):
    U = U.data
    a = np.dot(np.transpose(U), U)
    b = np.dot(np.dot(W1, X), U)
    a = np.transpose(a)
    b = np.transpose(b)
    return np.transpose(np.linalg.solve(a, b)).astype(np.float32)

def update_P2(W2, Y, V):
    V = V.data
    a = np.dot(np.transpose(V), V)
    b = np.dot(np.dot(W2, Y), V)
    a = np.transpose(a)
    b = np.transpose(b)
    return np.transpose(np.linalg.solve(a, b)).astype(np.float32)

def update_W1(X, lambda_val, corrupt_ratio, P1, U, p):
    U = U.data
    S1 = (1 - corrupt_ratio) * np.dot(X, np.transpose(X))
    S1 += lambda_val * np.dot(P1, np.dot(np.transpose(U), np.transpose(X)))
    Q1 = (1 - corrupt_ratio) * np.dot(X, np.transpose(X))
    tmp = (1 - corrupt_ratio) * (1 - corrupt_ratio) * (np.ones([p, p]) - np.diag(np.ones([p]))) * np.dot(X, np.transpose(X))
    tmp += (1 - corrupt_ratio) * np.diag(np.ones([p])) * np.dot(X, np.transpose(X))
    Q1 += lambda_val * tmp
    return np.linalg.solve(Q1, S1).astype(np.float32)

def update_W2(Y, lambda_val, corrupt_ratio, P2, V, q):
    V = V.data
    S2 = (1 - corrupt_ratio) * np.dot(Y, np.transpose(Y))
    S2 += lambda_val * np.dot(P2, np.dot(np.transpose(V), np.transpose(Y)))
    Q2 = (1 - corrupt_ratio) * np.dot(Y, np.transpose(Y))
    tmp = (1 - corrupt_ratio) * (1 - corrupt_ratio) * (np.ones([q, q]) - np.diag(np.ones([q]))) * np.dot(Y, np.transpose(Y))
    tmp += (1 - corrupt_ratio) * np.diag(np.ones([q])) * np.dot(Y, np.transpose(Y))
    Q2 += lambda_val * tmp
    return np.linalg.solve(Q2, S2).astype(np.float32)



class Model(chainer.Chain):

    def __init__(self, m, n, d):
        super(Model, self).__init__()
        with self.init_scope():
            self.u = L.Linear(d, m)
            self.v = L.Linear(d, n)

    def obtain_value(self, ):
        u = self.u.W.data
        v = self.v.W.data
        return [u, v]
    def obtain_loss(self, lambda_val, P1, W1, X, P2, W2, Y, A, R, alpha, beta):
        loss = 0
        loss += lambda_val * F.sum(F.square(F.matmul(P1, F.transpose(self.u.W)) - F.matmul(W1, X)))
        loss += lambda_val * F.sum(F.square(F.matmul(P2, F.transpose(self.v.W)) - F.matmul(W2, Y)))
        loss += alpha * F.sum(F.square(A * (R - F.matmul(self.u.W, F.transpose(self.v.W)))))
        loss += beta * ((F.sum(F.square(self.u.W)) + F.sum(F.square(self.v.W))))
        return loss


# model definition
model = Model(m, n, d)
optimizer = optimizers.SGD(0.002)
optimizer.setup(model)
U = model.u.W
V = model.v.W


for epoch in range(Epoch):
    W1 = update_W1(X, lambda_val, corrupt_ratio, P1, U, p)
    W2 = update_W2(Y, lambda_val, corrupt_ratio, P2, V, q)
    P1 = update_P1(W1, X, U)
    P2 = update_P2(W2, Y, V)
    loss = model.obtain_loss(lambda_val, P1, W1, X, P2, W2, Y, A, R, alpha, beta)

    model.zerograds()
    loss = model.obtain_loss(lambda_val, P1, W1, X, P2, W2, Y, A, R, alpha, beta)
    print("epoch", loss.data)
    loss.backward()
    optimizer.update()
    U = model.u.W
    V = model.v.W

output = [A,R,X,Y,W1,W2,P1,P2]

with open('output.dump', 'wb') as f:
    pickle.dump(output, f)
