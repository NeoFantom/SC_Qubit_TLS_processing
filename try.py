# write a program that extracts the pixels into a set.

import numpy as np
import cv2

data_path = "./TLS_DATA_2024Nov_2025May.npy"
data = np.load(data_path, allow_pickle=True).item()
for i in range(0, 72):
    all_eqlen = len(data[i]["xs"]) == len(data[i]["ys"]) == len(data[i]["times"])
    print(f'{"all equal length" if all_eqlen else "not all equal length"}')
    all_list = (type(data[i]["xs"]) is list) and (type(data[i]["ys"]) is list) and (type(data[i]["times"]) is list)
    print(f'{"all lists" if all_list else "not all lists"}')
    # ls = set()
    # for j in range(len(data[i]["xs"])):
    #     x = data[i]["xs"][j]
    #     dx = x[1] - x[0]
    #     b, t = x[0] - dx / 2, x[-1] + dx / 2
    #     ls.add((b, t))
    # print(*ls, sep="\n")

# xs_lens = set()
# for q, qdata in data.items():
#     print(q)
#     print(type(qdata["xs"]))
#     xs_lens.add(len(qdata["xs"]))

# xlens = set()
# for q, qdata in data.items():
#     for x in qdata["xs"]:
#         xlens.add(len(x))

# print(f'{xs_lens = }')
# print(f'{xlens = }')