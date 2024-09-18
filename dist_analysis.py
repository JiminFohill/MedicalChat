# import torch
from torch.utils.data import DataLoader
from dataloaders.data_helper_ import TextLoader,collate_fn
from dataloaders.preprocess import load_eval_json
from transformers.trainer import get_scheduler
from accelerate import notebook_launcher
from accelerate import Accelerator
import torch
from adapter_model import AdapterModel
from tqdm import tqdm
from utils.matrix import eval_matrix
import math
import os
from transformers import AutoModel, AutoTokenizer, AutoConfig
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
model_path = '/root/Documents/mChatAdapter/chatglm2'
adapter_path = './adapter_parallel_m_205000.pth'
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, config=config, trust_remote_code=True).cuda()
weights = torch.load(adapter_path)

adapter_weight = [
    'transformer.encoder.layers.0.adapter_encoder.dense_h_to_4h.weight',
    'transformer.encoder.layers.0.adapter_encoder.dense_4h_to_h.weight',
    'transformer.encoder.layers.27.adapter_encoder.dense_h_to_4h.weight',
    'transformer.encoder.layers.27.adapter_encoder.dense_4h_to_h.weight']

copy_weights_source = [
    'transformer.encoder.layers.0.mlp.dense_h_to_4h.weight',
    'transformer.encoder.layers.0.mlp.dense_4h_to_h.weight',
    'transformer.encoder.layers.27.mlp.dense_h_to_4h.weight',
    'transformer.encoder.layers.27.mlp.dense_4h_to_h.weight']

for aw, mw in zip(adapter_weight, copy_weights_source):
    mw_ = model.get_parameter(mw)
    print(mw_.shape)
    aw_ = weights[aw]
    print(aw_.shape)
    dist = torch.cosine_similarity(mw_.view(-1), aw_.view(-1),0)
    cosine = torch.cosine_similarity(mw_, aw_, 1)
    print(aw, mw, dist)

    w1_np = mw_.cpu().detach().numpy()
    w2_np = aw_.cpu().detach().numpy()
    cosine = cosine.cpu().detach().numpy()
    # Ê¹ÓÃPCA½«Ã¿¸öÈ¨ÖØ¾ØÕó½µÎ¬µ½2DÒÔ±ã¿ÉÊÓ»¯
    # pca = PCA(n_components=2)
    # w1_pca = pca.fit_transform(w1_np)
    # w2_pca = pca.transform(w2_np)  # ×¢ÒâÕâÀïÊ¹ÓÃtransform¶ø²»ÊÇfit_transform£¬ÒòÎªÎÒÃÇÒÑ¾­ÄâºÏÁËÄ£ÐÍ
    #
    # # ¿ÉÊÓ»¯PCA½á¹û
    # plt.figure(figsize=(10, 5))
    #
    # # »æÖÆ w1 µÄPCA½á¹û
    # plt.subplot(1, 2, 1)
    # plt.scatter(w1_pca[:, 0], w1_pca[:, 1], label='w1')
    # plt.title('PCA of Weights: '+aw)
    # plt.xlabel('PCA Component 1')
    # plt.ylabel('PCA Component 2')
    # plt.legend()
    #
    # # »æÖÆ w2 µÄPCA½á¹û
    # plt.subplot(1, 2, 2)
    # plt.scatter(w2_pca[:, 0], w2_pca[:, 1], label='w2')
    # plt.title('PCA of Weights: '+mw)
    # plt.xlabel('PCA Component 1')
    # plt.ylabel('PCA Component 2')
    # plt.legend()
    #
    # plt.tight_layout()
    # plt.show()

    pca = PCA(n_components=2)
    reduced_data1 = pca.fit_transform(w1_np)
    reduced_data2 = pca.transform(w2_np)

    cosine_sims_matrix = cosine_similarity(w1_np, w2_np)

    # ÌáÈ¡¶Ô½ÇÏßÉÏµÄÔªËØ£¬¼´¶ÔÓ¦ÐÐÖ®¼äµÄÏàËÆ¶È
    print(cosine.shape)
    cosine_sims = cosine #np.diag(cosine)
    #threshold = np.mean(cosine_sims) + np.std(cosine_sims)
    # cosine_sims = (cosine_sims+1)/2
    # print(cosine_sims.shape)
    # print(cosine_sims)
    # Éè¶¨Ò»¸öãÐÖµÀ´Çø·ÖÏàËÆºÍ²»ÏàËÆµÄµã
    # ÀýÈç£¬ÎÒÃÇ¿ÉÒÔÉè¶¨ãÐÖµÎª0.5£¬ÕâÒâÎ¶×ÅÓàÏÒÏàËÆ¶ÈµÍÓÚ0.5µÄµã½«±»ÊÓÎª²»ÏàËÆ
    threshold = 0.8

    # ±ê¼Ç²»ÏàËÆµÄµã
    # cosine_sims < threshold »á·µ»ØÒ»¸ö²¼¶ûÊý×é£¬ÆäÖÐFalse±íÊ¾ÏàËÆ£¬True±íÊ¾²»ÏàËÆ
    dissimilar_points = cosine_sims < threshold
    similar_points = cosine_sims >= threshold
    print(np.sum(similar_points))
    print(np.sum(dissimilar_points))
    # ¿ÉÊÓ»¯½µÎ¬ºóµÄÊý¾Ýµã£¬²»ÏàËÆµÄµãÓÃºìÉ«±ê¼Ç
    plt.figure(figsize=(10, 6))
    plt.scatter(reduced_data1[similar_points, 0], reduced_data1[similar_points, 1],
                label='Similar Points',  color='green', alpha=0.5)
    plt.scatter(reduced_data2[dissimilar_points, 0], reduced_data2[dissimilar_points, 1],
               label='Dissimilar Points', color='red', alpha=0.5)


    # ×¢Òâ£ºÎÒÃÇÐèÒªÊ¹ÓÃ.any(axis=1)À´¼ì²éÃ¿Ò»ÐÐÖÐÊÇ·ñÓÐTrue£¨¼´²»ÏàËÆµã£©
    # Èç¹ûÓÐÈÎºÎTrue£¬ÎÒÃÇ¾Í½«¸Ãµã±ê¼ÇÎª²»ÏàËÆ£¬·ñÔò±ê¼ÇÎªÏàËÆ

    plt.legend()
    plt.title('PCA Visualization with Cosine Similarity and Dissimilar Points Marked')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.show()
# weights1 = torch.rand(1024, 10000)
# weights2 = torch.rand(1024, 10000)

# # keys = list(weights.keys())
# # weights1 = weights[keys[0]]
# # weights2 = weights[keys[1]]
#
# dist = torch.dist(weights1, weights2)
# print(dist)
#
# import matplotlib.pyplot as plt
# import numpy as np
#
# # 将距离转换为 numpy 数组以便使用 matplotlib
# dist_np = dist.numpy()
#
# # 创建直方图
# plt.hist(dist_np, bins=30, density=True, alpha=0.6, color='g')
# plt.title('Histogram of Euclidean distances')
# plt.xlabel('Distance')
# plt.ylabel('Probability')
# plt.show()


# import torch
# import numpy as np
# adapter_path = '../paralleling_prefix_adapter_s/adapter_205000.pth'
# weights = torch.load(adapter_path,map_location=torch.device('cpu'))
# print(weights)
# # 假设 weights1 和 weights2 是你的权重矩阵
# weights1 = weights['transformer.encoder.layers.0.adapter_encoder.dense_h_to_4h.weight']
# weights2 = weights['transformer.encoder.layers.27.adapter_encoder.dense_h_to_4h.weight']
#
# # 将 PyTorch 张量转换为 NumPy 数组
# weights1_np = weights1.cpu().numpy()
# weights2_np = weights2.cpu().numpy()
#
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
#
# # 设置 t-SNE 参数。你可以根据需要进行调整
# tsne = TSNE(n_components=2, random_state=0)
#
# # 对两个权重矩阵进行 t-SNE 降维
# weights1_2d = tsne.fit_transform(weights1_np)
# weights2_2d = tsne.fit_transform(weights2_np)
#
# # 可视化权重矩阵的 t-SNE 结果
# plt.figure(figsize=(6, 6))
# plt.scatter(weights1_2d[:, 0], weights1_2d[:, 1], c='b', label='weights1')
# plt.scatter(weights2_2d[:, 0], weights2_2d[:, 1], c='r', label='weights2')
# plt.legend()
# plt.show()

#transformer.encoder.layers.0.adapter_encoder.dense_h_to_4h.weight transformer.encoder.layers.0.mlp.dense_h_to_4h.weight 0.9889

#transformer.encoder.layers.0.adapter_encoder.dense_4h_to_h.weight transformer.encoder.layers.0.mlp.dense_4h_to_h.weight 0.9894

#transformer.encoder.layers.27.adapter_encoder.dense_h_to_4h.weight transformer.encoder.layers.27.mlp.dense_h_to_4h.weight 0.9927

#transformer.encoder.layers.27.adapter_encoder.dense_4h_to_h.weight transformer.encoder.layers.27.mlp.dense_4h_to_h.weight 0.9932
