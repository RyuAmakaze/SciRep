#Define Path 
load_path = "/a/yu-yamaoka/Scientific_reports/DINO_Output/TRAIN_CP_20231201_111120_Ours4MinSize=40_Crop=128/epoch95_output"
save_path = load_path
data_num = 170000

import os
import torch
import numpy as np
import umap
import umap.plot
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Visualization
# カスタムカラーマップの作成
custom_colors = ['gray', 'green', 'violet', 'mediumturquoise', 'black', 'darkgreen', 'magenta', 'darkcyan', 'gainsboro', 'whitesmoke']
color_map = ListedColormap(custom_colors)

# タグのマッピング
class_to_tag = {
    0: 'Day0',
    1: 'Day3',
    2: 'Day5',
    3: 'Day7',
    4: 'Day0-test',
    5: 'Day3-test',
    6: 'Day5-test',
    7: 'Day7-test',
    8: 'Day11-test',
    9: 'Day14-test',
    # Add more mappings if needed
}

# ラベルをタグに変換する関数
def map_label_to_tag(label):
    return [class_to_tag[l] for l in label]

# Visualization
def umap_visualization(data, label, filename):
    mapper = umap.UMAP(n_components=2).fit(data)
    embedding = mapper.transform(data)

    plt.figure(figsize=(8, 6))

    # カスタムカラーマップを使用してデータをプロット
    for class_label in np.unique(label):
        plt.scatter(embedding[label == class_label, 0], embedding[label == class_label, 1],
                    c=[color_map(class_label)], label=f'{class_to_tag[class_label]}')

    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.title('UMAP Visualization with Custom Colors and Tags')
    plt.legend()
    plt.savefig(filename + ".png")
    plt.show()

os.makedirs(save_path, exist_ok=True)


train_features = torch.load(os.path.join(load_path, "trainfeat.pth"))
train_labels = torch.load(os.path.join(load_path, "trainlabels.pth"))

test_features = torch.load(os.path.join(load_path, "testfeat.pth"))
test_labels = torch.load(os.path.join(load_path, "testlabels.pth"))
test_labels += 4

train_test_features = torch.cat((train_features, test_features), dim=0)
train_test_labels = torch.cat((train_labels, test_labels), dim=0)

    
# Low memeory対策：Random Sampling
rand_idx = np.random.permutation(len(train_features))
umap_visualization(train_features[rand_idx][0:data_num].detach().cpu(), train_labels[rand_idx][0:data_num].detach().cpu(),
                    os.path.join(save_path, "train"+str(data_num)))

rand_idx = np.random.permutation(len(train_test_features))
umap_visualization(train_test_features[rand_idx][0:data_num].detach().cpu(), train_test_labels[rand_idx][0:data_num].detach().cpu(),
                    os.path.join(save_path, "train_and_labels"+str(data_num)))

# Low memeory対策：Random Sampling
rand_idx = np.random.permutation(len(test_features))
umap_visualization(test_features[rand_idx][0:data_num].detach().cpu(), test_labels[rand_idx][0:data_num].detach().cpu(),
                    os.path.join(save_path, "test"+str(data_num)))

