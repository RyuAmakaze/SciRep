#Define Path 
load_path = "/a/yu-yamaoka/Scientific_reports/DINO_Output/TRAIN_CP_20231201_111120_Ours4MinSize=40_Crop=64_FromPreTrainModel/epoch15_output"
save_path = load_path
data_num = 70000
random_seed = 42

train_image_path = "/a/yu-yamaoka/Scientific_reports/Crop_Data/256_TRAIN_ver1206/TRAIN_CP_20231201_111120_Ours4MinSize=40_Crop=64"
test_image_path = "/a/yu-yamaoka/Scientific_reports/Crop_Data/256_TRAIN_ver1206/TEST_CP_20231201_111120_Ours4_MinSize=40_Crop=64/image"


import os
import torch
import numpy as np
import umap
import umap.plot
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from torchvision import datasets

# Get image paths
class ReturnIndexDataset(datasets.ImageFolder):
    def __getitem__(self, idx):
        img, lab = super(ReturnIndexDataset, self).__getitem__(idx)
        return img, idx



# Visualization
# カスタムカラーマップの作成
custom_colors = ['gray', 'green', 'violet', 'mediumturquoise', 'black', 'lightgreen', 'magenta', 'darkcyan', 'gainsboro', 'whitesmoke']
color_map = ListedColormap(custom_colors)

# タグのマッピング
# Folder名でDay0, Day3, ..., Day11, Day14になっているためPytorchがalphanumerical orderで読み込むときにDay0, Day11, Day14, Day3, Day5, Day7という順序になる
class_to_tag = {
    0: 'Day0',
    1: 'Day3',
    2: 'Day5',
    3: 'Day7',
    4: 'Day0-test',
    7: 'Day3-test',
    8: 'Day5-test',
    9: 'Day7-test',
    5: 'Day11-test',
    6: 'Day14-test',
    # Add more mappings if needed
}

# ラベルをタグに変換する関数
def map_label_to_tag(label):
    return [class_to_tag[l] for l in label]

# Visualization
def umap_visualization(data, label, filename, image_names=None):
    mapper = umap.UMAP(n_components=2).fit(data)
    embedding = mapper.transform(data)

    plt.figure(figsize=(10, 8))
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.title('UMAP Visualization with Custom Colors and Tags')

    
    # カスタムカラーマップを使用してデータをプロット
    for class_label in np.unique(label):
        plt.scatter(embedding[label == class_label, 0], embedding[label == class_label, 1],
                    c=[color_map(class_label)], label=f'{class_to_tag[class_label]}')
    
    plt.legend()
    plt.savefig(filename + ".png")
      
    if image_names is not None:
        # Mapping between UMAP points and image paths
        umap_image_names = dict(zip(range(len(image_names)), image_names))
        for i, (x,y) in enumerate(embedding):
            plt.annotate(umap_image_names[i], (x,y), fontsize=8, alpha=0.7)

    plt.savefig(filename + "_name.png")
    plt.show()

os.makedirs(save_path, exist_ok=True)


train_features = torch.load(os.path.join(load_path, "trainfeat.pth"))
train_labels = torch.load(os.path.join(load_path, "trainlabels.pth"))

test_features = torch.load(os.path.join(load_path, "testfeat.pth"))
test_labels = torch.load(os.path.join(load_path, "testlabels.pth"))
test_labels += 4

train_test_features = torch.cat((train_features, test_features), dim=0)
train_test_labels = torch.cat((train_labels, test_labels), dim=0)

# Get image filename list
dataset_train = ReturnIndexDataset(train_image_path, transform=None)
train_image_names = [os.path.basename(n)[:-4] for n, _ in dataset_train.imgs]
dataset_test = ReturnIndexDataset(test_image_path, transform=None)
test_image_names = [os.path.basename(n)[:-4] for n, _ in dataset_test.imgs]
train_test_names = train_image_names + test_image_names

np.random.seed(random_seed)
# Low memeory対策：Random Sampling
rand_idx_train_feat = np.random.permutation(len(train_features))
#print(np.take(train_image_names, rand_idx_train_feat)[0:3], train_labels[rand_idx_train_feat][0:3].detach().cpu())
umap_visualization(train_features[rand_idx_train_feat][0:data_num].detach().cpu(), train_labels[rand_idx_train_feat][0:data_num].detach().cpu(), 
                    os.path.join(save_path, f"train{str(data_num)}"), np.take(train_image_names, rand_idx_train_feat)[0:data_num])


rand_idx_train_test_feat = np.random.permutation(len(train_test_features))
umap_visualization(train_test_features[rand_idx_train_test_feat][0:data_num].detach().cpu(), train_test_labels[rand_idx_train_test_feat][0:data_num].detach().cpu(),
                    os.path.join(save_path, f"train_and_test{str(data_num)}"), np.take(train_test_names, rand_idx_train_test_feat)[0:data_num])

# Low memeory対策：Random Sampling
rand_idx = np.random.permutation(len(test_features))
umap_visualization(test_features[rand_idx][0:data_num].detach().cpu(), test_labels[rand_idx][0:data_num].detach().cpu(),
                    os.path.join(save_path, f"test{str(data_num)}"), np.take(test_image_names, rand_idx)[0:data_num])
