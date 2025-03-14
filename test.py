import argparse
from collections import Counter
import csv
import json
import random
import scipy.sparse as sp
import torch
import dgl
import numpy as np
import pandas as pd
import os
from datetime import datetime
from models import HoLe
from utils import check_modelfile_exists
from utils import csv2file
from utils import evaluation
from utils import get_str_time
from utils import set_device
from feature_test.text.text import getText
from feature_test.graph.graph import getGraph
import Levenshtein
from bs4 import BeautifulSoup
import zss
from sklearn.cluster import KMeans
import itertools

def load_custom_data(adj_matrix, features, labels=None, n_clusters=20):
    graph = dgl.from_scipy(adj_matrix)

    if isinstance(features, dict):
        features = torch.stack([features[page_id] for page_id in sorted(features.keys())])
    
    graph.ndata["feat"] = features
    
    if labels is None:
        labels = torch.zeros(graph.num_nodes(), dtype=torch.long)
    else:
        labels = torch.tensor(labels, dtype=torch.long) 
    return graph, labels, n_clusters

def load_adj_matrix(file_path):
    adj_matrix = np.loadtxt(file_path)
    adj_sparse = sp.csr_matrix(adj_matrix)
    return adj_sparse

def save_to_txt(file_path, folder, elapsed_time, sampled_page_ids, avg_distance):
    with open(file_path, 'w') as f:
        f.write(f"Folder: {folder}\n")
        f.write(f"Elapsed Time: {elapsed_time}\n\n")
        f.write("\nSampled Indices:\n")
        f.write(", ".join(map(str, sampled_page_ids)) + "\n")
        f.write(f"平均编辑距离: {avg_distance}\n")

def save_edit_matrices(folder, cluster_matrices, mean_distance):
    """保存编辑距离矩阵"""
    os.makedirs("./results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"./results/edit_matrices_{folder}_{timestamp}.txt"
    with open(filename, 'w') as f:
        f.write(f"全局平均编辑距离: {mean_distance:.4f}\n\n")
        for cluster_idx, matrix in cluster_matrices.items():
            f.write(f"聚类 {cluster_idx} 的编辑距离矩阵 ({matrix.shape[0]}x{matrix.shape[1]}):\n")
            np.savetxt(f, matrix, fmt='%d')
            f.write("\n\n")

def extract_html_structure(html_path):
    """提取HTML的标签结构"""
    try:
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        soup = BeautifulSoup(html_content, 'html.parser')
        root = soup.html or soup.body
        
        def build_tree(element):
            if not element or element.name is None:
                return zss.Node("__null__")
            node = zss.Node(element.name)
            for child in element.children:
                if child.name is not None:
                    node.addkid(build_tree(child))
            return node
        
        return build_tree(root) if root else zss.Node("__empty__")
    except Exception as e:
        print(f"Error parsing {html_path}: {e}")
        return zss.Node("__error__")

def extract_dom_text(html_path):
    """提取dom文本内容"""
    try:
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        soup = BeautifulSoup(html_content, 'html.parser')
        return soup.get_text(separator=' ', strip=True)
    except Exception as e:
        print(f"Error extracting text from {html_path}: {e}")
        return ""

def sample_pages(cluster_labels, features, page_ids):
    """
    选择距离聚类中心最近的页面
    """
    unique_clusters = np.unique(cluster_labels)
    sampled_page_ids = []

    for cluster in unique_clusters:
        indices = np.where(cluster_labels == cluster)[0]
        if len(indices) == 0:
            continue
        
        indices_tensor = torch.from_numpy(indices).long()
        cluster_features = features[indices_tensor]
        
        # 计算聚类中心
        center = torch.mean(cluster_features, dim=0)
        
        distances = torch.norm(cluster_features - center, dim=1)
        closest_idx = torch.argmin(distances)
        sampled_page_ids.append(page_ids[indices[closest_idx]])

    return sampled_page_ids


"""完整流程"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="HoLe",
        description="Homophily-enhanced Structure Learning for Graph Clustering",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="custom",
        help="Dataset used in the experiment",
    )
    parser.add_argument(
        "-g",
        "--gpu_id",
        type=int,
        default=0,
        help="gpu id",
    )
    args = parser.parse_args()

    base_path = './feature_test'

    with open('/developer/laisicen/code/HoLe/resource/rule_ids.txt', 'r') as file:
        rule_ids = [line.strip() for line in file]
    # 读取 axe 到 wcag 的映射关系
    rule_reflect = {}
    with open('/developer/laisicen/code/HoLe/resource/rule_reflect.csv', 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            if len(row) >= 2:
                axe_rule, wcag_rule = row[0], row[1]
                rule_reflect[axe_rule] = wcag_rule
            else:
                continue
        
    dom_path = os.path.join(base_path, 'UIST_DOMData')
    """
    文件位置:
      源代码和截图: /UIST_DOMData/{folders}/{hctask_id}-xxx/{page_id}   (.jpg)
      axe结果: /UIST_axeData/axe_res/{hctask_id}/{page_id}
      邻接矩阵: /UIST_GraphData/{hctask_id}/adj_matrix.txt
    """
    folders = [f for f in os.listdir(dom_path) if os.path.isdir(os.path.join(dom_path, f))]
    for folder in folders:
        start_time = datetime.now()
        folder_path = os.path.join(dom_path, folder) # ./feature_test/UIST_DOMData/北京市残疾人联合会_20241225054409
        
        subfolderName = next((d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))), None)
        subfolder = os.path.join(folder_path, subfolderName)
        if subfolderName:
            subfolderName = str(subfolderName) 
            print(subfolderName)
        else:
            print(f"{folder}中没有对应的网页源文件，跳过")
            continue
        
        hctask_id = subfolderName.split('-')[0]
        page_ids = []
        page_features = {}
        page_feature_strs = {}
        
        axe_base_path = os.path.join(base_path, f'UIST_axeData/axe_res/{hctask_id}')
        adj_matrix_path = os.path.join(base_path, f'UIST_GraphData/{hctask_id}/adj_matrix.txt')
        
        for fname in os.listdir(subfolder):
            if fname.lower().endswith((".jpg", ".logs", ".log")):
                continue
            page_id = fname
            page_ids.append(page_id)

            """开始处理文本特征"""
            html_path = os.path.join(subfolder, f'{page_id}')
            # print(html_path)
            text_feature = np.zeros(768)
            if os.path.exists(html_path):
                try:
                    text_feature = getText(html_path)
                    print(f"文本特征提取成功")
                except Exception as e:
                    print(f"文本特征提取失败({html_path}): {e}")
            
            """开始处理图像特征"""
            jpg_path = os.path.join(subfolder, f'{page_id}.jpg')
            graph_feature = np.zeros(768)
            if os.path.exists(jpg_path):
                try:
                    graph_feature = getGraph(jpg_path)
                    print(f"图像特征提取成功")
                except Exception as e:
                    print(f"图像特征提取失败({jpg_path}): {e}")

            """开始处理规则结果特征"""
            axe_path = os.path.join(axe_base_path, f'{page_id}.json')
            with open(axe_path, 'r') as file:
                data = json.load(file)
            ids = []
            for key in ['incomplete', 'violations', 'inapplicable']:
                ids.extend([item['id'] for item in data.get(key, [])])
            mapped_ids = []
            for axe_id in ids:
                if axe_id in rule_reflect:
                    mapped_ids.append(rule_reflect[axe_id])
                else:
                    mapped_ids.append(axe_id)     
            id_counts = Counter(mapped_ids)
            rule_feature = [id_counts.get(rule_id, 0) for rule_id in rule_ids]  
            
            combined_feature = np.concatenate([rule_feature, text_feature, graph_feature])
            combined_feature = combined_feature / np.linalg.norm(combined_feature)
            page_features[page_id] = torch.tensor(combined_feature, dtype=torch.float32)

        adj_matrix = load_adj_matrix(adj_matrix_path)
        features = page_features
        labels = None
        n_clusters = 20

        graph, labels, n_clusters = load_custom_data(adj_matrix, features, labels, n_clusters)
        features = graph.ndata["feat"]

        final_params = {}
        dim = 500
        n_lin_layers = 1
        dump = True
        device = set_device(str(args.gpu_id))
        lr = 0.001  # 学习率
        n_gnn_layers = [8]  # GNN 层数
        pre_epochs = [150]  # 预训练轮数
        epochs = 50  # 训练轮数
        inner_act = lambda x: x  # 激活函数
        udp = 10  # UDP 参数
        node_ratios = [1]  # 节点采样比例
        add_edge_ratio = 0.5  # 添加边的比例
        del_edge_ratios = [0.01]  # 删除边的比例
        gsl_epochs_list = [10]  # GSL 轮数
        regularization = 1  # 正则化参数

        for gsl_epochs in gsl_epochs_list:
            runs = 10
            for n_gnn_layer in n_gnn_layers:
                for pre_epoch in pre_epochs:
                    for del_edge_ratio in del_edge_ratios:
                        for node_ratio in node_ratios:
                            final_params["dim"] = dim
                            final_params["n_gnn_layers"] = n_gnn_layer
                            final_params["n_lin_layers"] = n_lin_layers
                            final_params["lr"] = lr
                            final_params["pre_epochs"] = pre_epoch
                            final_params["epochs"] = epochs
                            final_params["udp"] = udp
                            final_params["inner_act"] = inner_act
                            final_params["add_edge_ratio"] = add_edge_ratio
                            final_params["node_ratio"] = node_ratio
                            final_params["del_edge_ratio"] = del_edge_ratio
                            final_params["gsl_epochs"] = gsl_epochs

                            time_name = get_str_time()
                            save_file = f"results/hole/hole_custom_gnn_{n_gnn_layer}_gsl_{gsl_epochs}_{time_name[:9]}_{folder}.csv"

                            warmup_filename = f"hole_custom_run_gnn_{n_gnn_layer}"

                            if not check_modelfile_exists(warmup_filename):
                                print("warmup first")
                                model = HoLe(
                                    hidden_units=[dim],
                                    in_feats=features.shape[1],
                                    n_clusters=n_clusters,
                                    n_gnn_layers=n_gnn_layer,
                                    n_lin_layers=n_lin_layers,
                                    lr=lr,
                                    n_pretrain_epochs=pre_epoch,
                                    n_epochs=epochs,
                                    norm="sym",
                                    renorm=True,
                                    tb_filename=f"custom_gnn_{n_gnn_layer}_node_{node_ratio}_{add_edge_ratio}_{del_edge_ratio}_pre_ep{pre_epoch}_ep{epochs}_dim{dim}_{random.randint(0, 999999)}",
                                    warmup_filename=warmup_filename,
                                    inner_act=inner_act,
                                    udp=udp,
                                    regularization=regularization,
                                )

                                model.fit(
                                    graph=graph,
                                    device=device,
                                    add_edge_ratio=add_edge_ratio,
                                    node_ratio=node_ratio,
                                    del_edge_ratio=del_edge_ratio,
                                    gsl_epochs=0,
                                    labels=labels,
                                    adj_sum_raw=adj_matrix.sum(),
                                    load=False,
                                    dump=dump,
                                )

                            seed_list = [random.randint(0, 999999) for _ in range(runs)]
                            for run_id in range(runs):
                                final_params["run_id"] = run_id
                                seed = seed_list[run_id]
                                final_params["seed"] = seed

                                model = HoLe(
                                    hidden_units=[dim],
                                    in_feats=features.shape[1],
                                    n_clusters=n_clusters,
                                    n_gnn_layers=n_gnn_layer,
                                    n_lin_layers=n_lin_layers,
                                    lr=lr,
                                    n_pretrain_epochs=pre_epoch,
                                    n_epochs=epochs,
                                    norm="sym",
                                    renorm=True,
                                    tb_filename=f"custom_gnn_{n_gnn_layer}_node_{node_ratio}_{add_edge_ratio}_{del_edge_ratio}_gsl_{gsl_epochs}_pre_ep{pre_epoch}_ep{epochs}_dim{dim}_{random.randint(0, 999999)}",
                                    warmup_filename=warmup_filename,
                                    inner_act=inner_act,
                                    udp=udp,
                                    reset=False,
                                    regularization=regularization,
                                    seed=seed,
                                )

                                model.fit(
                                    graph=graph,
                                    device=device,
                                    add_edge_ratio=add_edge_ratio,
                                    node_ratio=node_ratio,
                                    del_edge_ratio=del_edge_ratio,
                                    gsl_epochs=gsl_epochs,
                                    labels=labels,
                                    adj_sum_raw=adj_matrix.sum(),
                                    load=False,
                                    dump=dump,
                                )

                                # 采样
                                with torch.no_grad():
                                    z_detached = model.get_embedding()
                                    z_np = z_detached.detach().cpu().numpy()
                                    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=100, random_state=42)
                                    q = kmeans.fit_predict(z_np)
                                    print("聚类结果:", np.unique(q))

                                    sampled_page_ids = sample_pages(q, features, page_ids)
                                    print("Sampled page IDs:", sampled_page_ids)

                                    """
                                    评估指标：
                                    1.纯 HTML 标签 structure，使用 树编辑距离（Tree Edit Distance, TED） ，体现页面 layout 差异代表性
                                    2.纯 DOM 内的文本，使用 Levenshtein 编辑距离，体现页面文本内容差异代表性
                                    """
                                    trees = {}
                                    texts = {}
                                    for page_id in sampled_page_ids:
                                        html_path = os.path.join(subfolder, page_id)
                                        try:
                                            trees[page_id] = extract_html_structure(html_path)
                                            texts[page_id] = extract_dom_text(html_path)
                                        except Exception as e:
                                            print(f"页面处理失败 [{page_id}]: {e}")

                                    # 计算所有代表页面之间的差异
                                    ted_distances = []
                                    lev_distances = []
                                    page_pairs = []

                                    # 生成所有唯一配对组合
                                    all_pairs = list(itertools.combinations(sampled_page_ids, 2))

                                    for page1, page2 in all_pairs:
                                        print(f"page1:{page1}, page2:{page2}")
                                        if page1 not in trees or page2 not in trees:
                                            continue
                                        
                                        # 计算树编辑距离
                                        try:
                                            ted = zss.simple_distance(trees[page1], trees[page2])
                                            ted_distances.append(ted)
                                        except Exception as e:
                                            print(f"TED计算失败 [{page1} vs {page2}]: {e}")
                                        
                                        # 计算文本编辑距离
                                        try:
                                            lev = Levenshtein.distance(texts[page1], texts[page2])
                                            lev_distances.append(lev)
                                        except Exception as e:
                                            print(f"Levenshtein计算失败 [{page1} vs {page2}]: {e}")

                                    avg_ted = np.mean(ted_distances) if ted_distances else 0
                                    avg_lev = np.mean(lev_distances) if lev_distances else 0
                                    combined_score = 0.75 * avg_ted + 0.25 * avg_lev

                                    print("\n聚类间差异评估结果:")
                                    print(f"平均布局差异 (TED): {avg_ted:.2f}")
                                    print(f"平均文本差异 (Levenshtein): {avg_lev:.2f}")
                                    print(f"综合评估指标: {combined_score:.2f}")


                                end_time = datetime.now()
                                elapsed_time = end_time - start_time
                                print(f"程序总耗时-{elapsed_time}")
                                
                                # save_to_txt(save_file, folder, elapsed_time, sampled_page_ids, mean_distance)
                                # print(f"write to {save_file}")
