import h5py
import scipy.io as io
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
import scipy
from image import *
import tqdm

#this is borrowed from https://github.com/davideverona/deep-crowd-counting_crowdnet
def gaussian_filter_density(gt):
    # print( gt.shape)
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density

    pts = np.array((np.nonzero(gt)[1], np.nonzero(gt)[0])).T # 非零坐标
    leafsize = 2048
    # build kdtree
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize) # 使用这种数据结构方便后续查找最近的点
    # query kdtree
    distances, locations = tree.query(pts, k=4) # 查找最近的4个点，为了后续进行高斯滤波，作为sigma

    # print('generate density...') # 使用呢迭代的方式生成 density, 虽然pt2d整体上和gt一样都是images.shape生成的0，1矩阵，但是gt是共现的，pt2d只针对某个非零坐标出现非零值。因为是在循环里面
    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1],pt[0]] = 1.
        if gt_count > 1:
            if any([distances[i][1]==np.inf,distances[i][2]==np.inf,distances[i][3]==np.inf]): # why the np.inf could happen?
                sigma = np.average(np.array(gt.shape))/2./2.
            else:
                sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
        else:
            sigma = np.average(np.array(gt.shape))/2./2. #case: 1 point
        try:
            density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
        except:
            import pdb;pdb.set_trace()
    # print( 'done.')
    return density

def batch_process(path_sets):
    img_paths = []
    for path in path_sets:
        for img_path in glob.glob(os.path.join(path, '*.jpg')):
            img_paths.append(img_path)

    for img_path in tqdm.tqdm(img_paths):
        # print(img_path)
        mat = io.loadmat(img_path.replace('.jpg','.mat').replace('images','ground_truth').replace('IMG_','GT_IMG_'))
        img= plt.imread(img_path)
        k = np.zeros((img.shape[0],img.shape[1])) # iamges.shape with 0,1 
        gt = mat["image_info"][0,0][0,0][0]
        for i in range(0,len(gt)):
            if int(gt[i][1])<img.shape[0] and int(gt[i][0])<img.shape[1]:
                k[int(gt[i][1]),int(gt[i][0])]=1
        # import pdb;pdb.set_trace()
        k = gaussian_filter_density(k)
        with h5py.File(img_path.replace('.jpg','.h5').replace('images','ground_truth'), 'w') as hf:
                hf['density'] = k

def batch_process_json(path_sets):
    img_paths = []
    for path in path_sets:
        for img_path in glob.glob(os.path.join(path, '*.jpg')):
            img_paths.append(img_path)

    for img_path in tqdm.tqdm(img_paths):
        # print(img_path)
        gt = np.load(img_path.replace('.jpg','.npy').replace('images','gtjson'))
        # import pdb;pdb.set_trace()
        img= plt.imread(img_path)
        k = np.zeros((img.shape[0],img.shape[1])) # iamges.shape with 0,1 
        for i in range(0,len(gt)):
            if int(gt[i][1])<img.shape[0] and int(gt[i][0])<img.shape[1]:
                k[int(gt[i][1]),int(gt[i][0])]=1
        # import pdb;pdb.set_trace()
        k = gaussian_filter_density(k)
        np.save(img_path.replace(".jpg", ".npy").replace("images", "ground_truth"), k)

def json2npy():
    import numpy as np
    import json

    path = "/workspace/pap/data/comp/test_data/gtjson"
    json_paths = []
    for img_path in glob.glob(os.path.join(path, '*.json')):
        json_paths.append(img_path)
    print(json_paths)


    for file_path in json_paths:
        print(file_path)
        # Open the JSON file and read its contents into a variable
        with open(file_path, 'r') as file:
            data = json.load(file)

        points_list = []

        # Iterate over each shape and extract the points
        for shape in data["shapes"]:
            points = shape.get("points", [])
            points_list.append(points)

        # Convert the list of points to a NumPy array
        points_array = np.array(points_list).reshape(-1,2)
        # print(points_array.shape)
        np.save(file_path.replace("gtjson", "gtjson").replace(".json", ".npy"), points_array)

if __name__ == "__main__":
    # #set the root to the Shanghai dataset you download
    # root = '/workspace/pap/data/attack_shanghai'

    # #now generate the ShanghaiA's ground truth
    # part_A_train = os.path.join(root,'part_A_final/train_data','images')
    # part_A_test = os.path.join(root,'part_A_final/test_data','images')
    # part_B_train = os.path.join(root,'part_B_final/train_data','images')
    # part_B_test = os.path.join(root,'part_B_final/test_data','images')

    # path_sets = [part_A_train,part_A_test]

    # batch_process(path_sets)

    # #now generate the ShanghaiB's ground truth
    # path_sets = [part_B_train,part_B_test]

    # batch_process(path_sets)

    ############ for comp #############

    # 1. json 2 npy
    # json2npy()

    # 2. gausion filter
    data_root = "/workspace/pap/data/comp"
        
    part_A_train = os.path.join(data_root,'train_data','images')
    part_A_test = os.path.join(data_root,'test_data','images')
    path_sets = [part_A_train]
    batch_process_json(path_sets)