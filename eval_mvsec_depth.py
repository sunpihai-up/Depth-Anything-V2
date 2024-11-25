import json
import numpy as np
import pandas as pd
import os

class EvalResults:
    def __init__(self):
        self.files = []
        self.results = []

    def append(self, fpath, result):
        self.files.append(fpath)
        self.results.append(result)

    def mean(self):
        num = len(self.results)
        if num == 0:
            return self.results

        output = {}
        for key in self.results[0].keys():
            value = np.mean([ dic[key] for dic in self.results ])
            output[key] = value

        return output

    def numpy(self):
        results = [ list(dic.values()) for dic in self.results ]
        return np.array(results)

    def dataframe(self):
        columns = list(self.results[0].keys())
        data = self.numpy()
        return pd.DataFrame(data, index=self.files, columns=columns)

    def to_csv(self, fpath_out):
        df = self.dataframe()
        df.to_csv(fpath_out)

def align_depth_least_square(
    gt_arr: np.ndarray,
    pred_arr: np.ndarray,
    return_scale_shift=True,
):
    ori_shape = pred_arr.shape  # input shape

    gt = gt_arr.squeeze()  # [H, W]
    pred = pred_arr.squeeze()

    assert (
        gt.shape == pred.shape
    ), f"{gt.shape}, {pred.shape}"

    gt_masked = gt.reshape((-1, 1))
    pred_masked = pred.reshape((-1, 1))

    # numpy solver
    _ones = np.ones_like(pred_masked)
    A = np.concatenate([pred_masked, _ones], axis=-1)
    X = np.linalg.lstsq(A, gt_masked, rcond=None)[0]
    scale, shift = X

    aligned_pred = pred_arr * scale + shift

    # restore dimensions
    aligned_pred = aligned_pred.reshape(ori_shape)

    if return_scale_shift:
        return aligned_pred, scale, shift
    else:
        return aligned_pred

def load_path_list(prediction_dir, gt_dir, json_path):
    depth2image_dict = {}
    with open(json_path, 'r') as f:
        depth2image_dict = json.load(f)
    
    prediction_files = []
    gt_files = []
    
    for depth_id, image_dict in depth2image_dict.items():
        prediction_files.append(os.path.join(prediction_dir, image_dict['image_id'] + '.npy'))
        gt_files.append(os.path.join(gt_dir, depth_id + '.npy'))
        
    return prediction_files, gt_files

def load_path_list_from_txt(split_path, prediction_dir):
    prediction_files = []
    gt_files = []
    
    with open(split_path, 'r') as f:
        lines = f.readlines()
        prediction_files = [os.path.join(prediction_dir, os.path.basename(line.split()[0])).replace('png', 'npy') for line in lines]
        gt_files = [line.strip().split()[1] for line in lines]

    return prediction_files, gt_files

def _evaluate(pred, gt, max_depth, min_depth, clip_pred, suffix=''):
    mask = (gt >= min_depth) & (gt <= max_depth)
    pred = pred[mask]
    gt = gt[mask]

    # Align depth (least square)
    pred, scale, shift = align_depth_least_square(gt, pred, return_scale_shift=True)
    
    if clip_pred:
        pred = pred.clip(min_depth, max_depth)

    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    diff = gt - pred
    diff_log = np.log(gt) - np.log(pred)

    abs_err  = (np.abs(diff)).mean()
    abs_rel  = (np.abs(diff) / gt).mean()
    sq_rel   = ((diff ** 2) / gt).mean()
    rmse     = np.sqrt((diff ** 2).mean())
    rmse_log = np.sqrt((diff_log ** 2).mean())
    silog    = np.sqrt((diff_log ** 2).mean() - (diff_log.mean()) ** 2) * 100
    log_10   = (np.abs(np.log10(gt) - np.log10(pred))).mean()

    result = {
        f'Abs{suffix}'    : abs_err,
        f'AbsRel{suffix}' : abs_rel,
        f'SqRel{suffix}'  : sq_rel,
        f'RMSE{suffix}'   : rmse,
        f'RMSElog{suffix}': rmse_log,
        f'SIlog{suffix}'  : silog,
        f'Log10{suffix}'  : log_10,
        f'a1{suffix}'     : a1,
        f'a2{suffix}'     : a2,
        f'a3{suffix}'     : a3,
    }
    return result

def evaluate_one_sample(pred, gt, max_depth, min_depth, cutoff_depth, clip_pred):
    list_max_depth = [max_depth] + cutoff_depth

    result = _evaluate(pred, gt, max_depth, min_depth, clip_pred, suffix='')
    for cutoff in cutoff_depth:
        result.update(_evaluate(pred, gt, cutoff, min_depth, clip_pred, suffix=f'_cutoff{int(cutoff)}'))
    return result

def eval_npy(predictions, gts, max_depth, min_depth, cutoff_depth, clip_pred):    
    eval_results = EvalResults()
    
    for i, (pred, gt) in enumerate(zip(predictions, gts)):
        result = evaluate_one_sample(pred.squeeze(), gt.squeeze(), max_depth, min_depth, cutoff_depth, clip_pred)
        eval_results.append(i, result)
        print(f'{i} / {len(gts)}')    

    return eval_results

def save_results(eval_results, dpath_out):
    output = eval_results.mean()

    print('====================')
    print(output)

    report = ''
    for k,v in output.items():
        report += f'{k}: {v}\n'

    with open(f'{dpath_out}/result.txt', 'w') as fp:
        fp.write(report)

    eval_results.to_csv(f'{dpath_out}/all_result.csv')

if __name__ == '__main__':
    # Load predictions and gts path list
    # prediction_dir = '/data_nvme/Depth-Estimation/mvsec-test/dav2-zeroshot/outdoor-day1/npy'
    # gt_dir = '/data_nvme/Depth-Estimation/MVSEC/outdoor-day/outdoor-day1/left_rect_depth'
    # json_path = '/data_nvme/Depth-Estimation/MVSEC/outdoor-day/outdoor-day1/depth2image.json'
    # prediction_files, gt_files = load_path_list(prediction_dir, gt_dir, json_path)
    
    prediction_dir = '/data_nvme/Depth-Estimation/dense-test/dav2-zeroshot/test/npy'
    split_path = '/data_nvme/Depth-Estimation/DENSE/test/dense_test.txt'
    prediction_files, gt_files = load_path_list_from_txt(split_path, prediction_dir)
    
    # Load predictions npy and gt npy
    predictions = [np.load(f) for f in prediction_files]
    gts = [np.load(f) for f in gt_files]
    print('predictions:', len(predictions), 'gts:', len(gts))
    
    # Evaluation parameters
    max_depth = 80
    min_depth = 0.1
    cutoff_depth = [10, 20, 30]
    clip_pred = True
    dpath_out = './eval_results'
    
    eval_results = eval_npy(predictions, gts, max_depth, min_depth, cutoff_depth, clip_pred)
    
    os.makedirs(dpath_out, exist_ok=True)
    save_results(eval_results, dpath_out)
