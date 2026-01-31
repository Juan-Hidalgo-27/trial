import argparse
from datetime import datetime
import os

from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
import torch
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
from xgboost import XGBClassifier

deficit_names = ['hearing', 'language', 'introspection', 'cognition', 'mood', 'memory', 'aversion', 'coordination', 'interoception', 'sleep', 'reward', 'visual recognition', 'visual perception', 'spatial reasoning', 'motor', 'somatosensory']

def command_line_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--savepath", type=str, default="", help="savepath")
    parser.add_argument("--loadpath", type=str, default="", help="loadpath")

    parser.add_argument('--k', '--k', type=int, default=[0], nargs='+', help='k-fold')
    parser.add_argument('--gene_or_receptor', '--gene_or_receptor', type=str, default=['genetics', 'receptor'], nargs='+', help='input representation')
    parser.add_argument('--lesion_or_disconnectome', '--lesion_or_disconnectome', type=str, default=['disco', 'lesion'], nargs='+', help='ground truth type')
    parser.add_argument('--lesion_deficit_thresh', '--lesion_deficit_thresh', type=float, default=[0.05, 0.05], nargs='+', help='threshold for deficit simulation')
    parser.add_argument('--deficits', '--deficits', type=int, default=list(range(1, 17)), nargs='+', help='functional parcellation deficits, 1 to 16')
    parser.add_argument('--biasdegree', '--biasdegree', type=float, default=[0, 0.25, 0.5, 0.75, 1], nargs='+', help='Degree of bias, 0 to 1')
    parser.add_argument('--biastype', '--biastype', type=str, default=['observed', 'unobserved'], nargs='+', help='bias simulation method - by location or agnostic')
    parser.add_argument('--te', '--te', type=float, default=[1, 0.75, 0.5, 0.25], nargs='+', help='treatment effect, 1 to 0')
    parser.add_argument('--re', '--re', type=float, default=[0, 0.25, 0.5, 0.75], nargs='+', help='recovery effect, 1 to 0')

    parser.add_argument('--bottlenecks', '--bottlenecks', type=int, default=[0, 2, 4, 8, 16, 32, 64, 128], nargs='+', help='autoencoder bottleneck numbers of components')
    parser.add_argument('--simpleatlases', '--simpleatlases', type=str, default=[False, "all_territories", "major_arterial_territories", "major_arterial_territories_lat", "major_territories", "clusters_lat"], nargs='+', help='simple atlases to loop through')
    parser.add_argument('--simpleatlas_argmaxs', '--simpleatlas_argmaxs', type=bool, default=[True], nargs='+', help='One-hot encode simple atlas intersection or use dice score')
    parser.add_argument('--vols', '--vols', type=bool, default=[True, False], nargs='+', help='include volume with simple atlas representations')
    parser.add_argument('--centroids', '--centroids', type=bool, default=[False], nargs='+', help='Include location of lesion centroid with simple atlas representation')

    parser.add_argument('--ml_models', '--ml_models', type=str, default=['logistic_regression', 'extra_trees', 'xgb'], nargs='+', help='sklearn-style classifiers to use')

    parser.add_argument('--use_vae', '--use_vae', type=bool, default=False, help='Use vae representation')
    parser.add_argument('--use_ae', '--use_ae', type=bool, default=False, help='Use ae representation')
    parser.add_argument('--use_nmf', '--use_nmf', type=bool, default=True, help='Use nmf representation')
    parser.add_argument('--use_pca', '--use_pca', type=bool, default=True, help='Use pca representation')

    args = parser.parse_args()
    ground_truth_params = (args.k, args.gene_or_receptor, args.lesion_or_disconnectome, args.lesion_deficit_thresh, args.deficits, args.biasdegree, args.biastype, args.te, args.re)
    representation_params = (args.bottlenecks, args.simpleatlases, args.simpleatlas_argmaxs, args.vols, args.centroids)
    selection = (args.use_vae, args.use_ae, args.use_nmf, args.use_pca)
    ml_models = args.ml_models

    ml_models_dict = {}
    if 'logistic_regression' in ml_models: ml_models_dict['logistic_regression'] = LogisticRegression()
    if 'extra_trees' in ml_models: ml_models_dict['extra_trees'] = ExtraTreesClassifier()
    if 'xgb' in ml_models: ml_models_dict['xgb'] = XGBClassifier()
    return (ground_truth_params, representation_params, selection, ml_models_dict, args.savepath, args.loadpath)


def estimate_PEHE(pred_ITE, true_ITE):
    return (((pred_ITE - true_ITE) ** 2).sum() / len(true_ITE)) ** 0.5

def process_results(pred_outcome_given_W1, pred_outcome_given_W0, true_ITE):
    pred_ITE = torch.sigmoid(pred_outcome_given_W1 - pred_outcome_given_W0)
    observed_PEHE = estimate_PEHE(pred_ITE, true_ITE).item()

    if len(true_ITE.shape) > 1:
        true_ITE = true_ITE.squeeze()
    true_ITE_xor = true_ITE[true_ITE != 0.5]
    pred_ITE_overlap = pred_ITE[true_ITE == 0.5]
    pred_ITE_xor = pred_ITE[true_ITE != 0.5]
    if len(true_ITE_xor) > 0:
        prescriptive_PEHE = estimate_PEHE(pred_ITE_xor, true_ITE_xor).item()
        presc_conf_matrix = confusion_matrix(true_ITE_xor, pred_ITE_xor > 0.5, 2)
        prescriptive_acc = (torch.sum(torch.diag(presc_conf_matrix)) / torch.sum(presc_conf_matrix)).item()
        sens, spec = sens_and_spec(presc_conf_matrix)
        prescriptive_balacc = 0.5 * (sens + spec)
        observed_tp = presc_conf_matrix[1, 1] + (pred_ITE_overlap > 0.5).sum()
        observed_fp, observed_fn = presc_conf_matrix[1, 0], presc_conf_matrix[0, 1]
        observed_tn = presc_conf_matrix[0, 0] + (pred_ITE_overlap < 0.5).sum()
    else:
        prescriptive_PEHE, prescriptive_acc, prescriptive_balacc = torch.nan, torch.nan, torch.nan
        presc_conf_matrix = torch.Tensor([[torch.nan, torch.nan], [torch.nan, torch.nan]])
        observed_tp = (pred_ITE_overlap > 0.5).sum()
        observed_fp, observed_fn = 0, 0
        observed_tn = (pred_ITE_overlap < 0.5).sum()
    observed_conf_matrix = torch.Tensor([[observed_tn, observed_fp], [observed_fn, observed_tp]])
    observed_acc = (torch.sum(torch.diag(observed_conf_matrix)) / torch.sum(observed_conf_matrix)).item()
    sens, spec = sens_and_spec(observed_conf_matrix)
    observed_balacc = 0.5 * (sens + spec)

    return (observed_PEHE, observed_acc, observed_balacc, observed_conf_matrix.cpu().detach().numpy().astype(int)), \
           (prescriptive_PEHE, prescriptive_acc, prescriptive_balacc, presc_conf_matrix.cpu().detach().numpy().astype(int))


class ground_truth_simulation:
    def __init__(self, LESION_OR_DISCONNECTOME, TE, RE, BIAS, BIASTYPE, K):
        self.LESION_OR_DISCONNECTOME = LESION_OR_DISCONNECTOME
        self.TE = TE
        self.RE = RE
        self.BIAS = BIAS
        self.seed = K * 10000
        self.BIASTYPE = BIASTYPE
        self.deficit_names = {1:'hearing',
                              2:'language',
                              3:'introspection',
                              4:'cognition',
                              5:'mood',
                              6:'memory',
                              7:'aversion',
                              8:'coordination',
                              9:'interoception',
                              10:'sleep',
                              11:'reward',
                              12:'visual recognition',
                              13:'visual perception',
                              14:'spatial reasoning',
                              15:'motor',
                              16:'somatosensory'}


    def simulate_bias(self, train, biastype):

        feature = ''
        if biastype == 'x':
            feature = 'centroid_x'
        elif biastype == 'y':
            feature = 'centroid_y'
        elif biastype == 'z':
            feature = 'centroid_z'

        # Group into bins by bias parameter
        train = train.sort_values(by=f"{self.LESION_OR_DISCONNECTOME}_{feature}")
        inds = train.index

        if self.BIAS == 0:
            np.random.seed(self.seed)
            self.seed += 1
            selection = np.random.choice([0, 1], len(train))
            allocation = [inds[i] for i in range(len(train)) if selection[i] == 1]

        elif self.BIAS <= 0.5:
            probs = np.linspace(0.5 - self.BIAS, 0.5 + self.BIAS, len(train))
            allocation = []
            for i in range(len(train)):
                np.random.seed(self.seed)
                self.seed += 1
                if np.random.choice([0, 1], p=[1 - probs[i], probs[i]]) == 1:
                    allocation.append(inds[i])
        else:
            allocation = []
            median = int(np.round(len(train) / 2))
            lowhalf = train.iloc[:median]
            uphalf = train.iloc[median:]

            proportion_fix = 2 * self.BIAS - 1
            allocation.append(list(uphalf.iloc[int(np.round(len(uphalf) * (1 - proportion_fix))):].index))
            uphalf_iter = uphalf.iloc[:int(np.round(len(uphalf) * (1 - proportion_fix)))]
            inds = uphalf_iter.index
            uphalf_iter_probs = np.linspace(0.5, 1, len(uphalf_iter))
            for i in range(len(uphalf_iter)):
                np.random.seed(self.seed)
                self.seed += 1
                if np.random.choice([0, 1], p=[1 - uphalf_iter_probs[i], uphalf_iter_probs[i]]) == 1:
                    allocation.append([inds[i]])

            if proportion_fix == 1:
                lowhalf_iter = lowhalf.iloc[-1:]
            else:
                lowhalf_iter = lowhalf.iloc[-int(np.round(len(lowhalf) * (1 - proportion_fix))):]
            inds = lowhalf_iter.index
            lowhalf_iter_probs = np.linspace(0, 0.5, len(lowhalf_iter))
            for i in range(len(lowhalf_iter_probs)):
                np.random.seed(self.seed)
                self.seed += 1
                if np.random.choice([0, 1], p=[1 - lowhalf_iter_probs[i], lowhalf_iter_probs[i]]) == 1:
                    allocation.append([inds[i]])

            allocation = [item for sublist in allocation for item in sublist]

        return allocation




    def simulate_agnostic_bias(self, train):

        # Group into bins by bias parameter
        inds = train.index
        bias_towards = train['y_true'].to_numpy()

        if self.BIAS == 0:
            np.random.seed(self.seed)
            self.seed += 1
            selection = np.random.choice([0, 1], len(train))
            return [inds[i] for i in range(len(train)) if selection[i] == 1]
        else:
            allocation = []
            for i in range(len(train)):
                if bias_towards[i] == 1:
                    prob_1 = self.BIAS
                elif bias_towards[i] == 0:
                    prob_1 = 1 - self.BIAS
                else:
                    prob_1 = 0.5
                np.random.seed(self.seed)
                self.seed += 1
                if np.random.choice([0, 1], 1, p=[1 - prob_1, prob_1]):
                    allocation.append(inds[i])
            return allocation


    def simulate_trial(self, train, roi_centroids):

        train.reset_index(inplace=True, drop=True)

        if self.BIASTYPE == 'unobserved':
            train['group'] = 0
            train.loc[self.simulate_agnostic_bias(train), 'group'] = 1
        else:
            roi_centroids_x = roi_centroids[0]
            roi_centroids_y = roi_centroids[1]
            roi_centroids_z = roi_centroids[2]
            xyz = [abs(roi_centroids_x[0] - roi_centroids_x[1]),
                   abs(roi_centroids_y[0] - roi_centroids_y[1]),
                   abs(roi_centroids_z[0] - roi_centroids_z[1])]

            if np.argmax(xyz) == 0:
                allocation = self.simulate_bias(train, 'x')
                train['group'] = 1 - np.argmax(roi_centroids_x)
                train.loc[allocation, 'group'] = np.argmax(roi_centroids_x)

            elif np.argmax(xyz) == 1:
                allocation = self.simulate_bias(train, 'y')
                train['group'] = 1 - np.argmax(roi_centroids_y)
                train.loc[allocation, 'group'] = np.argmax(roi_centroids_y)

            else:
                allocation = self.simulate_bias(train, 'z')
                train['group'] = 1 - np.argmax(roi_centroids_z)
                train.loc[allocation, 'group'] = np.argmax(roi_centroids_z)

        # Select group who received the treatment for which they are
        # truly susceptible to apply treatment effects
        susceptible = train.loc[(train['group'] == train['y_true']) | (train['y_true'] == 0.5)]
        np.random.seed(self.seed)
        self.seed += 1
        respond = np.random.choice(susceptible.index,
                                   int(np.round(len(susceptible) * self.TE)),
                                   replace=False)

        # Apply treatment effect by overwriting the respond column
        # to those in the proportion of those correctly treated selected
        # by the random treatment effect.
        train['respond'] = 0
        train.loc[respond, 'respond'] = 1

        # Apply the recovery effect to the whole cohort
        if self.RE != 0:
            np.random.seed(self.seed)
            self.seed += 1
            spont_resp = np.random.choice(train.index,
                                          int(round(len(train) * self.RE)),
                                          replace=False)
            train.loc[spont_resp, 'respond'] = 1

        W_train = np.array(train['group'])
        Y_train = np.array(train['respond'])

        return W_train, Y_train


class genGT:
    def __init__(self, loadpath, savepath, k, gene_or_receptor, lesion_or_disconnectome, lesion_deficit_thresh,
                 deficits, biasdegree, biastype, te, re, params,
                 simpleatlases, simpleatlas_argmaxs, vols, centroids, bottlenecks):
        self.loadpath = loadpath
        self.savepath = savepath
        self.k = k
        self.gene_or_receptor = gene_or_receptor
        self.lesion_or_disconnectome = lesion_or_disconnectome
        self.deficits = deficits
        self.biasdegree = biasdegree
        self.te = te
        self.re = re
        self.params = params
        self.simpleatlases = simpleatlases
        self.simpleatlas_argmaxs = simpleatlas_argmaxs
        self.vols = vols
        self.centroids = centroids
        self.bottlenecks = bottlenecks
        self.biastype = biastype
        self.lesion_deficit_thresh = lesion_deficit_thresh
        self.deficit_names = {1:'hearing',
                              2:'language',
                              3:'introspection',
                              4:'cognition',
                              5:'mood',
                              6:'memory',
                              7:'aversion',
                              8:'coordination',
                              9:'interoception',
                              10:'sleep',
                              11:'reward',
                              12:'visual recognition',
                              13:'visual perception',
                              14:'spatial reasoning',
                              15:'motor',
                              16:'somatosensory'}


    def generate_iteration(self):
        if self.lesion_or_disconnectome.startswith('lesion'):
            filename = f'lesion_{self.lesion_deficit_thresh}'
            input_type = 'lesion'
        elif self.lesion_or_disconnectome.startswith('disco'):
            filename = f'disco_{self.lesion_deficit_thresh}'
            input_type = 'disco'
        else:
            raise ValueError('Lesion or disconnectome index not recognised.')

        if self.gene_or_receptor.startswith('receptor'):
            filename += f'_receptor_{self.k}'
        elif self.gene_or_receptor.startswith("gene"):
            filename +=  f'_genetics_{self.k}'
        else:
            raise ValueError('Gene or receptor index not recognised.')
        loadpath = os.path.join(self.loadpath, filename)

        train = pd.read_pickle(os.path.join(loadpath, f"train_{input_type}_{self.lesion_deficit_thresh}_{self.k}.pkl"))
        test = pd.read_pickle(os.path.join(loadpath, f"test_{input_type}_{self.lesion_deficit_thresh}_{self.k}.pkl"))

        roi_centroids = np.array(pd.read_json(os.path.join(loadpath, "centroids.json"))[self.deficits].tolist())

        train_slice = train.loc[(train[f"{self.deficit_names[self.deficits]}_W0"] == 1) | (train[f"{self.deficit_names[self.deficits]}_W1"] == 1)].reset_index()
        if len(train_slice) > 0:
            train_slice.loc[train_slice[f"{self.deficit_names[self.deficits]}_W1"] == 1, 'y_true'] = 1
            train_slice.loc[train_slice[f"{self.deficit_names[self.deficits]}_W0"] == 1, 'y_true'] = 0
            train_slice.loc[(train_slice[f"{self.deficit_names[self.deficits]}_W0"] == 1) & (train_slice[f"{self.deficit_names[self.deficits]}_W1"] == 1), 'y_true'] = 0.5
        else:
            train_slice = None

        test_slice = test.loc[(test[f"{self.deficit_names[self.deficits]}_W0"] == 1) | (test[f"{self.deficit_names[self.deficits]}_W1"] == 1)].reset_index()
        if len(test_slice) > 0:
            test_slice.loc[test_slice[f"{self.deficit_names[self.deficits]}_W1"] == 1, 'y_true'] = 1
            test_slice.loc[test_slice[f"{self.deficit_names[self.deficits]}_W0"] == 1, 'y_true'] = 0
            test_slice.loc[(test_slice[f"{self.deficit_names[self.deficits]}_W0"] == 1) & (test_slice[f"{self.deficit_names[self.deficits]}_W1"] == 1), 'y_true'] = 0.5
        else:
            test_slice = None
        return train_slice, test_slice, roi_centroids



    def prepare_representations(self, train_df, test_df, ae=True, vae=True, nmf=True, pca=True):
        X_train, X_test = {}, {}
        if self.bottlenecks > 0:
            if ae:
                X_train['ae'] = np.stack(train_df[
                                             f"ae_{self.lesion_or_disconnectome}_{int(self.bottlenecks)}_K{self.k}"]).astype(
                    float).reshape(len(train_df), self.bottlenecks)
                X_test['ae'] = np.stack(test_df[
                                            f"ae_{self.lesion_or_disconnectome}_{int(self.bottlenecks)}_K{self.k}"]).astype(
                    float).reshape(len(test_df), self.bottlenecks)

            if vae:
                X_train['vae'] = np.stack(train_df[
                                              f"vae_means_{self.lesion_or_disconnectome}_{int(self.bottlenecks)}_K{self.k}"]).astype(
                    float).reshape(len(train_df), self.bottlenecks)
                X_test['vae'] = np.stack(test_df[
                                             f"vae_means_{self.lesion_or_disconnectome}_{int(self.bottlenecks)}_K{self.k}"]).astype(
                    float).reshape(len(test_df), self.bottlenecks)

            if nmf:
                X_train['nmf'] = np.stack(train_df[
                                              f"nmf_{self.lesion_or_disconnectome}_{int(self.bottlenecks)}_K{self.k}"]).astype(
                    float).reshape(len(train_df), self.bottlenecks)
                X_test['nmf'] = np.stack(test_df[
                                             f"nmf_{self.lesion_or_disconnectome}_{int(self.bottlenecks)}_K{self.k}"]).astype(
                    float).reshape(len(test_df), self.bottlenecks)

            if pca:
                X_train['pca'] = np.stack(train_df[
                                              f"pca_{self.lesion_or_disconnectome}_{int(self.bottlenecks)}_K{self.k}"]).astype(
                    float).reshape(len(train_df), self.bottlenecks)
                X_test['pca'] = np.stack(test_df[
                                             f"pca_{self.lesion_or_disconnectome}_{int(self.bottlenecks)}_K{self.k}"]).astype(
                    float).reshape(len(test_df), self.bottlenecks)
        else:
            simpleatlas_done = False
            if self.simpleatlases:
                X_train_simpleatlas = np.stack(train_df[f"{self.lesion_or_disconnectome}_{self.simpleatlases}"].to_numpy())
                X_test_simpleatlas = np.stack(test_df[f"{self.lesion_or_disconnectome}_{self.simpleatlases}"].to_numpy())
                simpleatlas_done = True

                if self.simpleatlas_argmaxs:
                    group_integers = np.argmax(X_train_simpleatlas, axis=1)
                    X_train_simpleatlas = np.zeros_like(X_train_simpleatlas).astype(
                        bool)
                    for i, group in enumerate(group_integers):
                        X_train_simpleatlas[i, group] = True

                    group_integers = np.argmax(X_test_simpleatlas, axis=1)
                    X_test_simpleatlas = np.zeros_like(X_test_simpleatlas).astype(bool)
                    for i, group in enumerate(group_integers):
                        X_test_simpleatlas[i, group] = True

            vols_done = False
            if self.vols:
                X_train_vols = train_df[
                    f"{self.lesion_or_disconnectome}_vol"].to_numpy()
                scaler = StandardScaler().fit(X_train_vols.reshape(-1, 1))
                X_train_vols = scaler.transform(X_train_vols.reshape(-1, 1))
                X_test_vols = scaler.transform(np.stack(
                    test_df[f"{self.lesion_or_disconnectome}_vol"].to_numpy()).reshape(
                    -1, 1))
                vols_done = True

            centroids_done = False
            if self.centroids:
                X_train_centroid_x = train_df[
                    f"{self.lesion_or_disconnectome}_centroid_x"].to_numpy()
                X_train_centroid_y = train_df[
                    f"{self.lesion_or_disconnectome}_centroid_y"].to_numpy()
                X_train_centroid_z = train_df[
                    f"{self.lesion_or_disconnectome}_centroid_z"].to_numpy()
                X_train_centroids = np.vstack(
                    [X_train_centroid_x, X_train_centroid_y,
                     X_train_centroid_z]).T
                centroid_scaler = StandardScaler().fit(
                    X_train_centroids)
                X_train_centroids = centroid_scaler.transform(
                    X_train_centroids)

                X_test_centroid_x = test_df[
                    f"{self.lesion_or_disconnectome}_centroid_x"].to_numpy()
                X_test_centroid_y = test_df[
                    f"{self.lesion_or_disconnectome}_centroid_y"].to_numpy()
                X_test_centroid_z = test_df[
                    f"{self.lesion_or_disconnectome}_centroid_z"].to_numpy()
                X_test_centroids = np.vstack(
                    [X_test_centroid_x, X_test_centroid_y,
                     X_test_centroid_z]).T
                X_test_centroids = centroid_scaler.transform(
                    X_test_centroids)
                centroids_done = True

            X_train_iter = np.zeros((len(train_df), 1))
            if simpleatlas_done: X_train_iter = np.hstack([X_train_iter, X_train_simpleatlas])
            if vols_done: X_train_iter = np.hstack([X_train_iter, X_train_vols])
            if centroids_done: X_train_iter = np.hstack([X_train_iter, X_train_centroids])
            X_train_iter = X_train_iter[:, 1:] if X_train_iter.shape[1] > 1 else X_train_iter
            X_train[(self.simpleatlases, self.simpleatlas_argmaxs, self.vols, self.centroids)] = X_train_iter

            X_test_iter = np.zeros((len(test_df), 1))
            if simpleatlas_done: X_test_iter = np.hstack([X_test_iter, X_test_simpleatlas])
            if vols_done: X_test_iter = np.hstack([X_test_iter, X_test_vols])
            if centroids_done: X_test_iter = np.hstack([X_test_iter, X_test_centroids])
            X_test_iter = X_test_iter[:, 1:] if X_test_iter.shape[1] > 1 else X_test_iter
            X_test[(self.simpleatlases, self.simpleatlas_argmaxs, self.vols, self.centroids)] = X_test_iter

        return X_train, X_test

    def simulate_observational_data(self, train_data, test_data, roi_centroids):
        for BIASTYPE in self.biastype:
            for TE, RE in zip(self.te, self.re):
                for BIAS in self.biasdegree:
                    xval_dict = {**{'K': self.k,
                                    'GENE_OR_RECEPTOR': self.gene_or_receptor,
                                    'LESION_OR_DISCONNECTOME': self.lesion_or_disconnectome,
                                    'DEFICIT': self.deficits,
                                    'DEFICIT_CLASS': self.deficit_names[self.deficits],
                                    'BIASTYPE': BIASTYPE,
                                    'TE': TE,
                                    'RE': RE,
                                    'BIAS': BIAS,
                                    'bottleneck': self.bottlenecks,
                                    'simpleatlas': self.simpleatlases,
                                    'simpleatlas_argmax': self.simpleatlas_argmaxs,
                                    'vols': self.vols,
                                    'centroids': self.centroids}, **self.params}

                    gt_sim = ground_truth_simulation(self.lesion_or_disconnectome, TE, RE, BIAS,
                                                     BIASTYPE, self.k)
                    w_train, y_train = gt_sim.simulate_trial(train_data, roi_centroids)

                    y_test = np.array(test_data['y_true'])
                    yield ((w_train, y_train), y_test), xval_dict



def process_onemodel(train_embeddings, W_train, Y_train, test_embeddings, true_ITE, ml_models, xval_dict):
    results_onemodel = pd.DataFrame()
    train_embeddings = torch.hstack([train_embeddings.cpu(), W_train]).detach().numpy()
    test_embeddings = test_embeddings.cpu().detach().numpy()
    test_embeddings_W1, test_embeddings_W0 = np.hstack([test_embeddings, np.ones([test_embeddings.shape[0], 1])]), np.hstack([test_embeddings, np.zeros([test_embeddings.shape[0], 1])])
    Y_train = Y_train.cpu().detach().numpy().ravel()
    true_ITE = true_ITE.cpu()

    for model_name in ml_models.keys():
        time = datetime.now()

        model = ml_models[model_name]
        if len(set(Y_train)) > 1:
            model.fit(train_embeddings, Y_train)
            pred_outcome_given_W1 = model.predict_proba(test_embeddings_W1)[:, 1]
            pred_outcome_given_W0 = model.predict_proba(test_embeddings_W0)[:, 1]
        else:
            pred_outcome_given_W1 = np.zeros(test_embeddings_W1.shape[0])
            pred_outcome_given_W1[:] = Y_train[0]
            pred_outcome_given_W0 = np.zeros(test_embeddings_W0.shape[0])
            pred_outcome_given_W0[:] = Y_train[0]

        observed_results, prescriptive_results = process_results(torch.Tensor(pred_outcome_given_W1).view(-1, 1),
                                                                 torch.Tensor(pred_outcome_given_W0).view(-1, 1),
                                                                 true_ITE)
        time_taken = datetime.now() - time
        onemodel_results = {**xval_dict,
                            'model': model_name,
                            'classifier_type': 'one',
                            'PEHE': observed_results[0],
                            'observed_acc': observed_results[1],
                            'observed_balacc': observed_results[2],
                            'observed_confusion_matrix': [list(observed_results[3].flatten())],
                            'PEHE_xor': prescriptive_results[0],
                            'prescriptive_acc': prescriptive_results[1],
                            'prescriptive_balacc': prescriptive_results[2],
                            'prescriptive_confusion_matrix': [list(prescriptive_results[3].flatten())],
                            'time_taken': time_taken}

        results_onemodel = pd.concat([results_onemodel, pd.DataFrame(onemodel_results, index=[0])])
    return results_onemodel

def process_twomodels(train_embeddings, W_train, Y_train, test_embeddings, true_ITE, ml_models, xval_dict):
    results_twomodels = pd.DataFrame()
    W_train = W_train.cpu().detach().numpy().ravel()
    Y_train = Y_train.cpu().detach().numpy().ravel()
    train_embeddings = train_embeddings.cpu().detach().numpy()
    test_embeddings = test_embeddings.cpu().detach().numpy()
    X_given_W1_observed, Y_given_W1_observed = train_embeddings[W_train == 1, :], Y_train[W_train == 1]
    X_given_W0_observed, Y_given_W0_observed = train_embeddings[W_train == 0, :], Y_train[W_train == 0]
    true_ITE = true_ITE.cpu()

    for model_name in ml_models.keys():
        time = datetime.now()

        if len(set(Y_given_W1_observed)) == 1:
            pred_outcome_given_W1 = np.zeros(test_embeddings.shape[0])
            pred_outcome_given_W1[:] = Y_given_W1_observed[0]
        else:
            model_W1 = ml_models[model_name]
            model_W1.fit(X_given_W1_observed, Y_given_W1_observed)
            pred_outcome_given_W1 = model_W1.predict_proba(test_embeddings)[:, 1]

        if len(set(Y_given_W0_observed)) == 1:
            pred_outcome_given_W0 = np.zeros(test_embeddings.shape[0])
            pred_outcome_given_W0[:] = Y_given_W0_observed[0]
        else:
            model_W0 = ml_models[model_name]
            model_W0.fit(X_given_W0_observed, Y_given_W0_observed)
            pred_outcome_given_W0 = model_W0.predict_proba(test_embeddings)[:, 1]

        observed_results, prescriptive_results = process_results(torch.Tensor(pred_outcome_given_W1).view(-1, 1),
                                                         torch.Tensor(pred_outcome_given_W0).view(-1, 1),
                                                         true_ITE)
        time_taken = datetime.now() - time
        twomodel_results = {**xval_dict,
                            'model': model_name,
                            'classifier_type': 'two',
                            'PEHE': observed_results[0],
                            'observed_acc': observed_results[1],
                            'observed_balacc': observed_results[2],
                            'observed_confusion_matrix': [list(observed_results[3].flatten())],
                            'PEHE_xor': prescriptive_results[0],
                            'prescriptive_acc': prescriptive_results[1],
                            'prescriptive_balacc': prescriptive_results[2],
                            'prescriptive_confusion_matrix': [list(prescriptive_results[3].flatten())],
                            'time_taken': time_taken}
        results_twomodels = pd.concat([results_twomodels, pd.DataFrame(twomodel_results, index=[0])])
    return results_twomodels


def confusion_matrix(true_labels, predictions, num_classes, torch_input=True):
    '''
    Compute the confusion matrix
    :param true_labels:
    :param predictions:
    :return:
    '''

    if torch_input:
        device = true_labels.device

        # int, slice or bool needed for slice indexing
        if not true_labels.type() == 'torch.LongTensor':
            true_labels = true_labels.type(torch.LongTensor)
        if not predictions.type() == 'torch.LongTensor':
            predictions = predictions.type(torch.LongTensor)

        con_mat = torch.zeros((num_classes, num_classes), device=device)
        N = true_labels.shape[0]

    else:
        con_mat = np.zeros((num_classes, num_classes))
        N = len(true_labels)

    # Loop over the vector of true_labels, each element of which selects the row in con_mat, and increment the column
    # defined by the corresponding element in predictions
    for k in range(N):
        con_mat[true_labels[k], predictions[k]] += 1

    return con_mat

def sens_and_spec(con_mat, return_numpy=True):
    """
    con_mat[x, y] is the tally of examples in class x that were predicted in y
    """
    tp = con_mat[1, 1]
    tn = con_mat[0, 0]
    fn = con_mat[1, 0]
    fp = con_mat[0, 1]

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    if return_numpy and not isinstance(con_mat, np.ndarray):
        sensitivity = sensitivity.cpu().detach().numpy()
        specificity = specificity.cpu().detach().numpy()

    return sensitivity, specificity


def prepare_array_mapping(k,
                          gene_or_receptor,
                          lesion_or_disconnectome,
                          lesion_deficit_thresh,
                          deficits,
                          bottlenecks,
                          simpleatlases,
                          simpleatlas_argmaxs,
                          vols,
                          centroids,
                          job_batch=10):
    params = []
    for K in range(k):
        for GENE_OR_RECEPTOR in gene_or_receptor:
            for LESION_OR_DISCONNECTOME, LESION_DEFICIT_THRESH in zip(lesion_or_disconnectome, lesion_deficit_thresh):
                for BOTTLENECK in bottlenecks:
                    for DEFICIT in deficits:
                        if BOTTLENECK == 0:
                            for SIMPLEATLAS in simpleatlases:
                                for SIMPLEATLAS_ARGMAX in simpleatlas_argmaxs:
                                    for VOL in vols:
                                        for CENTROID in centroids:
                                            params.append([K, GENE_OR_RECEPTOR, LESION_OR_DISCONNECTOME, LESION_DEFICIT_THRESH, DEFICIT, BOTTLENECK, SIMPLEATLAS, SIMPLEATLAS_ARGMAX, VOL, CENTROID])
                        else:
                            params.append([K, GENE_OR_RECEPTOR, LESION_OR_DISCONNECTOME,
                                           LESION_DEFICIT_THRESH, DEFICIT, BOTTLENECK, False, False, False, False])

    if job_batch:
        batches, n = [], 0
        n_batches = int(np.ceil(len(params) / job_batch))
        for batch in range(n_batches):
            batches.append(params[n:n + job_batch])
            n += job_batch
    else:
        batches = np.array(params)
    return batches


class prescription_processor:
    def __init__(self, params):
        ground_truth_params, representation_params, selection, ml_models, savepath, loadpath = params
        self.k, self.gene_or_receptor, self.lesion_or_disconnectome, self.lesion_deficit_thresh, self.deficits, self.biasdegree, self.biastype, self.te, self.re = ground_truth_params
        self.bottlenecks, self.simpleatlases, self.simpleatlas_argmaxs, self.vols, self.centroids = representation_params
        self.run_vae, self.run_ae, self.run_nmf, self.run_pca = selection
        self.ml_models = ml_models
        self.mni_dim = [91, 109, 91]
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.savepath = savepath
        self.loadpath = loadpath
        self.min_n = 15

    def run_iteration(self, train_data, test_data, xval_dict, X_train, X_test):
        results_df = pd.DataFrame()

        bottleneck = xval_dict['bottleneck']
        w_train, y_train = train_data
        if len(y_train) > self.min_n - 1:
            w_train, y_train = torch.from_numpy(w_train).view(-1, 1).float(), torch.from_numpy(y_train).view(-1, 1).float()
            true_ITE = torch.Tensor(test_data).view(-1, 1)
            if bottleneck > 0:
                for rep in X_train.keys():
                    results_onemodel = process_onemodel(X_train[rep], w_train, y_train, X_test[rep], true_ITE,
                                                        self.ml_models,
                                                        {**xval_dict, "representation": f"{bottleneck}D_{rep}"})
                    results_df = pd.concat([results_df, results_onemodel])
                    results_twomodels = process_twomodels(X_train[rep], w_train, y_train, X_test[rep], true_ITE,
                                                          self.ml_models,
                                                          {**xval_dict, "representation": f"{bottleneck}D_{rep}"})
                    results_df = pd.concat([results_df, results_twomodels])

            else:
                for params in X_train.keys():
                    X_train_iter = X_train[params]
                    X_test_iter = X_test[params]

                    results_onemodel = process_onemodel(X_train_iter, w_train, y_train, X_test_iter, true_ITE,
                                                        self.ml_models, {**xval_dict, "representation": "no_img"})
                    results_df = pd.concat([results_df, results_onemodel])
                    results_twomodels = process_twomodels(X_train_iter, w_train, y_train, X_test_iter, true_ITE,
                                                          self.ml_models, {**xval_dict, "representation": "no_img"})
                    results_df = pd.concat([results_df, results_twomodels])
        return results_df



    def run_all(self, iter_params, savepath):
        savepath_grid = os.path.join(savepath, f"prescription")
        if not os.path.exists(savepath_grid): os.makedirs(savepath_grid)

        iter_params = pd.DataFrame(iter_params, columns=['k', 'gene_or_receptor',
                                                         'lesion_or_disconnectome', 'lesion_deficit_thresh',
                                                         'deficits', 'bottlenecks', 'simpleatlases',
                                                         'simpleatlas_argmaxs', 'vols', 'centroids'])
        k = iter_params['k'].to_numpy().astype(int)
        gene_or_receptor = iter_params['gene_or_receptor'].to_numpy()
        lesion_or_disconnectome = iter_params['lesion_or_disconnectome'].to_numpy()
        lesion_deficit_thresh = iter_params['lesion_deficit_thresh'].to_numpy().astype(float)
        deficits = iter_params['deficits'].to_numpy().astype(int)
        bottlenecks = iter_params['bottlenecks'].to_numpy().astype(int)
        simpleatlases = iter_params['simpleatlases'].to_numpy()
        simpleatlas_argmaxs = iter_params['simpleatlas_argmaxs'].to_numpy()
        vols = iter_params['vols'].to_numpy()
        centroids = iter_params['centroids'].to_numpy()

        simpleatlases = [False if x == "False" else x for x in simpleatlases]
        simpleatlas_argmaxs = [True if x == "True" else False for x in simpleatlas_argmaxs]
        vols = [True if x == "True" else False for x in vols]
        centroids = [True if x == "True" else False for x in centroids]

        total_sims = len(k)

        if 0 not in bottlenecks:
            simpleatlases, simpleatlas_argmaxs = [False] * total_sims, [False] * total_sims
            vols, centroids = [False] * total_sims, [False] * total_sims

        all_conditions = tqdm(zip(k, gene_or_receptor, lesion_or_disconnectome, lesion_deficit_thresh, deficits, bottlenecks,
                                  simpleatlases, simpleatlas_argmaxs, vols, centroids), total=total_sims)
        conditions_iter = 0
        for K, GENE_OR_RECEPTOR, LESION_OR_DISCONNECTOME, LESION_DEFICIT_THRESH, DEFICIT, BOTTLENECK, SIMPLEATLAS, SIMPLEATLAS_ARGMAX, VOL, CENTROID in all_conditions:

            prepare_generator = genGT(self.loadpath, savepath_grid, K, GENE_OR_RECEPTOR, LESION_OR_DISCONNECTOME,
                                      LESION_DEFICIT_THRESH,
                                      DEFICIT, self.biasdegree, self.biastype, self.te, self.re, {}, SIMPLEATLAS,
                                      SIMPLEATLAS_ARGMAX,
                                      VOL, CENTROID, BOTTLENECK)
            train_df, test_df, roi_centroids = prepare_generator.generate_iteration()
            if train_df is not None and test_df is not None:
                X_train, X_test = prepare_generator.prepare_representations(train_df, test_df, ae=self.run_ae, vae=self.run_vae,
                                                                            nmf=self.run_nmf, pca=self.run_pca)
                for representation in X_train.keys():
                    X_train[representation] = torch.from_numpy(X_train[representation]).float()
                    X_test[representation] = torch.from_numpy(X_test[representation]).float()
                for (train_data, test_data), xval_dict in prepare_generator.simulate_observational_data(train_df, test_df,
                                                                                                      roi_centroids):
                    results_df = self.run_iteration(train_data, test_data, xval_dict, X_train, X_test)
                    results_df.to_pickle(os.path.join(savepath_grid,
                                                      f"prescriptive_results_core_iter{conditions_iter}.pkl"))
                    conditions_iter += 1


    def run(self):
        iter_params = prepare_array_mapping(self.k,
                                            self.gene_or_receptor,
                                            self.lesion_or_disconnectome,
                                            self.lesion_deficit_thresh,
                                            self.deficits,
                                            self.bottlenecks,
                                            self.simpleatlases,
                                            self.simpleatlas_argmaxs,
                                            self.vols,
                                            self.centroids, job_batch=0)

        self.run_all(iter_params, self.savepath)


if __name__ == "__main__":
    parameters = command_line_options()
    presc = prescription_processor(parameters)
    presc.run()