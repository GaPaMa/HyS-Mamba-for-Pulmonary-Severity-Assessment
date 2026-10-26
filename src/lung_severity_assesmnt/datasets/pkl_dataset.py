import os
import re
import glob
import cv2
import numpy as np
import torch
import pandas as pd
import random
from torch.utils.data import Dataset
from typing import Optional, Dict, Any, List, Union, Tuple

from lung_severity_assesmnt.registry import DATASETS
from lung_severity_assesmnt.utils.data import get_transform


def _natural_key(s: str):
    # "10.jpg" > "2.jpg" correctly
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def _read_image(path: str, grayscale_to_rgb: bool = True) -> np.ndarray:
    buf = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    # 16-bit -> 8-bit (safe normalization)
    if img.dtype == np.uint16:
        mx = float(img.max())
        img = (img.astype(np.float32) / mx * 255.0).astype(np.uint8) if mx > 0 else (img / 256).astype(np.uint8)

    # grayscale -> 3ch
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1) if grayscale_to_rgb else img[..., None]

    # BGRA -> BGR
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]

    # BGR -> RGB
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def _strip_and_drop_unnamed(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip() for c in df.columns]
    keep = [c for c in df.columns if not str(c).lower().startswith("unnamed")]
    return df[keep]


def _resolve_path(ds_root: str, img_dir: Optional[str], raw: str) -> str:
    p = str(raw).strip()
    if os.path.isabs(p) and os.path.exists(p):
        return p

    candidates = []
    candidates.append(os.path.join(ds_root, p))
    if img_dir is not None:
        candidates.append(os.path.join(img_dir, p))

    for c in candidates:
        if os.path.exists(c):
            return c

    # try extension swap even if extension exists (common issue: csv says .jpg but files are .png)
    base, _ext = os.path.splitext(os.path.basename(p))
    for root in [img_dir, ds_root]:
        if not root:
            continue
        for e in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
            c = os.path.join(root, base + e)
            if os.path.exists(c):
                return c

    # last resort: recursive by basename
    bn = os.path.basename(p)
    hits = glob.glob(os.path.join(ds_root, "**", bn), recursive=True)
    if img_dir is not None:
        hits += glob.glob(os.path.join(img_dir, "**", bn), recursive=True)
    hits = [h for h in hits if os.path.isfile(h)]
    if hits:
        return hits[0]

    raise FileNotFoundError(f"Image not found: {raw} | ds_root={ds_root} | img_dir={img_dir}")


def _auto_img_dir(ds_root: str, subset: str) -> str:
    # CT_Dataset: Train/Val/Test folders
    cand = os.path.join(ds_root, subset)
    if os.path.isdir(cand):
        return cand

    # RALO_Aug: Lung_Rep1
    cand = os.path.join(ds_root, "Lung_Rep1")
    if os.path.isdir(cand):
        return cand

    return ds_root


def _list_images(img_dir: str, sort_mode: str = "natural") -> List[str]:
    files = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"):
        files += glob.glob(os.path.join(img_dir, ext))
    if sort_mode == "natural":
        files.sort(key=lambda p: _natural_key(os.path.basename(p)))
    else:
        files.sort()
    return files


@DATASETS.register("pkl_dataset")
class PKLDataset(Dataset):
    """
    Supports:
      - Original <dataset>.pkl (Fold/Image/Age) [unchanged]
      - CSV datasets (header OR headerless)

    Key args for your medical sets:
      ann_file: path to CSV
      csv_has_header: False for CT_Dataset
      csv_columns: names when no header (e.g. ["image","cip","flag"])
      img_dir: folder containing images (RALO: .../RALO_Aug/Lung_Rep1)
      img_from_dir: True when CSV has labels only (like lung_rep1.csv)
      img_sort: "natural" (recommended) or "lex"
      img_col: column name or index for image filenames (when present)
      target_cols: list of columns (names or indices)
      grayscale_to_rgb: keep True for ImageNet pretrained backbones

    region_replace:
      leave for faces only; do not use for medical.
    """

    def __init__(
        self,
        path: str,
        dataset: str,
        subset: str,
        transform: list,
        region_replace: Optional[Dict[str, Any]] = None,
        ann_file: Optional[str] = None,
        csv_has_header: bool = True,
        csv_columns: Optional[List[str]] = None,
        img_dir: Optional[str] = None,
        img_from_dir: bool = False,
        img_sort: str = "natural",
        img_col: Optional[Union[str, int]] = None,
        target_cols: Optional[List[Union[str, int]]] = None,
        grayscale_to_rgb: bool = True,
    ):
        self.path = path
        self.dataset = dataset
        self.subset = subset
        self.transform = get_transform(transform)
        self.grayscale_to_rgb = grayscale_to_rgb

        ds_root = os.path.join(path, dataset)
        pkl_path = os.path.join(path, dataset + ".pkl")

        self.images: List[str] = []
        self.labels: Union[List[float], np.ndarray] = []

        # -------------------------
        # PKL mode (MORPH etc.)
        # -------------------------
        if os.path.isfile(pkl_path):
            df = pd.read_pickle(pkl_path)
            mask = df["Fold"].astype(str).str.lower() == str(subset).lower()
            df = df.loc[mask]
            self.images = [os.path.join(path, dataset, fname) for fname in df["Image"].tolist()]
            self.labels = [float(a) for a in df["Age"].tolist()]

        # -------------------------
        # CSV mode (medical)
        # -------------------------
        else:
            if ann_file is None:
                raise FileNotFoundError("CSV mode requires ann_file. Pass ann_file explicitly.")

            df = pd.read_csv(ann_file, header=0 if csv_has_header else None)

            if not csv_has_header:
                if csv_columns is None:
                    # default: col0 col1 col2...
                    csv_columns = [f"col{i}" for i in range(df.shape[1])]
                df.columns = csv_columns[: df.shape[1]]
            else:
                df = _strip_and_drop_unnamed(df)
                df.columns = [str(c).strip() for c in df.columns]

            if img_dir is None:
                img_dir = _auto_img_dir(ds_root, subset)

            # images
            if img_from_dir:
                self.images = _list_images(img_dir, sort_mode=img_sort)
                if len(self.images) != len(df):
                    raise ValueError(
                        f"img_from_dir=True but images ({len(self.images)}) != rows ({len(df)}).\n"
                        f"img_dir={img_dir}\nann_file={ann_file}"
                    )
            else:
                if img_col is None:
                    img_col = df.columns[0]  # default first col
                if isinstance(img_col, int):
                    img_col = df.columns[img_col]

                self.images = [_resolve_path(ds_root, img_dir, v) for v in df[img_col].tolist()]

            # labels
            if target_cols is None:
                raise ValueError("CSV mode requires target_cols. Pass target_cols explicitly.")

            tgt_names = []
            for c in target_cols:
                if isinstance(c, int):
                    tgt_names.append(df.columns[c])
                else:
                    tgt_names.append(str(c).strip())

            # strip target col names too (because lung_rep1 has spaces sometimes)
            df.columns = [str(c).strip() for c in df.columns]
            tgt_names = [t.strip() for t in tgt_names]

            missing = [t for t in tgt_names if t not in df.columns]
            if missing:
                raise ValueError(f"Missing target columns: {missing}\nColumns={list(df.columns)}")

            t = df[tgt_names].astype(float).values.astype(np.float32)
            if t.shape[1] == 1:
                t = t[:, 0]
            self.labels = t

        # region replace (faces only)
        self.region_replacer = None
        if region_replace is not None:
            raise ValueError("Face region replacement is not included in this medical repository.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        image = _read_image(self.images[idx], grayscale_to_rgb=self.grayscale_to_rgb)

        y = self.labels[idx]
        label_t = torch.tensor(y, dtype=torch.float32) if isinstance(y, (np.ndarray, np.generic)) else torch.tensor(float(y), dtype=torch.float32)

        if self.region_replacer is not None:
            image, new_label = self.region_replacer(image, float(label_t.item()))
            label_t = torch.tensor(float(new_label), dtype=torch.float32)

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        image_t = torch.tensor(image, dtype=torch.float32) if not torch.is_tensor(image) else image.float()
        return image_t, label_t



# ------------------------------------------------------------
# RALO datasets matching Bouthaina Slika's loading protocol
# ------------------------------------------------------------

RAW_RALO_TEST_IDS = [0, 14, 15, 16, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 61, 62, 63, 64, 65, 66, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 111, 112, 113, 162, 163, 164, 165, 166, 167, 168, 169, 173, 174, 180, 181, 257, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 312, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 404, 405, 406, 407, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 471, 474, 475, 476, 477, 478, 517, 530, 531, 532, 533, 534, 564, 565, 566, 572, 606, 613, 659, 660, 661, 662, 663, 664, 665, 668, 771, 772, 773, 774, 775, 776, 777, 781, 782, 783, 784, 786, 787, 788, 789, 790, 791, 792, 793, 794, 795, 796, 797, 798, 799, 800, 801, 802, 810, 869, 883, 884, 885, 886, 887, 888, 889, 890, 891, 892, 893, 894, 896, 897, 898, 899, 916, 917, 918, 919, 920, 921, 922, 923, 924, 925, 926, 927, 930, 932, 976, 977, 978, 979, 983, 984, 985, 986, 987, 988, 989, 990, 991, 992, 1010, 1011, 1071, 1083, 1084, 1085, 1086, 1087, 1107, 1108, 1126, 1127, 1128, 1129, 1130, 1131, 1132, 1133, 1134, 1135, 1136, 1137, 1151, 1152, 1153, 1154, 1155, 1156, 1157, 1158, 1159, 1160, 1161, 1162, 1163, 1164, 1165, 1167, 1168, 1169, 1170, 1171, 1172, 1173, 1174, 1175, 1176, 1177, 1179, 1221, 1222, 1223, 1224, 1225, 1226, 1238, 1239, 1271, 1272, 1273, 1274, 1275, 1276, 1277, 1278, 1279, 1280, 1281, 1282, 1283, 1284, 1285, 1286, 1287, 1316, 1324, 1325, 1326, 1327, 1328, 1329, 1330, 1331, 1332, 1333, 1334, 1335, 1336, 1337, 1338, 1339, 1340, 1341, 1342, 1373, 1374, 1375, 1376, 1377, 1378, 1379, 1380, 1381, 1382, 1383, 1384, 1385, 1386, 1507, 1508, 1509, 1527, 1529, 1530, 1531, 1532, 1561, 1562, 1587, 1598, 1599, 1600, 1601, 1602, 1603, 1604, 1605, 1606, 1607, 1608, 1609, 1627, 1671, 1672, 1673, 1674, 1675, 1676, 1677, 1682, 1683, 1692, 1693, 1694, 1695, 1696, 1697, 1698, 1699, 1715, 1745, 1749, 1762, 1763, 1772, 1800, 1801, 1802, 1803, 1813, 1814, 1817, 1818, 1819, 1820, 1821, 1822, 1823, 1824, 1825, 1826, 1827, 1828, 1829, 1830, 1831, 1832, 1833, 1834, 1869, 1870, 1871, 1872, 1876, 1877, 1878, 1879, 1880, 1881, 1910, 1936, 1937, 1938, 1941, 1942, 1943, 1944, 1945, 1946, 1947, 1948, 1949, 1950, 1961, 1978, 1979, 1980, 2009, 2017, 2018, 2019, 2020, 2021, 2022, 2069, 2070, 2071, 2072, 2073, 2074, 2108, 2109, 2155, 2160, 2161, 2162, 2163, 2164, 2165, 2166, 2167, 2182, 2183, 2189, 2195, 2196, 2197, 2198, 2199, 2200, 2201, 2202, 2203, 2204, 2213, 2214, 2215, 2216, 2217, 2218, 2225, 2255, 2257, 2258, 2259, 2260, 2261, 2262, 2263, 2264, 2265, 2277, 2278, 2279, 2285, 2286, 2287, 2288, 2289, 2291, 2340, 2342, 2345, 2354, 2355, 2356, 2358]
RALO_AUG_TEST_IDS = [1186, 1200, 1201, 1208, 1209, 1211, 1212, 1213, 1214, 1215, 1216, 1217, 1218, 1219, 1220, 1221, 1222, 1223, 1224, 1225, 1226, 1227, 1228, 1229, 1247, 1248, 1249, 1250, 1251, 1252, 1260, 1261, 1262, 1263, 1264, 1265, 1266, 1267, 1268, 1270, 1297, 1298, 1299, 1348, 1349, 1350, 1351, 1352, 1353, 1354, 1355, 1359, 1360, 1366, 1367, 1443, 1445, 1446, 1447, 1448, 1449, 1450, 1451, 1452, 1453, 1454, 1455, 1456, 1457, 1458, 1459, 1460, 1461, 1462, 1463, 1498, 1500, 1501, 1502, 1503, 1504, 1505, 1506, 1507, 1508, 1509, 1510, 1511, 1512, 1513, 1514, 1515, 1516, 1517, 1590, 1591, 1592, 1593, 1603, 1604, 1605, 1606, 1607, 1608, 1609, 1610, 1611, 1612, 1613, 1614, 1615, 1616, 1617, 1618, 1619, 1620, 1621, 1657, 1660, 1661, 1662, 1663, 1664, 1703, 1716, 1717, 1718, 1719, 1720, 1750, 1751, 1752, 1758, 1792, 1799, 1845, 1846, 1847, 1848, 1849, 1850, 1851, 1854, 1957, 1958, 1959, 1960, 1961, 1962, 1963, 1967, 1968, 1969, 1970, 1972, 1973, 1974, 1975, 1976, 1977, 1978, 1979, 1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1996, 2055, 2069, 2070, 2071, 2072, 2073, 2074, 2075, 2076, 2077, 2078, 2079, 2080, 2082, 2083, 2084, 2085, 2102, 2103, 2104, 2105, 2106, 2107, 2108, 2109, 2110, 2111, 2112, 2113, 2116, 2118, 2162, 2163, 2164, 2165, 2169, 2170, 2171, 2172, 2173, 2174, 2175, 2176, 2177, 2178, 2196, 2197, 2257, 2269, 2270, 2271, 2272, 2273, 2293, 2294, 2312, 2313, 2314, 2315, 2316, 2317, 2318, 2319, 2320, 2321, 2322, 2323, 2337, 2338, 2339, 2340, 2341, 2342, 2343, 2344, 2345, 2346, 2347, 2348, 2349, 2350, 2351, 2353, 2354, 2355, 2356, 2357, 2358, 2359, 2360, 2361, 2362, 2363, 2365, 2407, 2408, 2409, 2410, 2411, 2412, 2424, 2425, 2457, 2458, 2459, 2460, 2461, 2462, 2463, 2464, 2465, 2466, 2467, 2468, 2469, 2470, 2471, 2472, 2473, 2502, 2510, 2511, 2512, 2513, 2514, 2515, 2516, 2517, 2518, 2519, 2520, 2521, 2522, 2523, 2524, 2525, 2526, 2527, 2528, 2559, 2560, 2561, 2562, 2563, 2564, 2565, 2566, 2567, 2568, 2569, 2570, 2571, 2572, 2693, 2694, 2695, 2713, 2715, 2716, 2717, 2718, 2747, 2748, 2773, 2784, 2785, 2786, 2787, 2788, 2789, 2790, 2791, 2792, 2793, 2794, 2795, 2813, 2857, 2858, 2859, 2860, 2861, 2862, 2863, 2868, 2869, 2878, 2879, 2880, 2881, 2882, 2883, 2884, 2885, 2901, 2931, 2935, 2948, 2949, 2958, 2986, 2987, 2988, 2989, 2999, 3000, 3003, 3004, 3005, 3006, 3007, 3008, 3009, 3010, 3011, 3012, 3013, 3014, 3015, 3016, 3017, 3018, 3019, 3020, 3055, 3056, 3057, 3058, 3062, 3063, 3064, 3065, 3066, 3067, 3096, 3122, 3123, 3124, 3127, 3128, 3129, 3130, 3131, 3132, 3133, 3134, 3135, 3136, 3147, 3164, 3165, 3166, 3195, 3203, 3204, 3205, 3206, 3207, 3208, 3255, 3256, 3257, 3258, 3259, 3260, 3294, 3295, 3341, 3346, 3347, 3348, 3349, 3350, 3351, 3352, 3353, 3368, 3369, 3375, 3381, 3382, 3383, 3384, 3385, 3386, 3387, 3388, 3389, 3390, 3399, 3400, 3401, 3402, 3403, 3404, 3411, 3441, 3443, 3444, 3445, 3446, 3447, 3448, 3449, 3450, 3451, 3463, 3464, 3465, 3471, 3472, 3473, 3474, 3475, 3477, 3526, 3528, 3531, 3540, 3541, 3542, 3544]


# def _read_image(path: str) -> np.ndarray:
#     buf = np.fromfile(path, dtype=np.uint8)
#     img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
#     if img is None:
#         raise FileNotFoundError(f"Could not read image: {path}")
#     if img.ndim == 2:
#         img = np.stack([img, img, img], axis=-1)
#     if img.ndim == 3 and img.shape[2] == 4:
#         img = img[:, :, :3]
#     if img.ndim == 3 and img.shape[2] == 3:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     return img


def _infer_protocol(ann_file: str) -> str:
    name = os.path.basename(str(ann_file)).lower()
    if name == 'lung_rep1.csv':
        return 'aug'
    if name == 'ralo-dataset-metadata.csv':
        return 'raw'
    raise ValueError(
        f"Could not infer RALO protocol from ann_file={ann_file}. "
        "Use protocol='raw' or protocol='aug'."
    )


def _build_split_indices(n_total: int, test_ids, subset: str, val_ratio: float = -1.0, val_seed: int = 42):
    test_ids = sorted([i for i in test_ids if 0 <= i < n_total])
    train_ids = [i for i in range(n_total) if i not in set(test_ids)]

    if subset == 'test':
        return test_ids

    if val_ratio is not None and val_ratio > 0:
        rng = random.Random(val_seed)
        pool = train_ids.copy()
        rng.shuffle(pool)
        n_val = max(1, int(round(len(pool) * float(val_ratio))))
        val_ids = sorted(pool[:n_val])
        train_ids = sorted(pool[n_val:])
        if subset == 'val':
            return val_ids
    elif subset == 'val':
        raise ValueError("subset='val' requires val_ratio > 0")

    if subset == 'train':
        return sorted(train_ids)

    raise ValueError(f"Invalid subset={subset}. Use train / val / test.")


class _BaseRALO(Dataset):
    def __init__(self, path, dataset, subset, transform, ann_file=None, img_dir=None, protocol='auto', val_ratio=-1.0, val_seed=42):
        self.path = path
        self.dataset = dataset
        self.subset = subset
        self.transform = get_transform(transform)
        self.val_ratio = val_ratio
        self.val_seed = val_seed

        ds_root = os.path.join(path, dataset)
        if ann_file is None:
            raise ValueError('Pass ann_file explicitly for RALO datasets.')
        if img_dir is None:
            raise ValueError('Pass img_dir explicitly for RALO datasets.')

        self.ann_file = ann_file
        self.img_dir = img_dir
        self.protocol = _infer_protocol(ann_file) if protocol == 'auto' else protocol
        if self.protocol not in ('raw', 'aug'):
            raise ValueError(f"protocol must be 'auto', 'raw' or 'aug', got {protocol}")

        self.df = self._load_dataframe()
        self.keep_ids = _build_split_indices(
            n_total=len(self.df),
            test_ids=RAW_RALO_TEST_IDS if self.protocol == 'raw' else RALO_AUG_TEST_IDS,
            subset=subset,
            val_ratio=val_ratio,
            val_seed=val_seed,
        )
        self.df = self.df.iloc[self.keep_ids].copy()
        self.df['_global_id'] = self.keep_ids
        self.images = [os.path.join(self.img_dir, f"{gid}.jpg") for gid in self.keep_ids]

    def _load_dataframe(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.df)

    def _load_image(self, idx: int):
        image = _read_image(self.images[idx])
        if self.transform is not None:
            image = self.transform(image=image)['image']
        if not torch.is_tensor(image):
            image = torch.tensor(image, dtype=torch.float32)
        else:
            image = image.float()
        return image


@DATASETS.register('ralo_bouthaina_single')
class RALOBouthainaSingle(_BaseRALO):
    """
    Single-target RALO loader that plugs into the AgeEstimation registry.

    Config keys:
      type='ralo_bouthaina_single'
      path='/.../datasets'
      dataset='RALO_Aug' or 'RALO'
      subset='train'|'val'|'test'
      transform=[...]
      ann_file='/.../lung_rep1.csv' OR '/.../ralo-dataset-metadata.csv'
      img_dir='/.../Lung_Rep1' OR '/.../CXR_images_scored'
      score='GE' or 'LO'
      protocol='auto'|'raw'|'aug'
      val_ratio=-1 or 0.1
      val_seed=42
      aug_label_mode='total' or 'avg_lr'
    """
    def __init__(self, path, dataset, subset, transform, score='GE', ann_file=None, img_dir=None,
                 protocol='auto', val_ratio=-1.0, val_seed=42, aug_label_mode='total'):
        self.score = str(score).upper()
        self.aug_label_mode = aug_label_mode
        super().__init__(path, dataset, subset, transform, ann_file, img_dir, protocol, val_ratio, val_seed)

        if self.score == 'GE':
            self.labels = self.df['Geographic Extent'].astype(np.float32).to_numpy()
        elif self.score == 'LO':
            self.labels = self.df['Lung Opacity'].astype(np.float32).to_numpy()
        else:
            raise ValueError(f"score must be 'GE' or 'LO', got {score}")

    def _load_dataframe(self):
        if self.protocol == 'raw':
            df = pd.read_csv(self.ann_file, header=1)
            df['Geographic Extent'] = (df['Total GEOGRAPHIC'].astype(float) + df['Total GEOGRAPHIC.1'].astype(float)) / 2.0
            df['Lung Opacity'] = (df['Total OPACITY'].astype(float) + df['Total OPACITY.1'].astype(float)) / 2.0
            return df

        df = pd.read_csv(self.ann_file, header=0)
        df.columns = [str(c).strip() for c in df.columns]
        if self.aug_label_mode == 'avg_lr':
            df['Geographic Extent'] = (df['Left Geo'].astype(float) + df['Right Geo'].astype(float)) / 2.0
            df['Lung Opacity'] = (df['Left Opacity'].astype(float) + df['Right Opacity'].astype(float)) / 2.0
        else:
            df['Geographic Extent'] = df['TOTAL Geo'].astype(float)
            df['Lung Opacity'] = df['TOTAL Opacity'].astype(float)
        return df

    def __getitem__(self, idx):
        image = self._load_image(idx)
        label = torch.tensor(float(self.labels[idx]), dtype=torch.float32)
        return image, label


@DATASETS.register('ralo_bouthaina_multi')
class RALOBouthainaMulti(_BaseRALO):
    """
    Two-target RALO loader returning [GE, LO].

    Same config as ralo_bouthaina_single except score is not needed.
    """
    def __init__(self, path, dataset, subset, transform, ann_file=None, img_dir=None,
                 protocol='auto', val_ratio=-1.0, val_seed=42, aug_label_mode='total'):
        self.aug_label_mode = aug_label_mode
        super().__init__(path, dataset, subset, transform, ann_file, img_dir, protocol, val_ratio, val_seed)
        self.labels = self.df[['Geographic Extent', 'Lung Opacity']].astype(np.float32).to_numpy()

    def _load_dataframe(self):
        if self.protocol == 'raw':
            df = pd.read_csv(self.ann_file, header=1)
            df['Geographic Extent'] = (df['Total GEOGRAPHIC'].astype(float) + df['Total GEOGRAPHIC.1'].astype(float)) / 2.0
            df['Lung Opacity'] = (df['Total OPACITY'].astype(float) + df['Total OPACITY.1'].astype(float)) / 2.0
            return df

        df = pd.read_csv(self.ann_file, header=0)
        df.columns = [str(c).strip() for c in df.columns]
        if self.aug_label_mode == 'avg_lr':
            df['Geographic Extent'] = (df['Left Geo'].astype(float) + df['Right Geo'].astype(float)) / 2.0
            df['Lung Opacity'] = (df['Left Opacity'].astype(float) + df['Right Opacity'].astype(float)) / 2.0
        else:
            df['Geographic Extent'] = df['TOTAL Geo'].astype(float)
            df['Lung Opacity'] = df['TOTAL Opacity'].astype(float)
        return df

    def __getitem__(self, idx):
        image = self._load_image(idx)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label


# RALO_BOUTHAINA_SINGLE_TEST_IDS = [0, 14, 15, 16, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 61, 62, 63, 64, 65, 66, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 111, 112, 113, 162, 163, 164, 165, 166, 167, 168, 169, 173, 174, 180, 181, 257, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 312, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 404, 405, 406, 407, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 471, 474, 475, 476, 477, 478, 517, 530, 531, 532, 533, 534, 564, 565, 566, 572, 606, 613, 659, 660, 661, 662, 663, 664, 665, 668, 771, 772, 773, 774, 775, 776, 777, 781, 782, 783, 784, 786, 787, 788, 789, 790, 791, 792, 793, 794, 795, 796, 797, 798, 799, 800, 801, 802, 810, 869, 883, 884, 885, 886, 887, 888, 889, 890, 891, 892, 893, 894, 896, 897, 898, 899, 916, 917, 918, 919, 920, 921, 922, 923, 924, 925, 926, 927, 930, 932, 976, 977, 978, 979, 983, 984, 985, 986, 987, 988, 989, 990, 991, 992, 1010, 1011, 1071, 1083, 1084, 1085, 1086, 1087, 1107, 1108, 1126, 1127, 1128, 1129, 1130, 1131, 1132, 1133, 1134, 1135, 1136, 1137, 1151, 1152, 1153, 1154, 1155, 1156, 1157, 1158, 1159, 1160, 1161, 1162, 1163, 1164, 1165, 1167, 1168, 1169, 1170, 1171, 1172, 1173, 1174, 1175, 1176, 1177, 1179, 1221, 1222, 1223, 1224, 1225, 1226, 1238, 1239, 1271, 1272, 1273, 1274, 1275, 1276, 1277, 1278, 1279, 1280, 1281, 1282, 1283, 1284, 1285, 1286, 1287, 1316, 1324, 1325, 1326, 1327, 1328, 1329, 1330, 1331, 1332, 1333, 1334, 1335, 1336, 1337, 1338, 1339, 1340, 1341, 1342, 1373, 1374, 1375, 1376, 1377, 1378, 1379, 1380, 1381, 1382, 1383, 1384, 1385, 1386, 1507, 1508, 1509, 1527, 1529, 1530, 1531, 1532, 1561, 1562, 1587, 1598, 1599, 1600, 1601, 1602, 1603, 1604, 1605, 1606, 1607, 1608, 1609, 1627, 1671, 1672, 1673, 1674, 1675, 1676, 1677, 1682, 1683, 1692, 1693, 1694, 1695, 1696, 1697, 1698, 1699, 1715, 1745, 1749, 1762, 1763, 1772, 1800, 1801, 1802, 1803, 1813, 1814, 1817, 1818, 1819, 1820, 1821, 1822, 1823, 1824, 1825, 1826, 1827, 1828, 1829, 1830, 1831, 1832, 1833, 1834, 1869, 1870, 1871, 1872, 1876, 1877, 1878, 1879, 1880, 1881, 1910, 1936, 1937, 1938, 1941, 1942, 1943, 1944, 1945, 1946, 1947, 1948, 1949, 1950, 1961, 1978, 1979, 1980, 2009, 2017, 2018, 2019, 2020, 2021, 2022, 2069, 2070, 2071, 2072, 2073, 2074, 2108, 2109, 2155, 2160, 2161, 2162, 2163, 2164, 2165, 2166, 2167, 2182, 2183, 2189, 2195, 2196, 2197, 2198, 2199, 2200, 2201, 2202, 2203, 2204, 2213, 2214, 2215, 2216, 2217, 2218, 2225, 2255, 2257, 2258, 2259, 2260, 2261, 2262, 2263, 2264, 2265, 2277, 2278, 2279, 2285, 2286, 2287, 2288, 2289, 2291, 2340, 2342, 2345, 2354, 2355, 2356, 2358]
# RALO_BOUTHAINA_SINGLE_TEST_IDS = [1186, 1200, 1201, 1208, 1209, 1211, 1212, 1213, 1214, 1215, 1216, 1217, 1218, 1219, 1220, 1221, 1222, 1223, 1224, 1225, 1226, 1227, 1228, 1229, 1247, 1248, 1249, 1250, 1251, 1252, 1260, 1261, 1262, 1263, 1264, 1265, 1266, 1267, 1268, 1270, 1297, 1298, 1299, 1348, 1349, 1350, 1351, 1352, 1353, 1354, 1355, 1359, 1360, 1366, 1367, 1443, 1445, 1446, 1447, 1448, 1449, 1450, 1451, 1452, 1453, 1454, 1455, 1456, 1457, 1458, 1459, 1460, 1461, 1462, 1463, 1498, 1500, 1501, 1502, 1503, 1504, 1505, 1506, 1507, 1508, 1509, 1510, 1511, 1512, 1513, 1514, 1515, 1516, 1517, 1590, 1591, 1592, 1593, 1603, 1604, 1605, 1606, 1607, 1608, 1609, 1610, 1611, 1612, 1613, 1614, 1615, 1616, 1617, 1618, 1619, 1620, 1621, 1657, 1660, 1661, 1662, 1663, 1664, 1703, 1716, 1717, 1718, 1719, 1720, 1750, 1751, 1752, 1758, 1792, 1799, 1845, 1846, 1847, 1848, 1849, 1850, 1851, 1854, 1957, 1958, 1959, 1960, 1961, 1962, 1963, 1967, 1968, 1969, 1970, 1972, 1973, 1974, 1975, 1976, 1977, 1978, 1979, 1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1996, 2055, 2069, 2070, 2071, 2072, 2073, 2074, 2075, 2076, 2077, 2078, 2079, 2080, 2082, 2083, 2084, 2085, 2102, 2103, 2104, 2105, 2106, 2107, 2108, 2109, 2110, 2111, 2112, 2113, 2116, 2118, 2162, 2163, 2164, 2165, 2169, 2170, 2171, 2172, 2173, 2174, 2175, 2176, 2177, 2178, 2196, 2197, 2257, 2269, 2270, 2271, 2272, 2273, 2293, 2294, 2312, 2313, 2314, 2315, 2316, 2317, 2318, 2319, 2320, 2321, 2322, 2323, 2337, 2338, 2339, 2340, 2341, 2342, 2343, 2344, 2345, 2346, 2347, 2348, 2349, 2350, 2351, 2353, 2354, 2355, 2356, 2357, 2358, 2359, 2360, 2361, 2362, 2363, 2365, 2407, 2408, 2409, 2410, 2411, 2412, 2424, 2425, 2457, 2458, 2459, 2460, 2461, 2462, 2463, 2464, 2465, 2466, 2467, 2468, 2469, 2470, 2471, 2472, 2473, 2502, 2510, 2511, 2512, 2513, 2514, 2515, 2516, 2517, 2518, 2519, 2520, 2521, 2522, 2523, 2524, 2525, 2526, 2527, 2528, 2559, 2560, 2561, 2562, 2563, 2564, 2565, 2566, 2567, 2568, 2569, 2570, 2571, 2572, 2693, 2694, 2695, 2713, 2715, 2716, 2717, 2718, 2747, 2748, 2773, 2784, 2785, 2786, 2787, 2788, 2789, 2790, 2791, 2792, 2793, 2794, 2795, 2813, 2857, 2858, 2859, 2860, 2861, 2862, 2863, 2868, 2869, 2878, 2879, 2880, 2881, 2882, 2883, 2884, 2885, 2901, 2931, 2935, 2948, 2949, 2958, 2986, 2987, 2988, 2989, 2999, 3000, 3003, 3004, 3005, 3006, 3007, 3008, 3009, 3010, 3011, 3012, 3013, 3014, 3015, 3016, 3017, 3018, 3019, 3020, 3055, 3056, 3057, 3058, 3062, 3063, 3064, 3065, 3066, 3067, 3096, 3122, 3123, 3124, 3127, 3128, 3129, 3130, 3131, 3132, 3133, 3134, 3135, 3136, 3147, 3164, 3165, 3166, 3195, 3203, 3204, 3205, 3206, 3207, 3208, 3255, 3256, 3257, 3258, 3259, 3260, 3294, 3295, 3341, 3346, 3347, 3348, 3349, 3350, 3351, 3352, 3353, 3368, 3369, 3375, 3381, 3382, 3383, 3384, 3385, 3386, 3387, 3388, 3389, 3390, 3399, 3400, 3401, 3402, 3403, 3404, 3411, 3441, 3443, 3444, 3445, 3446, 3447, 3448, 3449, 3450, 3451, 3463, 3464, 3465, 3471, 3472, 3473, 3474, 3475, 3477, 3526, 3528, 3531, 3540, 3541, 3542, 3544]


# def _resolve_numeric_image(img_dir: str, idx: int) -> str:
#     """Resolve <idx> image in img_dir while keeping Bouthaina's numeric mapping."""
#     for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
#         p = os.path.join(img_dir, f"{idx}{ext}")
#         if os.path.exists(p):
#             return p
#     raise FileNotFoundError(f"Could not find numeric image for index={idx} in {img_dir}")


# def _slice_with_indices(items, indices: List[int]):
#     if isinstance(items, np.ndarray):
#         return items[indices]
#     return [items[i] for i in indices]


# def _make_train_val_indices(train_indices: List[int], val_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
#     if val_ratio <= 0:
#         return train_indices, []
#     rng = np.random.RandomState(seed)
#     shuffled = np.array(train_indices, dtype=np.int64)
#     rng.shuffle(shuffled)
#     n_val = max(1, int(round(len(shuffled) * float(val_ratio))))
#     val_idx = sorted(shuffled[:n_val].tolist())
#     train_idx = sorted(shuffled[n_val:].tolist())
#     return train_idx, val_idx


# class _BaseRALOBouthainaDataset(Dataset):
#     def __init__(
#         self,
#         path: str,
#         dataset: str,
#         subset: str,
#         transform: list,
#         ann_file: Optional[str] = None,
#         img_dir: Optional[str] = None,
#         grayscale_to_rgb: bool = True,
#         val_ratio: float = 0.0,
#         val_seed: int = 42,
#     ):
#         self.path = path
#         self.dataset = dataset
#         self.subset = str(subset).lower()
#         self.transform = get_transform(transform)
#         self.grayscale_to_rgb = grayscale_to_rgb
#         self.val_ratio = float(val_ratio)
#         self.val_seed = int(val_seed)
#         self.ds_root = os.path.join(path, dataset)

#         if ann_file is None:
#             ann_file = self.default_ann_file(self.ds_root)
#         if img_dir is None:
#             img_dir = os.path.join(self.ds_root, "Lung_Rep1")

#         self.ann_file = ann_file
#         self.img_dir = img_dir

#         self.images: List[str] = []
#         self.labels: Union[List[float], np.ndarray] = []

#     def default_ann_file(self, ds_root: str) -> str:
#         raise NotImplementedError

#     def _finalize_subset(self, test_indices: List[int]):
#         total_n = len(self.images)
#         all_idx = list(range(total_n))
#         test_set = set(test_indices)
#         train_pool = [i for i in all_idx if i not in test_set]
#         train_idx, val_idx = _make_train_val_indices(train_pool, self.val_ratio, self.val_seed)

#         if self.subset == "train":
#             chosen = train_idx
#         elif self.subset == "val":
#             if self.val_ratio <= 0:
#                 raise ValueError(
#                     "subset='val' requested but val_ratio=0. Set val_ratio>0 or reuse train/test explicitly."
#                 )
#             chosen = val_idx
#         elif self.subset == "test":
#             chosen = list(test_indices)
#         else:
#             raise ValueError(f"Unsupported subset '{self.subset}'. Use train/val/test.")

#         self.images = _slice_with_indices(self.images, chosen)
#         self.labels = _slice_with_indices(self.labels, chosen)

#     def __len__(self):
#         return len(self.images)

#     def __getitem__(self, idx: int):
#         image = _read_image(self.images[idx], grayscale_to_rgb=self.grayscale_to_rgb)
#         y = self.labels[idx]
#         if isinstance(y, np.ndarray):
#             label_t = torch.tensor(y, dtype=torch.float32)
#         elif isinstance(y, (list, tuple)):
#             label_t = torch.tensor(np.asarray(y, dtype=np.float32), dtype=torch.float32)
#         else:
#             label_t = torch.tensor(float(y), dtype=torch.float32)

#         if self.transform is not None:
#             image = self.transform(image=image)["image"]

#         image_t = torch.tensor(image, dtype=torch.float32) if not torch.is_tensor(image) else image.float()
#         return image_t, label_t

#     def __str__(self):
#         return f"{self.__class__.__name__}({self.subset}): {len(self)} images"


# @DATASETS.register("ralo_bouthaina_single")
# class RALOBouthainaSingleDataset(_BaseRALOBouthainaDataset):
#     """
#     Reproduces Bouthaina's ViTReg-IP RALO loader.

#     - Reads the raw RALO metadata CSV.
#     - Builds image paths as 0.jpg, 1.jpg, ... from Lung_Rep1.
#     - Uses the fixed official test split from her data.py.
#     - score='GE' or score='LO' for one-target regression.
#     - Optional val subset can be carved from the training pool with val_ratio.
#     """

#     def __init__(
#         self,
#         path: str,
#         dataset: str,
#         subset: str,
#         transform: list,
#         score: str = "GE",
#         ann_file: Optional[str] = None,
#         img_dir: Optional[str] = None,
#         grayscale_to_rgb: bool = True,
#         val_ratio: float = 0.0,
#         val_seed: int = 42,
#     ):
#         self.score = str(score).upper()
#         super().__init__(
#             path=path,
#             dataset=dataset,
#             subset=subset,
#             transform=transform,
#             ann_file=ann_file,
#             img_dir=img_dir,
#             grayscale_to_rgb=grayscale_to_rgb,
#             val_ratio=val_ratio,
#             val_seed=val_seed,
#         )

#         df = pd.read_csv(self.ann_file, skiprows=1)
#         df.columns = [str(c).strip() for c in df.columns]

#         if self.score == "GE":
#             # labels = (df["Total GEOGRAPHIC"].astype(float) + df["Total GEOGRAPHIC.1"].astype(float)) / 2.0
#             labels = (df["Left Geo"].astype(float) + df["Right Geo"].astype(float)) / 2.0
#         elif self.score == "LO":
#             # labels = (df["Total OPACITY"].astype(float) + df["Total OPACITY.1"].astype(float)) / 2.0
#             labels = (df["Left Opacity"].astype(float) + df["Right Opacity"].astype(float)) / 2.0
#         else:
#             raise ValueError(f"score must be 'GE' or 'LO', got {score}")

#         self.labels = labels.values.astype(np.float32)
#         self.images = [_resolve_numeric_image(self.img_dir, idx) for idx in range(len(self.labels))]
#         self._finalize_subset(RALO_BOUTHAINA_SINGLE_TEST_IDS)

#     def default_ann_file(self, ds_root: str) -> str:
#         for name in ("ralo-dataset-metadata.csv", "lung_rep1.csv"):
#             cand = os.path.join(ds_root, name)
#             if os.path.exists(cand):
#                 return cand
#         raise FileNotFoundError("Could not auto-find raw RALO metadata CSV.")


# @DATASETS.register("ralo_bouthaina_multi")
# class RALOBouthainaMultiDataset(_BaseRALOBouthainaDataset):
#     """
#     Reproduces Bouthaina's MViTReg-IP RALO loader.

#     - Reads lung_rep1.csv with TOTAL Geo / TOTAL Opacity.
#     - Builds image paths as 0.jpg, 1.jpg, ... from Lung_Rep1.
#     - Uses the fixed official test split from her multidata.py.
#     - target_mode='both' returns [GE, LO], while 'GE' or 'LO' returns a single target.
#     - Optional val subset can be carved from the training pool with val_ratio.
#     """

#     def __init__(
#         self,
#         path: str,
#         dataset: str,
#         subset: str,
#         transform: list,
#         target_mode: str = "both",
#         ann_file: Optional[str] = None,
#         img_dir: Optional[str] = None,
#         grayscale_to_rgb: bool = True,
#         val_ratio: float = 0.0,
#         val_seed: int = 42,
#     ):
#         self.target_mode = str(target_mode).upper()
#         super().__init__(
#             path=path,
#             dataset=dataset,
#             subset=subset,
#             transform=transform,
#             ann_file=ann_file,
#             img_dir=img_dir,
#             grayscale_to_rgb=grayscale_to_rgb,
#             val_ratio=val_ratio,
#             val_seed=val_seed,
#         )

#         df = pd.read_csv(self.ann_file)
#         df = _strip_and_drop_unnamed(df)
#         df.columns = [str(c).strip() for c in df.columns]

#         ge = df["TOTAL Geo"].astype(float).values.astype(np.float32)
#         lo = df["TOTAL Opacity"].astype(float).values.astype(np.float32)

#         if self.target_mode == "BOTH":
#             self.labels = np.stack([ge, lo], axis=1).astype(np.float32)
#         elif self.target_mode == "GE":
#             self.labels = ge
#         elif self.target_mode == "LO":
#             self.labels = lo
#         else:
#             raise ValueError(f"target_mode must be BOTH, GE or LO, got {target_mode}")

#         self.images = [_resolve_numeric_image(self.img_dir, idx) for idx in range(len(ge))]
#         self._finalize_subset(RALO_BOUTHAINA_MULTI_TEST_IDS)

#     def default_ann_file(self, ds_root: str) -> str:
#         for name in ("lung_rep1.csv", "ralo_dataset_final.csv"):
#             cand = os.path.join(ds_root, name)
#             if os.path.exists(cand):
#                 return cand
#         raise FileNotFoundError("Could not auto-find lung_rep1.csv / aggregated RALO CSV.")
