# Hướng dẫn chi tiết K-Means, KNN, KD-Tree, Ball Tree và Classification Metrics trong Scikit-Learn

> Tài liệu được chỉnh sửa và bổ sung từ bản Markdown ban đầu.  
> Các công thức toán học đã được sửa lại theo cú pháp LaTeX chuẩn để hiển thị tốt trên GitHub, VS Code, Obsidian và Jupyter Notebook.
> **Lưu ý hiển thị công thức:** Các công thức dạng khối được viết trên một dòng theo dạng `$$ công thức $$` để tránh lỗi ở một số trình Markdown.


---

## MỤC LỤC

1. [Quy ước dữ liệu trong Scikit-Learn](#0-quy-ước-dữ-liệu-trong-scikit-learn)
2. [Phân cụm K-Means](#1-phân-cụm-k-means-kmeans)
3. [K-Nearest Neighbors Classification](#2-k-nearest-neighbors-classification-kneighborsclassifier)
4. [KD-Tree và Ball Tree](#3-kd-tree-và-ball-tree)
5. [Classification Metrics](#4-classification-metrics)
6. [Ví dụ tổng hợp](#5-ví-dụ-tổng-hợp)
7. [Bảng ghi nhớ nhanh](#6-bảng-ghi-nhớ-nhanh)

---

# 0. Quy ước dữ liệu trong Scikit-Learn

Các model trong Scikit-Learn thường nhận dữ liệu đặc trưng `X` có dạng:

```text
X.shape = (n_samples, n_features)
```

Trong đó:

- `n_samples`: số mẫu dữ liệu;
- `n_features`: số đặc trưng của mỗi mẫu.

Ví dụ:

```python
import numpy as np

X = np.array([
    [1.0, 2.0],
    [1.5, 1.8],
    [8.0, 8.0],
    [9.0, 8.5]
])

print(X.shape)  # (4, 2)
```

Nếu dữ liệu chỉ có một đặc trưng, vẫn phải chuyển về ma trận hai chiều:

```python
X = np.array([1.0, 2.0, 3.0]).reshape(-1, 1)

print(X.shape)  # (3, 1)
```

---

# 1. Phân cụm K-Means (`KMeans`)

`sklearn.cluster.KMeans` là thuật toán học không giám sát, dùng để chia dữ liệu thành \(K\) cụm dựa trên khoảng cách giữa các điểm dữ liệu và các tâm cụm.

## 1.1. Mục tiêu toán học của K-Means

K-Means tìm cách tối thiểu hóa **Within-Cluster Sum of Squares**, còn được gọi là **WCSS** hoặc **inertia**:

$$ \operatorname{WCSS} = \sum_{i=1}^{K} \sum_{\mathbf{x}\in C_i} \left\| \mathbf{x}-\boldsymbol{\mu}_i \right\|_2^2 $$

Trong đó:

- \(K\): số cụm;
- \(C_i\): tập các điểm thuộc cụm thứ \(i\);
- \(\mathbf{x}\): một điểm dữ liệu;
- \(\boldsymbol{\mu}_i\): centroid của cụm \(i\);
- \(\|\mathbf{x}-\boldsymbol{\mu}_i\|_2^2\): bình phương khoảng cách Euclid.

Centroid của cụm \(C_i\) được tính bằng trung bình của các điểm trong cụm:

$$ \boldsymbol{\mu}_i = \frac{1}{|C_i|} \sum_{\mathbf{x}\in C_i}\mathbf{x} $$

Khoảng cách Euclid giữa hai vector \(\mathbf{x}\) và \(\mathbf{y}\):

$$ d(\mathbf{x},\mathbf{y}) = \sqrt{ \sum_{j=1}^{D} (x_j-y_j)^2 } $$

Trong WCSS, ta dùng **bình phương khoảng cách**, nên không cần căn bậc hai:

$$ d^2(\mathbf{x},\mathbf{y}) = \sum_{j=1}^{D} (x_j-y_j)^2 $$

---

## 1.2. Các bước của thuật toán

1. Chọn \(K\) centroid ban đầu.
2. Gán mỗi điểm vào centroid gần nhất.
3. Tính lại centroid của từng cụm.
4. Lặp lại bước 2 và bước 3.
5. Dừng khi centroid gần như không thay đổi hoặc đạt `max_iter`.

---

## 1.3. Cú pháp khởi tạo

```python
from sklearn.cluster import KMeans

kmeans = KMeans(
    n_clusters=8,
    init="k-means++",
    n_init="auto",
    max_iter=300,
    tol=1e-4,
    verbose=0,
    random_state=None,
    copy_x=True,
    algorithm="lloyd"
)
```

---

## 1.4. Các tham số của `KMeans`

| Tham số | Kiểu dữ liệu | Mặc định | Công dụng |
|---|---|---:|---|
| `n_clusters` | `int` | `8` | Số cụm cần tạo, đồng thời là số centroid |
| `init` | `str`, `callable`, array-like | `"k-means++"` | Cách khởi tạo centroid |
| `n_init` | `int` hoặc `"auto"` | `"auto"` | Số lần chạy với các khởi tạo khác nhau |
| `max_iter` | `int` | `300` | Số vòng lặp tối đa của mỗi lần chạy |
| `tol` | `float` | `1e-4` | Ngưỡng hội tụ |
| `verbose` | `int` | `0` | Mức độ in thông tin khi chạy |
| `random_state` | `int`, `RandomState`, `None` | `None` | Kiểm soát tính ngẫu nhiên |
| `copy_x` | `bool` | `True` | Có sao chép dữ liệu trước khi xử lý hay không |
| `algorithm` | `"lloyd"` hoặc `"elkan"` | `"lloyd"` | Biến thể thuật toán K-Means |

### `n_clusters`

```python
KMeans(n_clusters=3)
```

Chia dữ liệu thành ba cụm.

### `init`

#### `init="k-means++"`

Chọn centroid ban đầu theo chiến lược giúp các centroid ít nằm quá gần nhau.

```python
KMeans(init="k-means++")
```

#### `init="random"`

Chọn ngẫu nhiên \(K\) mẫu trong dữ liệu làm centroid.

```python
KMeans(init="random")
```

#### Truyền centroid tự định nghĩa

```python
initial_centroids = np.array([
    [1.0, 2.0],
    [8.0, 9.0]
])

model = KMeans(
    n_clusters=2,
    init=initial_centroids,
    n_init=1
)
```

Shape phải là:

```text
(n_clusters, n_features)
```

### `n_init`

```python
KMeans(n_init=10)
```

Model chạy nhiều lần với các centroid ban đầu khác nhau, rồi giữ lần có `inertia_` nhỏ nhất.

### `max_iter`

```python
KMeans(max_iter=300)
```

Số vòng lặp tối đa cho mỗi lần khởi tạo.

### `tol`

```python
KMeans(tol=1e-4)
```

Thuật toán dừng khi độ thay đổi của centroid đủ nhỏ.

### `random_state`

```python
KMeans(random_state=42)
```

Giúp kết quả có thể tái lập.

### `copy_x`

- `True`: tạo bản sao dữ liệu trước một số phép xử lý nội bộ;
- `False`: có thể tiết kiệm bộ nhớ nhưng dữ liệu có thể bị thay đổi tạm thời rồi khôi phục.

### `algorithm`

- `"lloyd"`: K-Means cổ điển;
- `"elkan"`: dùng bất đẳng thức tam giác để giảm một số phép tính khoảng cách.

---

## 1.5. Các method của `KMeans`

### `fit(X, y=None, sample_weight=None)`

Huấn luyện model và tìm centroid.

```python
kmeans.fit(X)
```

Tham số:

- `X`: dữ liệu có shape `(n_samples, n_features)`;
- `y`: bị bỏ qua;
- `sample_weight`: trọng số của từng mẫu.

Kết quả trả về chính model:

```python
returned_model = kmeans.fit(X)

print(returned_model is kmeans)  # True
```

---

### `fit_predict(X, y=None, sample_weight=None)`

Huấn luyện rồi trả về nhãn cụm của từng mẫu.

```python
labels = kmeans.fit_predict(X)
```

Shape:

```text
(n_samples,)
```

---

### `predict(X)`

Gán mỗi mẫu mới vào centroid gần nhất.

```python
new_labels = kmeans.predict(X_new)
```

Quy tắc dự đoán:

$$ \widehat{c}(\mathbf{x}) = \underset{i\in\{1,\ldots,K\}}{\operatorname{argmin}} \left\| \mathbf{x}-\boldsymbol{\mu}_i \right\|_2 $$

---

### `transform(X)`

Trả về khoảng cách từ mỗi mẫu đến từng centroid.

```python
distances = kmeans.transform(X)
```

Shape:

```text
(n_samples, n_clusters)
```

Ví dụ với hai centroid:

```text
[
    [khoảng cách đến centroid 0, khoảng cách đến centroid 1],
    ...
]
```

---

### `fit_transform(X, y=None, sample_weight=None)`

Huấn luyện rồi chuyển dữ liệu sang không gian khoảng cách tới centroid.

```python
distances = kmeans.fit_transform(X)
```

---

### `score(X, y=None, sample_weight=None)`

Trả về số âm của hàm mục tiêu K-Means.

```python
score = kmeans.score(X)
```

Thông thường:

$$ \operatorname{score}(X) = -\operatorname{WCSS}(X) $$

Do Scikit-Learn quy ước score càng lớn càng tốt, còn WCSS càng nhỏ càng tốt.

---

### `get_params(deep=True)`

Lấy dictionary các tham số.

```python
params = kmeans.get_params()
print(params)
```

---

### `set_params(**params)`

Thay đổi tham số model.

```python
kmeans.set_params(
    n_clusters=4,
    max_iter=500
)
```

Sau khi đổi tham số, cần gọi lại `fit()`.

---

## 1.6. Các attribute của `KMeans`

> Các attribute có dấu gạch dưới cuối tên chỉ xuất hiện sau khi gọi `fit()` hoặc `fit_predict()`.

| Attribute | Shape / kiểu | Công dụng |
|---|---|---|
| `cluster_centers_` | `(n_clusters, n_features)` | Tọa độ centroid |
| `labels_` | `(n_samples,)` | Nhãn cụm của từng mẫu huấn luyện |
| `inertia_` | `float` | WCSS của kết quả phân cụm |
| `n_iter_` | `int` | Số vòng lặp thực tế đã chạy |
| `n_features_in_` | `int` | Số feature model nhận khi fit |
| `feature_names_in_` | `(n_features_in_,)` | Tên feature nếu đầu vào có tên cột dạng chuỗi |

Ví dụ:

```python
kmeans.fit(X)

print(kmeans.cluster_centers_)
print(kmeans.labels_)
print(kmeans.inertia_)
print(kmeans.n_iter_)
print(kmeans.n_features_in_)
```

---

## 1.7. Elbow Method

Elbow Method thử nhiều giá trị \(K\), sau đó vẽ WCSS theo \(K\).

```python
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

wcss_values = []
k_values = range(1, 10)

for k in k_values:
    model = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=42
    )

    model.fit(X)
    wcss_values.append(model.inertia_)

plt.plot(k_values, wcss_values, marker="o")
plt.xlabel("Số cụm K")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()
```

Chọn điểm mà WCSS chuyển từ giảm mạnh sang giảm chậm.

---

# 2. K-Nearest Neighbors Classification (`KNeighborsClassifier`)

`KNeighborsClassifier` là thuật toán học có giám sát dùng cho bài toán phân loại.

KNN là **lazy learning**: model không học ra một phương trình tham số rõ ràng mà lưu dữ liệu huấn luyện để truy vấn khi dự đoán.

---

## 2.1. Quy tắc dự đoán của KNN

Với trọng số đồng đều:

$$ \widehat{y} = \underset{c}{\operatorname{argmax}} \sum_{i\in N_K(\mathbf{x})} \mathbb{1}(y_i=c) $$

Trong đó:

- \(N_K(\mathbf{x})\): tập \(K\) hàng xóm gần nhất của \(\mathbf{x}\);
- \(\mathbb{1}(y_i=c)\): bằng 1 nếu hàng xóm \(i\) thuộc lớp \(c\), ngược lại bằng 0.

Với `weights="distance"`:

$$ w_i = \frac{1}{d(\mathbf{x},\mathbf{x}_i)} $$

Phiếu của hàng xóm gần hơn có trọng số lớn hơn.

---

## 2.2. Khoảng cách Minkowski

Metric mặc định của KNN là Minkowski:

$$ d_p(\mathbf{x},\mathbf{y}) = \left( \sum_{j=1}^{D} |x_j-y_j|^p \right)^{1/p} $$

### Khi \(p=1\): Manhattan

$$ d_1(\mathbf{x},\mathbf{y}) = \sum_{j=1}^{D} |x_j-y_j| $$

### Khi \(p=2\): Euclidean

$$ d_2(\mathbf{x},\mathbf{y}) = \sqrt{ \sum_{j=1}^{D} (x_j-y_j)^2 } $$

### Chebyshev

$$ d_{\infty}(\mathbf{x},\mathbf{y}) = \max_j |x_j-y_j| $$

---

## 2.3. Cú pháp khởi tạo

```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(
    n_neighbors=5,
    weights="uniform",
    algorithm="auto",
    leaf_size=30,
    p=2,
    metric="minkowski",
    metric_params=None,
    n_jobs=None
)
```

---

## 2.4. Các tham số của `KNeighborsClassifier`

| Tham số | Kiểu dữ liệu | Mặc định | Công dụng |
|---|---|---:|---|
| `n_neighbors` | `int` | `5` | Số hàng xóm dùng để dự đoán |
| `weights` | `"uniform"`, `"distance"`, callable | `"uniform"` | Trọng số phiếu bầu |
| `algorithm` | `"auto"`, `"ball_tree"`, `"kd_tree"`, `"brute"` | `"auto"` | Thuật toán tìm hàng xóm |
| `leaf_size` | `int` | `30` | Kích thước lá của KDTree/BallTree |
| `p` | `float` | `2` | Bậc của khoảng cách Minkowski |
| `metric` | `str` hoặc callable | `"minkowski"` | Loại khoảng cách |
| `metric_params` | `dict` hoặc `None` | `None` | Tham số bổ sung cho metric |
| `n_jobs` | `int` hoặc `None` | `None` | Số CPU dùng khi truy vấn |

### `n_neighbors`

```python
KNeighborsClassifier(n_neighbors=5)
```

- \(K\) quá nhỏ: nhạy với nhiễu;
- \(K\) quá lớn: ranh giới quyết định có thể quá mượt.

### `weights`

#### `"uniform"`

Mỗi hàng xóm có một phiếu bằng nhau.

#### `"distance"`

Hàng xóm gần hơn có ảnh hưởng lớn hơn.

#### Callable

```python
def custom_weights(distances):
    return 1 / (distances + 1e-8)

knn = KNeighborsClassifier(weights=custom_weights)
```

### `algorithm`

| Giá trị | Ý nghĩa |
|---|---|
| `"auto"` | Tự chọn thuật toán phù hợp |
| `"ball_tree"` | Dùng BallTree |
| `"kd_tree"` | Dùng KDTree |
| `"brute"` | So sánh trực tiếp với toàn bộ dữ liệu |

### `leaf_size`

Ảnh hưởng đến tốc độ xây cây, tốc độ truy vấn và bộ nhớ.

### `p`

Chỉ có ý nghĩa trực tiếp khi dùng metric Minkowski:

- `p=1`: Manhattan;
- `p=2`: Euclidean.

### `metric`

Một số metric thường gặp:

```python
metric="euclidean"
metric="manhattan"
metric="minkowski"
metric="chebyshev"
metric="precomputed"
```

### `metric_params`

Truyền tham số bổ sung cho metric dưới dạng dictionary.

### `n_jobs`

```python
KNeighborsClassifier(n_jobs=-1)
```

`-1` nghĩa là sử dụng tất cả CPU khả dụng cho các thao tác hỗ trợ song song.

---

## 2.5. Các method của `KNeighborsClassifier`

### `fit(X, y)`

Lưu dữ liệu huấn luyện, nhãn và xây cấu trúc tìm kiếm nếu cần.

```python
knn.fit(X_train, y_train)
```

---

### `predict(X)`

Dự đoán nhãn cho dữ liệu mới.

```python
y_pred = knn.predict(X_test)
```

---

### `predict_proba(X)`

Trả về xác suất ước lượng cho từng lớp.

```python
probabilities = knn.predict_proba(X_test)
```

Shape:

```text
(n_samples, n_classes)
```

---

### `score(X, y, sample_weight=None)`

Trả về mean accuracy:

$$ \operatorname{Accuracy} = \frac{\text{số mẫu dự đoán đúng}} {\text{tổng số mẫu}} $$

```python
accuracy = knn.score(X_test, y_test)
```

---

### `kneighbors(X=None, n_neighbors=None, return_distance=True)`

Tìm hàng xóm gần nhất.

```python
distances, indices = knn.kneighbors(
    X_test,
    n_neighbors=3,
    return_distance=True
)
```

Kết quả:

- `distances`: khoảng cách đến hàng xóm;
- `indices`: vị trí hàng xóm trong tập dữ liệu dùng để fit.

---

### `kneighbors_graph(X=None, n_neighbors=None, mode="connectivity")`

Tạo sparse graph của các hàng xóm.

```python
graph = knn.kneighbors_graph(
    X_test,
    n_neighbors=3,
    mode="distance"
)
```

`mode`:

- `"connectivity"`: cạnh có giá trị 0 hoặc 1;
- `"distance"`: cạnh chứa khoảng cách.

---

### `get_params(deep=True)`

```python
params = knn.get_params()
```

---

### `set_params(**params)`

```python
knn.set_params(
    n_neighbors=7,
    weights="distance"
)
```

---

## 2.6. Các attribute của `KNeighborsClassifier`

| Attribute | Công dụng |
|---|---|
| `classes_` | Các lớp xuất hiện trong dữ liệu huấn luyện |
| `effective_metric_` | Metric thực tế được sử dụng |
| `effective_metric_params_` | Các tham số metric thực tế |
| `n_features_in_` | Số feature đầu vào |
| `feature_names_in_` | Tên feature nếu đầu vào có tên cột |
| `n_samples_fit_` | Số mẫu đã dùng khi fit |
| `outputs_2d_` | Cho biết target có dạng nhiều output hay không |

Ví dụ:

```python
knn.fit(X_train, y_train)

print(knn.classes_)
print(knn.effective_metric_)
print(knn.effective_metric_params_)
print(knn.n_features_in_)
print(knn.n_samples_fit_)
```

---

## 2.7. Vì sao KNN cần scaling?

Giả sử hai feature có độ lớn rất khác nhau:

```text
Tuổi:       18 → 60
Thu nhập:   5,000,000 → 100,000,000
```

Feature có thang đo lớn sẽ chi phối khoảng cách.

Dùng `StandardScaler`:

$$ z = \frac{x-\mu}{\sigma} $$

Ví dụ đúng quy trình:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Không được gọi `fit_transform()` riêng trên tập test vì sẽ gây data leakage hoặc làm train/test dùng hai hệ quy chiếu khác nhau.

---

# 3. KD-Tree và Ball Tree

KDTree và BallTree là các cấu trúc dữ liệu hỗ trợ truy vấn hàng xóm gần nhất.

> Không nên khẳng định độ phức tạp luôn là \(O(\log N)\). Hiệu quả phụ thuộc số chiều, phân bố dữ liệu, metric và số hàng xóm cần tìm. Trong không gian chiều cao, cấu trúc cây có thể giảm hiệu quả và gần với brute force.

---

## 3.1. KDTree

KDTree chia không gian bằng các siêu phẳng vuông góc với các trục tọa độ.

Phù hợp khi:

- dữ liệu dense;
- số chiều thấp hoặc trung bình;
- metric được KDTree hỗ trợ;
- cần truy vấn nhiều lần.

### Cú pháp

```python
from sklearn.neighbors import KDTree

tree = KDTree(
    X,
    leaf_size=40,
    metric="minkowski",
    **metric_kwargs
)
```

### Tham số

| Tham số | Công dụng |
|---|---|
| `X` | Dữ liệu dùng để xây cây |
| `leaf_size` | Số điểm gần đúng tại các node lá |
| `metric` | Metric khoảng cách |
| `**metric_kwargs` | Tham số bổ sung của metric |

Xem metric được hỗ trợ:

```python
print(KDTree.valid_metrics)
```

---

## 3.2. BallTree

BallTree tổ chức dữ liệu bằng các vùng hình cầu trong không gian nhiều chiều.

Thường hữu ích khi:

- metric không được KDTree hỗ trợ;
- cần Haversine cho tọa độ địa lý;
- dữ liệu có cấu trúc phù hợp với phân vùng hình cầu.

### Cú pháp

```python
from sklearn.neighbors import BallTree

tree = BallTree(
    X,
    leaf_size=40,
    metric="minkowski",
    **metric_kwargs
)
```

Xem metric được hỗ trợ:

```python
print(BallTree.valid_metrics)
```

---

## 3.3. Các method chung của KDTree và BallTree

### `query(X, k=1, return_distance=True, dualtree=False, breadth_first=False, sort_results=True)`

Tìm \(k\) hàng xóm gần nhất.

```python
distances, indices = tree.query(
    X_query,
    k=3,
    return_distance=True
)
```

| Tham số | Công dụng |
|---|---|
| `X` | Các điểm cần truy vấn |
| `k` | Số hàng xóm |
| `return_distance` | Có trả khoảng cách hay không |
| `dualtree` | Xây thêm cây cho tập truy vấn |
| `breadth_first` | Duyệt theo chiều rộng thay vì chiều sâu |
| `sort_results` | Sắp xếp kết quả theo khoảng cách |

---

### `query_radius(X, r, return_distance=False, count_only=False, sort_results=False)`

Tìm tất cả điểm nằm trong bán kính \(r\).

```python
indices = tree.query_radius(
    X_query,
    r=1.5
)
```

| Tham số | Công dụng |
|---|---|
| `X` | Điểm truy vấn |
| `r` | Bán kính |
| `return_distance` | Có trả khoảng cách hay không |
| `count_only` | Chỉ trả số lượng hàng xóm |
| `sort_results` | Sắp xếp theo khoảng cách |

---

### `kernel_density(X, h, kernel="gaussian", atol=0, rtol=1e-8, breadth_first=True, return_log=False)`

Ước lượng mật độ kernel.

Công thức KDE tổng quát:

$$ \widehat{f}(\mathbf{x}) = \frac{1}{N h^D} \sum_{i=1}^{N} K\left( \frac{\mathbf{x}-\mathbf{x}_i}{h} \right) $$

Trong đó:

- \(N\): số mẫu;
- \(D\): số chiều;
- \(h\): bandwidth;
- \(K\): kernel.

Ví dụ:

```python
density = tree.kernel_density(
    X_query,
    h=0.5,
    kernel="gaussian"
)
```

---

### `two_point_correlation(X, r, dualtree=False)`

Đếm số cặp điểm có khoảng cách nhỏ hơn hoặc bằng từng giá trị bán kính trong `r`.

```python
counts = tree.two_point_correlation(
    X_query,
    r=[0.5, 1.0, 2.0]
)
```

---

### `get_arrays()`

Trả về các mảng nội bộ dùng để biểu diễn cây.

```python
arrays = tree.get_arrays()
```

---

### `get_n_calls()`

Trả về số lần tính khoảng cách.

```python
calls = tree.get_n_calls()
```

---

### `reset_n_calls()`

Đặt lại bộ đếm số lần tính khoảng cách.

```python
tree.reset_n_calls()
```

---

### `get_tree_stats()`

Trả về:

```text
(number_of_trims, number_of_leaves, number_of_splits)
```

```python
stats = tree.get_tree_stats()
```

---

## 3.4. Các attribute của KDTree và BallTree

| Attribute | Công dụng |
|---|---|
| `data` | Dữ liệu được dùng để xây cây |
| `valid_metrics` | Danh sách các metric được hỗ trợ |

Ví dụ:

```python
print(tree.data)
print(tree.valid_metrics)
```

---

## 3.5. Ví dụ KDTree

```python
import numpy as np
from sklearn.neighbors import KDTree

X = np.array([
    [1.0, 2.0],
    [1.5, 1.8],
    [8.0, 8.0],
    [9.0, 8.5]
])

tree = KDTree(
    X,
    leaf_size=2,
    metric="euclidean"
)

distances, indices = tree.query(
    [[1.2, 2.1]],
    k=2
)

print("Khoảng cách:", distances)
print("Chỉ số:", indices)
```

---

## 3.6. Ví dụ BallTree với Haversine

Khoảng cách Haversine trên mặt cầu:

$$ a = \sin^2\left(\frac{\Delta\varphi}{2}\right) + \cos(\varphi_1)\cos(\varphi_2) \sin^2\left(\frac{\Delta\lambda}{2}\right) $$

$$ c = 2\arctan2\left(\sqrt{a},\sqrt{1-a}\right) $$

$$ d = R c $$

Trong đó:

- \(\varphi\): vĩ độ theo radian;
- \(\lambda\): kinh độ theo radian;
- \(R\): bán kính Trái Đất.

```python
import numpy as np
from sklearn.neighbors import BallTree

locations_deg = np.array([
    [16.0471, 108.2068],  # Đà Nẵng
    [21.0285, 105.8542],  # Hà Nội
    [10.8231, 106.6297]   # TP.HCM
])

locations_rad = np.radians(locations_deg)

tree = BallTree(
    locations_rad,
    metric="haversine"
)

distance_rad, indices = tree.query(
    locations_rad[:1],
    k=3
)

earth_radius_km = 6371.0
distance_km = distance_rad * earth_radius_km

print(indices)
print(distance_km)
```

---

## 3.7. So sánh nhanh

| Tiêu chí | Brute Force | KDTree | BallTree |
|---|---|---|---|
| Cách hoạt động | So sánh trực tiếp | Chia theo trục | Chia theo vùng cầu |
| Số chiều thấp | Tốt với tập nhỏ | Thường hiệu quả | Có thể hiệu quả |
| Số chiều cao | Có thể cạnh tranh | Dễ giảm hiệu quả | Cũng có thể giảm hiệu quả |
| Metric | Rộng | Hạn chế hơn | Rộng hơn |
| Sparse data | Thường dùng brute | Không phù hợp | Không phù hợp |
| Chi phí xây cấu trúc | Thấp | Có | Có |

---

# 4. Classification Metrics

## 4.1. Confusion Matrix

Với bài toán phân loại nhị phân:

| | Dự đoán Positive | Dự đoán Negative |
|---|---:|---:|
| Thực tế Positive | TP | FN |
| Thực tế Negative | FP | TN |

- **TP — True Positive**: dự đoán Positive và đúng;
- **TN — True Negative**: dự đoán Negative và đúng;
- **FP — False Positive**: dự đoán Positive nhưng sai;
- **FN — False Negative**: dự đoán Negative nhưng sai.

---

## 4.2. Accuracy

### Công thức

$$ \operatorname{Accuracy} = \frac{TP+TN}{TP+TN+FP+FN} $$

Với bài toán đa lớp:

$$ \operatorname{Accuracy} = \frac{ \sum_{i=1}^{N} \mathbb{1}(y_i=\widehat{y}_i) }{N} $$

### Hàm

```python
from sklearn.metrics import accuracy_score

accuracy_score(
    y_true,
    y_pred,
    normalize=True,
    sample_weight=None
)
```

### Tham số

| Tham số | Công dụng |
|---|---|
| `y_true` | Nhãn thật |
| `y_pred` | Nhãn dự đoán |
| `normalize=True` | Trả về tỉ lệ |
| `normalize=False` | Trả về số mẫu đúng |
| `sample_weight` | Trọng số từng mẫu |

### Ví dụ

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
```

---

## 4.3. Precision

### Công thức

$$ \operatorname{Precision} = \frac{TP}{TP+FP} $$

Precision trả lời:

> Trong các mẫu model dự đoán là Positive, bao nhiêu mẫu thực sự Positive?

### Hàm

```python
from sklearn.metrics import precision_score

precision_score(
    y_true,
    y_pred,
    labels=None,
    pos_label=1,
    average="binary",
    sample_weight=None,
    zero_division="warn"
)
```

### Tham số

| Tham số | Công dụng |
|---|---|
| `y_true` | Nhãn thật |
| `y_pred` | Nhãn dự đoán |
| `labels` | Các lớp cần tính |
| `pos_label` | Lớp được coi là Positive |
| `average` | Cách tổng hợp metric đa lớp |
| `sample_weight` | Trọng số mẫu |
| `zero_division` | Cách xử lý khi mẫu số bằng 0 |

---

## 4.4. Recall

### Công thức

$$ \operatorname{Recall} = \frac{TP}{TP+FN} $$

Recall trả lời:

> Trong các mẫu thực sự Positive, model tìm được bao nhiêu mẫu?

### Hàm

```python
from sklearn.metrics import recall_score

recall_score(
    y_true,
    y_pred,
    labels=None,
    pos_label=1,
    average="binary",
    sample_weight=None,
    zero_division="warn"
)
```

Các tham số có ý nghĩa tương tự `precision_score`.

---

## 4.5. Specificity

Specificity đo khả năng nhận diện đúng lớp Negative:

$$ \operatorname{Specificity} = \frac{TN}{TN+FP} $$

Scikit-Learn không có hàm `specificity_score` độc lập phổ biến, nhưng trong bài toán nhị phân có thể tính bằng recall của lớp Negative:

```python
from sklearn.metrics import recall_score

specificity = recall_score(
    y_true,
    y_pred,
    pos_label=0
)
```

---

## 4.6. F1-Score

### Công thức

$$ \operatorname{F1} = 2 \cdot \frac{ \operatorname{Precision} \cdot \operatorname{Recall} }{ \operatorname{Precision} + \operatorname{Recall} } $$

Công thức tương đương:

$$ \operatorname{F1} = \frac{2TP}{2TP+FP+FN} $$

### Hàm

```python
from sklearn.metrics import f1_score

f1_score(
    y_true,
    y_pred,
    labels=None,
    pos_label=1,
    average="binary",
    sample_weight=None,
    zero_division="warn"
)
```

F1 hữu ích khi cần cân bằng Precision và Recall.

---

## 4.7. F-Beta Score

F-Beta tổng quát hóa F1:

$$ F_{\beta} = (1+\beta^2) \frac{ \operatorname{Precision} \cdot \operatorname{Recall} }{ \beta^2\operatorname{Precision} + \operatorname{Recall} } $$

- \(\beta=1\): F1-score;
- \(\beta>1\): ưu tiên Recall;
- \(\beta<1\): ưu tiên Precision.

```python
from sklearn.metrics import fbeta_score

score = fbeta_score(
    y_true,
    y_pred,
    beta=2,
    average="binary"
)
```

---

## 4.8. Tham số `average`

| Giá trị | Ý nghĩa |
|---|---|
| `"binary"` | Chỉ tính cho lớp `pos_label` |
| `"micro"` | Cộng toàn bộ TP, FP, FN rồi tính |
| `"macro"` | Tính từng lớp rồi lấy trung bình đều |
| `"weighted"` | Trung bình có trọng số theo số mẫu từng lớp |
| `"samples"` | Tính từng mẫu, dùng cho multilabel |
| `None` | Trả metric riêng cho từng lớp |

### Macro average

Với \(C\) lớp:

$$ \operatorname{MacroMetric} = \frac{1}{C} \sum_{c=1}^{C} \operatorname{Metric}_c $$

### Weighted average

$$ \operatorname{WeightedMetric} = \frac{ \sum_{c=1}^{C} n_c\operatorname{Metric}_c }{ \sum_{c=1}^{C}n_c } $$

Trong đó \(n_c\) là support của lớp \(c\).

---

## 4.9. Support

Support là số mẫu thật thuộc mỗi lớp:

$$ \operatorname{Support}_c = \sum_{i=1}^{N} \mathbb{1}(y_i=c) $$

Support không phải metric chất lượng, mà cho biết số lượng mẫu dùng để tính metric của lớp.

---

## 4.10. Classification Report

### Hàm

```python
from sklearn.metrics import classification_report

classification_report(
    y_true,
    y_pred,
    labels=None,
    target_names=None,
    sample_weight=None,
    digits=2,
    output_dict=False,
    zero_division="warn"
)
```

### Tham số

| Tham số | Công dụng |
|---|---|
| `y_true` | Nhãn thật |
| `y_pred` | Nhãn dự đoán |
| `labels` | Các lớp cần đưa vào báo cáo |
| `target_names` | Tên hiển thị của từng lớp |
| `sample_weight` | Trọng số mẫu |
| `digits` | Số chữ số thập phân khi in |
| `output_dict` | Trả dictionary thay vì chuỗi |
| `zero_division` | Cách xử lý chia cho 0 |

### Ví dụ dạng text

```python
from sklearn.metrics import classification_report

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Class 0", "Class 1"],
        digits=4,
        zero_division=0
    )
)
```

### Ví dụ dạng dictionary

```python
report = classification_report(
    y_test,
    y_pred,
    output_dict=True,
    zero_division=0
)

print(report["macro avg"])
```

### Chuyển sang DataFrame

```python
import pandas as pd
from sklearn.metrics import classification_report

report = classification_report(
    y_test,
    y_pred,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).T
print(report_df)
```

---

## 4.11. Các attribute của metric?

Các hàm như:

```python
accuracy_score(...)
precision_score(...)
recall_score(...)
f1_score(...)
classification_report(...)
```

là **function**, không phải class model.

Vì vậy chúng:

- không cần `fit()`;
- không có các learned attributes như `classes_`, `coef_`, `labels_`;
- trả kết quả ngay sau khi được gọi.

Ví dụ:

```python
f1 = f1_score(y_true, y_pred)
```

`f1` là một số, không phải một model.

---

# 5. Ví dụ tổng hợp

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# 1. Đọc dữ liệu
X, y = load_iris(return_X_y=True)

# 2. Chia train/test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. Tạo pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    (
        "knn",
        KNeighborsClassifier(
            n_neighbors=5,
            weights="distance",
            algorithm="auto",
            metric="minkowski",
            p=2,
            n_jobs=-1
        )
    )
])

# 4. Huấn luyện
pipeline.fit(X_train, y_train)

# 5. Dự đoán
y_pred = pipeline.predict(X_test)

# 6. Tính metric
accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

print("Accuracy:", accuracy)
print("Precision macro:", precision)
print("Recall macro:", recall)
print("F1 macro:", f1)

print("\nClassification report:")
print(
    classification_report(
        y_test,
        y_pred,
        digits=4,
        zero_division=0
    )
)
```

---

# 6. Bảng ghi nhớ nhanh

## 6.1. K-Means

| Thành phần | Ý nghĩa |
|---|---|
| `n_clusters` | Số cụm |
| `cluster_centers_` | Centroid |
| `labels_` | Nhãn cụm |
| `inertia_` | WCSS |
| `n_iter_` | Số vòng lặp |
| `fit_predict()` | Fit và trả nhãn |
| `transform()` | Khoảng cách tới từng centroid |

## 6.2. KNN

| Thành phần | Ý nghĩa |
|---|---|
| `n_neighbors` | Số hàng xóm |
| `weights` | Trọng số phiếu |
| `metric` | Loại khoảng cách |
| `p` | Bậc Minkowski |
| `predict()` | Dự đoán nhãn |
| `predict_proba()` | Xác suất từng lớp |
| `kneighbors()` | Khoảng cách và index hàng xóm |
| `classes_` | Các lớp |
| `effective_metric_` | Metric thực tế |

## 6.3. KDTree và BallTree

| Method / attribute | Ý nghĩa |
|---|---|
| `query()` | Tìm K hàng xóm |
| `query_radius()` | Tìm điểm trong bán kính |
| `kernel_density()` | Ước lượng mật độ |
| `two_point_correlation()` | Đếm cặp điểm theo bán kính |
| `get_n_calls()` | Số phép tính khoảng cách |
| `get_tree_stats()` | Thống kê cây |
| `data` | Dữ liệu xây cây |
| `valid_metrics` | Metric được hỗ trợ |

## 6.4. Classification Metrics

| Metric | Công thức |
|---|---|
| Accuracy | \(\frac{TP+TN}{TP+TN+FP+FN}\) |
| Precision | \(\frac{TP}{TP+FP}\) |
| Recall | \(\frac{TP}{TP+FN}\) |
| Specificity | \(\frac{TN}{TN+FP}\) |
| F1 | \(\frac{2TP}{2TP+FP+FN}\) |

---

# 7. Tài liệu chính thức

- KMeans: <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html>
- KNeighborsClassifier: <https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html>
- KDTree: <https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KDTree.html>
- BallTree: <https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.BallTree.html>
- Accuracy: <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html>
- Precision: <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html>
- Recall: <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html>
- F1: <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html>
- Classification report: <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html>