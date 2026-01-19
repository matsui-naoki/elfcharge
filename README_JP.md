# elfcharge

**ELFベースのエレクトライド電子カウント**

VASP計算結果（ELF, 電荷密度）からエレクトライドおよび格子間電子系の電子数を解析するPythonパッケージです。

## 参考論文

本実装はBadELFアルゴリズムに基づいています：

> Weaver, J. R., et al. "Counting Electrons in Electrides"
> *J. Am. Chem. Soc.* **2023**, 145, 26472-26476.
> DOI: [10.1021/jacs.3c11019](https://doi.org/10.1021/jacs.3c11019)

## 機能

- **ELFベースの空間分割**: 結合上のELF極小値を用いた空間分割
- **エレクトライド検出**: ELF極大値から格子間電子サイトを特定
- **電荷積算**: 各原子・エレクトライドサイトの電子数と酸化数を計算
- **構造出力**: エレクトライドサイト付き構造をCIF/POSCARで出力
- **可視化**: ELFプロファイル、断面図、電子分布のプロット

## インストール

```bash
# リポジトリをクローン
git clone https://github.com/yourusername/elfcharge.git
cd elfcharge

# pipでインストール
pip install -e .

# 可視化サポート付き
pip install -e ".[visualization]"
```

## クイックスタート

### 簡易ラッパー関数（推奨）

```python
from elfcharge import run_badelf_analysis

# 一回の関数呼び出しで完全な解析を実行
result = run_badelf_analysis(
    elfcar_path="ELFCAR",
    chgcar_path="CHGCAR_sum",
    zval={'Y': 11, 'C': 4, 'F': 7},
    oxidation_states={'Y': 3, 'C': -4, 'F': -1},
    core_radii={'Y': 0.8},
    elf_threshold=0.5,
    apply_smooth=True,
    save_dir="./badelf_results",
    save_plots=True,
    save_cif=True
)

# 結果へのアクセス
print(result.oxidation.species_avg_oxidation)  # 元素種ごとの平均酸化数
print(result.charges.atom_electrons)           # 各原子の電子数
print(result.electride_sites)                  # 検出されたエレクトライドサイト
```

### ステップバイステップAPI

```python
from elfcharge import read_elfcar, read_chgcar, ELFAnalyzer
from elfcharge import create_structure_with_electrides

# VASPファイルの読み込み（オプションでスムージング）
elf_data = read_elfcar("ELFCAR", smooth=True, smooth_size=3)
chg_data = read_chgcar("CHGCAR_sum")

# アナライザーの初期化
analyzer = ELFAnalyzer(elf_data)

# 隣接原子ペアの検索（酸化数指定でCrystalNNの精度向上）
bonds = analyzer.get_neighbor_pairs_pymatgen(
    oxidation_states={'Y': 3, 'C': -4, 'F': -1}
)

# 結合上のELF解析
for i, bond in enumerate(bonds):
    bonds[i] = analyzer.analyze_bond_elf(bond, core_radii={'Y': 0.8})

# 結合解析から原子半径を計算
atom_radii = analyzer.compute_atom_radii(bonds)

# エレクトライドサイトの検出（atom_radiiのboundary_radiusを使用）
electride_sites = analyzer.find_interstitial_electrides(
    elf_threshold=0.5,
    atom_radii=atom_radii,
    use_boundary_radius=True
)

# エレクトライドサイト付き構造の出力
structure = create_structure_with_electrides(elf_data, electride_sites)
structure.to(filename="structure_with_electrides.cif")
```

## 使用例

`examples/example_elfcharge.ipynb`に以下を含む完全なワークフローがあります：
- VASPファイルの読み込み
- エレクトライドサイトの検出
- 空間分割
- 電荷積算
- 酸化数計算
- 可視化

## アルゴリズム概要

### ワークフロー

```
入力: ELFCAR, CHGCAR
           ↓
┌─────────────────────────────────┐
│ 1. データ読み込み                │
│    - 格子ベクトル                │
│    - 原子位置                    │
│    - ELF/電荷密度グリッド        │
└─────────────────────────────────┘
           ↓
┌─────────────────────────────────┐
│ 2. 隣接原子ペアの検索            │
│    - CrystalNNによる結合検出     │
└─────────────────────────────────┘
           ↓
┌─────────────────────────────────┐
│ 3. 結合ELF解析                   │
│    - 結合上のELF極小値を検出     │
│    - ELF半径を計算               │
└─────────────────────────────────┘
           ↓
┌─────────────────────────────────┐
│ 4. エレクトライドサイト検出      │
│    - 原子境界外のELF極大値を検出 │
└─────────────────────────────────┘
           ↓
┌─────────────────────────────────┐
│ 5. 空間分割                      │
│    - 原子間: ELF極小面による     │
│      Voronoi分割                 │
│    - 原子-エレクトライド:        │
│      単純Voronoi                 │
└─────────────────────────────────┘
           ↓
┌─────────────────────────────────┐
│ 6. 電荷積算                      │
│    - 各領域のCHGCARを積算        │
│    - 酸化数を計算                │
└─────────────────────────────────┘
           ↓
出力: 電子数、酸化数
```

### 分割戦略

| 境界タイプ | 手法 | 理由 |
|-----------|------|------|
| 原子間 | ELF極小面によるVoronoi | 凸型原子領域の維持 |
| 原子-エレクトライド | 単純Voronoi | 非球形状を許容 |

### 主要な式

**電荷積算:**
```
N_electrons = Σ(CHGCAR[region]) / N_grid_total
```

**酸化数:**
```
酸化数 = ZVAL - N_electrons
```

**最小イメージ距離:**
```python
diff_frac = r_frac - atom_frac
diff_frac = diff_frac - np.round(diff_frac)  # [-0.5, 0.5]に折り返し
diff_cart = diff_frac @ lattice
distance = np.linalg.norm(diff_cart)
```

## ZVAL設定

酸化数計算にはPOTCARからの価電子数を指定します：

```python
ZVAL = {
    'Y': 11,   # PAW_PBE Y_sv
    'C': 4,    # PAW_PBE C
    'F': 7,    # PAW_PBE F
}
```

## 主要パラメータ

### run_badelf_analysis()のパラメータ

| パラメータ | 型 | 説明 |
|-----------|------|------|
| `elfcar_path` | str | ELFCARファイルへのパス |
| `chgcar_path` | str | CHGCARファイルへのパス（CHGCAR_sum推奨） |
| `zval` | Dict[str, float] | 元素種ごとの価電子数（POTCARから） |
| `oxidation_states` | Dict[str, float] | CrystalNN用の期待酸化数 |
| `core_radii` | Dict[str, float] | ELF極小検索で除外するコア半径 |
| `atom_cutoffs` | Dict[str, float] | エレクトライド検出用フォールバックカットオフ |
| `elf_threshold` | float | エレクトライドサイトの最小ELF値（デフォルト: 0.5） |
| `apply_smooth` | bool | ELFにスムージングを適用（デフォルト: False） |
| `smooth_size` | int | スムージングカーネルサイズ（デフォルト: 3） |
| `save_dir` | str | プロットとCIFの出力ディレクトリ |
| `save_plots` | bool | 可視化プロットを生成（デフォルト: True） |
| `save_cif` | bool | エレクトライド付き構造を出力（デフォルト: True） |

### グリッド解像度

BadELF論文の推奨値に基づいてグリッド解像度をチェックします：
- **16 voxels/Å**: 0.2%未満の誤差（ほとんどの解析に許容可能）
- **40 voxels/Å**: 完全収束結果

### エレクトライド検出
- `elf_threshold`: エレクトライドサイトの最小ELF値（デフォルト: 0.5）
- `atom_radii`: 格子間分類に`boundary_radius`（ELF極小までの最大距離）を使用
- `atom_cutoffs`: フォールバックの元素種固有距離カットオフ

### 結合解析
- `core_radii`: ELF極小検索時に除外する原子コア半径
  - 重元素（例：La, Y）ではコア領域のELFが低いため重要
- `oxidation_states`: CrystalNNの隣接検出精度を向上

## 可視化

```python
from elfcharge.visualize import plot_elf_slice_with_partition

# 分割境界付きELF断面図
fig = plot_elf_slice_with_partition(
    elf_data, labels,
    plane='xy', position=0.5,
    electride_sites=electride_sites,
    interpolate=True,           # 高解像度
    interpolate_factor=3,
    coord_system='cartesian',   # または 'fractional'
)
fig.savefig('elf_partition.png', dpi=150)
```

## モジュール構成

```
elfcharge/
├── __init__.py      # パッケージ初期化
├── io.py            # VASP I/O（ELFCAR, CHGCAR）
├── analysis.py      # ELF解析（結合、極小値、半径）
├── partition.py     # 空間分割（Voronoi）
├── integrate.py     # 電荷積算
├── structure.py     # エレクトライド付き構造出力
├── visualize.py     # 可視化ツール
└── core.py          # 高レベルAPI
```

## APIリファレンス

### データクラス

```python
@dataclass
class GridData:
    """3Dグリッドデータのコンテナ"""
    lattice: np.ndarray          # (3, 3) 格子ベクトル [Å]
    species: List[str]           # 元素記号
    num_atoms: List[int]         # 元素種ごとの原子数
    frac_coords: np.ndarray      # (N_atoms, 3) 分率座標
    grid: np.ndarray             # (NGX, NGY, NGZ) グリッドデータ
    ngrid: Tuple[int, int, int]  # グリッド次元

@dataclass
class BondPair:
    """原子間結合情報"""
    atom_i: int
    atom_j: int
    jimage: Tuple[int, int, int]  # 周期イメージ
    elf_minimum_frac: np.ndarray  # ELF極小位置
    elf_minimum_value: float
    distance_i: float  # 原子iから極小までの距離
    distance_j: float  # 原子jから極小までの距離

@dataclass
class ElectrideSite:
    """エレクトライドサイト情報"""
    frac_coord: np.ndarray
    cart_coord: np.ndarray
    elf_value: float
    grid_index: Tuple[int, int, int]
```

### 主要関数

```python
# 高レベルAPI（推奨）
run_badelf_analysis(
    elfcar_path, chgcar_path,
    zval=None, oxidation_states=None, core_radii=None,
    elf_threshold=0.5, apply_smooth=False, smooth_size=3,
    save_dir=None, save_plots=True, save_cif=True
) -> BadELFResult

# I/O
read_elfcar(filepath, smooth=False, smooth_size=3) -> GridData
read_chgcar(filepath) -> GridData
check_grid_resolution(data, min_voxels_per_angstrom=16.0) -> dict

# 解析
ELFAnalyzer(elf_data)
  .get_neighbor_pairs_pymatgen(oxidation_states=None) -> List[BondPair]
  .analyze_bond_elf(bond, core_radii=None) -> BondPair
  .compute_atom_radii(bonds) -> Dict[int, AtomRadii]
  .find_interstitial_electrides(
      elf_threshold, atom_cutoffs=None,
      atom_radii=None, use_boundary_radius=True
  ) -> List[ElectrideSite]

# 構造出力
create_structure_with_electrides(elf_data, electride_sites) -> Structure

# 可視化
plot_elf_slice_with_partition(elf_data, labels, plane, position, ...)
plot_elf_along_bond(elf_data, bond)
plot_radial_elf(elf_data, atom_index)
plot_electron_distribution(atom_electrons, species_list)
```

## 動作要件

- Python >= 3.8
- numpy >= 1.20
- scipy >= 1.7
- pymatgen >= 2022.0.0
- matplotlib >= 3.5（オプション、可視化用）

## 既知の制限事項

1. **ELF平面調整**: 一部の系ではELF極小平面分割が不安定な場合があります。その場合は単純Voronoi分割にフォールバックします。

2. **グリッドサイズの不一致**: ELFCARとCHGCARのグリッドサイズが異なる場合、最近傍リサンプリングが自動適用されます。

3. **共有結合性物質**: 強い共有結合を持つ物質では電子分布の精度が低下する場合があります。

## ライセンス

MIT License

## 作者

Naoki Matsui

## 開発

このリポジトリは [Claude Opus 4.5](https://www.anthropic.com/claude)（Anthropic）を使用して実装されました。

## 引用

本パッケージを使用する場合は、オリジナルのBadELF論文を引用してください：

```bibtex
@article{weaver2023counting,
  title={Counting Electrons in Electrides},
  author={Weaver, James R. and others},
  journal={J. Am. Chem. Soc.},
  volume={145},
  pages={26472--26476},
  year={2023},
  doi={10.1021/jacs.3c11019}
}
```
