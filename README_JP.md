# elfcharge

**ELFベースのエレクトライド電子カウント**

VASP計算結果（ELF, 電荷密度）からエレクトライドおよび格子間電子系の電子数を解析するPythonパッケージです。

## 参考論文

本実装はBadELFアルゴリズムを参考に作成しました：

> Weaver, J. R., et al. "Counting Electrons in Electrides"
> *J. Am. Chem. Soc.* **2023**, 145, 26472-26476.
> DOI: [10.1021/jacs.3c11019](https://doi.org/10.1021/jacs.3c11019)

## 機能

- **ELFベースの半径算出**: 結合上のELF極小値を用いたELF半径算出
- **エレクトライド検出**: ELF極大値から格子間電子サイトを特定
- **Voronoi分割**: 単純Voronoiによる電子密度分布の空間分割
- **電荷積算**: 各原子・エレクトライドサイトの電子数と酸化数を計算
- **構造出力**: エレクトライドサイト付き構造をCIF/POSCARで出力

## インストール

```bash
# リポジトリをクローン
git clone https://github.com/yourusername/elfcharge.git
cd elfcharge

# pipでインストール
pip install -e .
```

## クイックスタート

```python
from elfcharge import run_badelf_analysis

# 完全な解析を実行
# 注意: CHGCAR（価電子のみ）を使用、CHGCAR_sumではない
result = run_badelf_analysis(
    elfcar_path="ELFCAR",
    chgcar_path="CHGCAR",  # 価電子のみ！
    oxidation_states={'Y': 3, 'C': -4},
    core_radii={'Y': 0.8, 'C': 0.3},
    elf_threshold=0.5,
    save_dir="./badelf_results"
)

# 結果へのアクセス
print(result.oxidation.species_avg_oxidation)
```

## アルゴリズム概要

```
1. ELFCAR, CHGCARの読み込み
2. 隣接原子ペアの検索（CrystalNN）
3. 結合上のELF解析 → ELF極小値を検出 → elf_radiiを計算
4. エレクトライドサイトの検出（原子境界外のELF極大値）
5. 空間分割（単純Voronoi）
6. 各領域の電荷積算
7. 酸化数の計算（ZVAL - 電子数）
```

### 分割戦略

| 境界 | 手法 |
|------|------|
| 原子間 | 単純Voronoi |
| 原子-エレクトライド | 単純Voronoi |
| エレクトライド領域 | 単一領域にマージ |

## 主要パラメータ

| パラメータ | 型 | 説明 |
|-----------|------|------|
| `elfcar_path` | str | ELFCARファイルへのパス |
| `chgcar_path` | str | CHGCARファイルへのパス（価電子のみ） |
| `zval` | Dict[str, float] | 元素種ごとの価電子数（オプション、DEFAULT_ZVALを使用） |
| `oxidation_states` | Dict[str, float] | CrystalNN精度向上用の期待酸化数 |
| `core_radii` | Dict[str, float] | ELF極小検索で除外するコア半径 |
| `elf_threshold` | float | エレクトライドサイトの最小ELF値（デフォルト: 0.5） |
| `apply_smooth` | bool | ELFにスムージングを適用（デフォルト: False） |
| `save_dir` | str | プロットとCIFの出力ディレクトリ |

### パラメータガイド

- **core_radii**: 結合上のELF極小検索時に使用。ELFが低い原子コア領域を除外。重元素（例：Y, La）で重要。
- **elf_threshold**: エレクトライドサイト検出の最小ELF値（デフォルト: 0.5）

## ZVAL設定

[Materials Project推奨POTCAR](https://docs.materialsproject.org/methodology/materials-methodology/calculation-details/gga+u-calculations/pseudopotentials)に基づく**DEFAULT_ZVAL**が組み込まれています。

```python
from elfcharge import DEFAULT_ZVAL

# デフォルト値の確認
print(DEFAULT_ZVAL['Y'])   # 11 (Y_sv)
print(DEFAULT_ZVAL['Mg'])  # 8 (Mg_pv)

# 特定のPOTCAR用に上書き
result = run_badelf_analysis(
    elfcar_path="ELFCAR",
    chgcar_path="CHGCAR",
    zval={'Li': 1},  # Li_svではなくLi
)
```

## モジュール構成

```
elfcharge/
├── __init__.py      # 公開エクスポート
├── io.py            # read_elfcar, read_chgcar
├── analysis.py      # ELFAnalyzer, ELFRadii, BondPair, ElectrideSite
├── partition.py     # VoronoiPartitioner, BadELFPartitioner
├── integrate.py     # ChargeIntegrator, OxidationCalculator, DEFAULT_ZVAL
├── structure.py     # create_structure_with_electrides, BadELFResult
├── visualize.py     # プロット関数（オプション）
└── core.py          # run_badelf_analysis, BadELFAnalyzer
```

## 動作要件

- Python >= 3.8
- numpy >= 1.20
- scipy >= 1.7
- pymatgen >= 2022.0.0
- matplotlib >= 3.5（オプション、可視化用）

## 制限事項

1. **グリッドサイズの不一致**: ELFCARとCHGCARのグリッドサイズが異なる場合、最近傍リサンプリングが自動適用されます。

2. **エレクトライド領域のマージ**: 検出されたすべてのエレクトライドサイトは電荷積算のために単一領域にマージされます。

## ライセンス

MIT License

## 作者

Naoki Matsui

## 開発

このリポジトリは [Claude Opus 4.5](https://www.anthropic.com/claude)（Anthropic）を使用して実装されました。

## 引用

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
