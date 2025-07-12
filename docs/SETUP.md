# SageMath MCP セットアップガイド

## 🚀 快速セットアップ

### mybinder.orgの場合

1. **即座開始**: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/jxta/jupyter-sage-mcp/main)
2. **環境確認**: 自動でSageMathとPythonカーネルが利用可能
3. **デモ実行**: `notebooks/sage_mcp_demo.ipynb`を開いて実行

### NII解析基盤の場合

#### 1. 環境準備
```bash
# SSH接続
ssh username@nii-login.jp

# モジュール読み込み
module load miniconda3

# リポジトリクローン
git clone https://github.com/jxta/jupyter-sage-mcp.git
cd jupyter-sage-mcp

# 環境構築
conda env create -f environment.yml
conda activate jupyter-mcp-sage
```

#### 2. JupyterLab起動
```bash
# ポートフォワーディング用
jupyter lab --port 8888 --no-browser \
    --ServerApp.token='nii_mcp_token' \
    --ip=0.0.0.0 \
    --ServerApp.allow_remote_access=True
```

#### 3. ローカル接続
```bash
# ローカル端末で実行（NIIへのSSH接続）
ssh -L 8888:localhost:8888 username@nii-login.jp
```

ブラウザで http://localhost:8888 にアクセス

## 🔧 詳細設定

### SageMathカーネル設定

```bash
# カーネル登録
sage -python -m sage_setup.jupyter.install --user

# 確認
jupyter kernelspec list

# 出力例:
# Available kernels:
#   python3    /path/to/python3
#   sagemath   /path/to/sagemath
```

### Claude Desktop統合

#### 設定ファイル作成
```bash
# macOS
nano ~/.config/Claude/claude_desktop_config.json

# Windows
notepad %APPDATA%\Claude\claude_desktop_config.json
```

#### 設定内容
```json
{
  "mcpServers": {
    "jupyter-sage-binder": {
      "command": "docker",
      "args": [
        "run", "-i", "--rm",
        "-e", "SERVER_URL=http://localhost:8888",
        "-e", "TOKEN=binder_mcp_token",
        "-e", "NOTEBOOK_PATH=notebooks/sage_mcp_demo.ipynb",
        "--network=host",
        "datalayer/jupyter-mcp-server:latest"
      ]
    },
    "sage-mcp-direct": {
      "command": "python",
      "args": ["src/mcp_sage_helper.py"],
      "env": {
        "JUPYTER_URL": "http://localhost:8888",
        "JUPYTER_TOKEN": "your_token_here"
      }
    }
  }
}
```

## 🧪 テストと検証

### 基本機能テスト

```python
# 1. MCP初期化テスト
from src.mcp_sage_helper import setup_sage_mcp
mcp = setup_sage_mcp()
print("✅ MCP初期化完了")

# 2. SageMath テスト
sage_result = mcp.execute_code("2+2", "sagemath")
print("SageMath結果:", sage_result)

# 3. Python テスト  
python_result = mcp.execute_code("import numpy; print(numpy.__version__)", "python")
print("Python結果:", python_result)

# 4. カーネル切り替えテスト
switch_result = mcp.switch_kernel_context("sagemath")
print("切り替え結果:", switch_result)
```

### 高度な機能テスト

```python
# 数論テスト
number_theory_test = """
# 素数テスト
primes_list = [p for p in range(2, 100) if is_prime(p)]
print("100未満の素数:", len(primes_list))

# 暗号学テスト
p = next_prime(100)
F = GF(p)
print(f"有限体 GF({p}) の要素数:", F.order())
"""

result = mcp.execute_code(number_theory_test, "sagemath")
print("数論テスト結果:", result)
```

## 🔍 トラブルシューティング

### よくあるエラーと解決法

#### 1. カーネルが見つからない
```bash
# 解決法
sage -python -m sage_setup.jupyter.install --user --force
jupyter kernelspec remove sagemath  # 既存削除
sage -python -m sage_setup.jupyter.install --user
```

#### 2. MCP接続失敗
```python
# デバッグ情報の取得
import logging
logging.basicConfig(level=logging.DEBUG)

mcp = setup_sage_mcp()
print("システム情報:", mcp.get_system_info())
```

#### 3. ポート競合
```bash
# 使用中ポートの確認
netstat -an | grep 8888

# 別ポートでの起動
jupyter lab --port 8889
```

#### 4. パッケージエラー
```bash
# パッケージの再インストール
pip install --force-reinstall datalayer_pycrdt
conda install -c conda-forge sagemath --force-reinstall
```

## 🎯 使用例とワークフロー

### 数学研究ワークフロー

```python
# 1. 問題設定（SageMath）
problem_setup = """
# 楕円曲線の研究
E = EllipticCurve([0, 1, 1, -2, 0])
print("楕円曲線:", E)
print("判別式:", E.discriminant())
print("j-invariant:", E.j_invariant())
"""

# 2. 数値計算（Python）
numerical_analysis = """
import numpy as np
import matplotlib.pyplot as plt

# 楕円曲線のプロット（近似）
x = np.linspace(-2, 2, 1000)
# y² = x³ + x² - 2x の近似プロット
y_squared = x**3 + x**2 - 2*x
valid_indices = y_squared >= 0
x_valid = x[valid_indices]
y_pos = np.sqrt(y_squared[valid_indices])
y_neg = -y_pos

plt.figure(figsize=(10, 6))
plt.plot(x_valid, y_pos, 'b-', label='y > 0')
plt.plot(x_valid, y_neg, 'b-', label='y < 0')
plt.grid(True)
plt.legend()
plt.title('Elliptic Curve y² = x³ + x² - 2x')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
"""

# 3. 実行
mcp.switch_kernel_context("sagemath")
sage_result = mcp.execute_code(problem_setup)

mcp.switch_kernel_context("python")
python_result = mcp.execute_code(numerical_analysis)
```

### データサイエンスワークフロー

```python
# 1. データ前処理（Python）
data_prep = """
import pandas as pd
import numpy as np

# サンプルデータ作成
np.random.seed(42)
data = pd.DataFrame({
    'x': np.random.randn(1000),
    'y': np.random.randn(1000),
    'category': np.random.choice(['A', 'B', 'C'], 1000)
})

print("データ概要:")
print(data.describe())
"""

# 2. 統計解析（Python + SageMath）
statistical_analysis = """
from scipy import stats
import numpy as np

# データの生成
np.random.seed(42)
group_a = np.random.normal(0, 1, 500)
group_b = np.random.normal(0.2, 1, 500)

# t検定
t_stat, p_value = stats.ttest_ind(group_a, group_b)
print(f"t統計量: {t_stat:.4f}")
print(f"p値: {p_value:.4f}")

# 効果量の計算
cohen_d = (np.mean(group_a) - np.mean(group_b)) / np.sqrt((np.var(group_a) + np.var(group_b)) / 2)
print(f"Cohen's d: {cohen_d:.4f}")
"""

# 実行
mcp.execute_code(data_prep, "python")
mcp.execute_code(statistical_analysis, "python")
```

## 🌐 外部連携

### GitHub統合

```bash
# Git設定
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# 変更のコミット
git add notebooks/
git commit -m "Add new analysis notebook"
git push origin main
```

### 外部MCP接続例

```python
# 外部MCPサーバーとの連携
async def connect_external_math_server():
    # 数学専用MCPサーバーへの接続
    connection = await mcp.connect_external_mcp(
        "wss://math-api.example.com/mcp",
        auth_token="your_api_token"
    )
    
    if connection["status"] == "success":
        print("✅ 外部数学MCPサーバーに接続")
        
        # 外部ツールの使用例
        wolframalpha_result = await mcp.call_external_tool(
            connection["connection_id"],
            "wolfram_query",
            {"query": "integrate x^2 dx"}
        )
        
        return wolframalpha_result
    else:
        print("❌ 外部MCP接続失敗")
        return None

# 実行例
# result = await connect_external_math_server()
```

## 📊 性能最適化

### メモリ効率化

```python
# 大規模計算用の設定
performance_config = """
# メモリ使用量の監視
import psutil
import gc

def monitor_memory():
    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"メモリ使用量: {memory_info.rss / 1024 / 1024:.2f} MB")
    
# ガベージコレクション
def cleanup_memory():
    gc.collect()
    print("メモリクリーンアップ完了")

# 使用例
monitor_memory()
# 大規模計算実行
cleanup_memory()
monitor_memory()
"""
```

### 並列処理

```python
# SageMathでの並列計算
parallel_sage = """
# 並列因数分解の例
def parallel_factorization(numbers):
    import multiprocessing as mp
    
    def factor_number(n):
        return (n, factor(n))
    
    # プロセスプールで並列実行
    with mp.Pool() as pool:
        results = pool.map(factor_number, numbers)
    
    return dict(results)

# 使用例
test_numbers = [2^31-1, 2^32-1, 2^33-1]
results = parallel_factorization(test_numbers)
for num, factors in results.items():
    print(f"{num}: {factors}")
"""
```

## 🔒 セキュリティ設定

### 安全なMCP設定

```python
# セキュリティ強化設定
security_config = {
    "execution_timeout": 30,  # 実行時間制限（秒）
    "memory_limit": "1GB",    # メモリ制限
    "allowed_imports": [      # 許可されたモジュール
        "numpy", "pandas", "matplotlib", 
        "scipy", "sympy", "sage"
    ],
    "blocked_functions": [    # 禁止された関数
        "exec", "eval", "open", "file"
    ],
    "sandbox_mode": True,     # サンドボックス有効
    "log_execution": True     # 実行ログ記録
}

# セキュリティチェック関数
def security_check(code):
    blocked_keywords = ['import os', 'import sys', '__import__']
    for keyword in blocked_keywords:
        if keyword in code:
            return False, f"Blocked keyword detected: {keyword}"
    return True, "Code is safe"
```

## 📈 監視とログ

### 実行ログの設定

```python
# ログ設定の詳細化
import logging
from datetime import datetime

def setup_detailed_logging():
    # ログフォーマットの設定
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(f'mcp_log_{datetime.now().strftime("%Y%m%d")}.log'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger('SageMCP')
    return logger

# 使用例
logger = setup_detailed_logging()
logger.info("MCP session started")
```

## 🔧 カスタマイズ

### 独自ツールの追加

```python
# カスタムMCPツールの実装
class CustomMathTools:
    def __init__(self, mcp_client):
        self.mcp = mcp_client
    
    def advanced_prime_test(self, n, iterations=50):
        """高度な素数判定"""
        code = f"""
        # Miller-Rabin素数判定
        def miller_rabin_test(n, k={iterations}):
            if n < 2: return False
            if n == 2 or n == 3: return True
            if n % 2 == 0: return False
            
            # n-1 = 2^r * d の形に分解
            r = 0
            d = n - 1
            while d % 2 == 0:
                r += 1
                d //= 2
            
            # k回テスト
            for _ in range(k):
                a = randint(2, n-2)
                x = power_mod(a, d, n)
                if x == 1 or x == n-1:
                    continue
                for _ in range(r-1):
                    x = power_mod(x, 2, n)
                    if x == n-1:
                        break
                else:
                    return False
            return True
        
        result = miller_rabin_test({n})
        print(f"{n} is {'probably prime' if result else 'composite'}")
        """
        
        return self.mcp.execute_code(code, "sagemath")
    
    def elliptic_curve_analysis(self, a, b):
        """楕円曲線の詳細解析"""
        code = f"""
        # 楕円曲線 y² = x³ + {a}x + {b} の解析
        E = EllipticCurve([{a}, {b}])
        
        print("楕円曲線:", E)
        print("判別式:", E.discriminant())
        print("j-invariant:", E.j_invariant())
        print("導手:", E.conductor())
        
        # 有理点の探索
        rational_points = E.rational_points(bound=10)
        print(f"有理点 (bound=10): {len(rational_points)} 個")
        for i, point in enumerate(rational_points[:5]):
            print(f"  {i+1}: {point}")
        """
        
        return self.mcp.execute_code(code, "sagemath")

# 使用例
custom_tools = CustomMathTools(mcp)
prime_result = custom_tools.advanced_prime_test(982451653)
curve_result = custom_tools.elliptic_curve_analysis(1, 1)
```

## 📋 ベストプラクティス

### 効率的な開発フロー

1. **段階的開発**
   ```python
   # 1. 小さなテストから開始
   test_code = "print('Hello, SageMath!')"
   result = mcp.execute_code(test_code, "sagemath")
   
   # 2. 徐々に複雑化
   complex_code = """
   var('x')
   f = x^3 - 2*x + 1
   roots = solve(f == 0, x)
   print("Roots:", roots)
   """
   result = mcp.execute_code(complex_code, "sagemath")
   ```

2. **エラーハンドリング**
   ```python
   def safe_execute(code, kernel="sagemath"):
       try:
           result = mcp.execute_code(code, kernel)
           if result["status"] == "error":
               print(f"実行エラー: {result['error']}")
               return None
           return result
       except Exception as e:
           print(f"予期しないエラー: {e}")
           return None
   ```

3. **結果の保存と共有**
   ```python
   # 結果をJSONファイルに保存
   import json
   
   def save_results(results, filename):
       with open(filename, 'w') as f:
           json.dump(results, f, indent=2, default=str)
       print(f"結果を {filename} に保存しました")
   
   # 使用例
   calculation_results = []
   for i in range(10):
       result = mcp.execute_code(f"factor({2**i - 1})", "sagemath")
       calculation_results.append(result)
   
   save_results(calculation_results, "factorization_results.json")
   ```

## 🎓 学習リソース

### 推奨学習パス

1. **SageMath基礎**
   - [SageMath Tutorial](https://doc.sagemath.org/html/en/tutorial/)
   - [SageMath Reference Manual](https://doc.sagemath.org/html/en/reference/)

2. **MCP理解**
   - [Model Context Protocol Documentation](https://modelcontextprotocol.io/)
   - [MCP Server Examples](https://github.com/modelcontextprotocol/servers)

3. **実践例**
   - `notebooks/sage_mcp_demo.ipynb`
   - [SageMath Examples](https://wiki.sagemath.org/Examples)

### コミュニティリソース

- **SageMath Community**: https://groups.google.com/g/sage-support
- **Jupyter Community**: https://discourse.jupyter.org/
- **MCP Community**: https://github.com/modelcontextprotocol/

## 🚀 次のステップ

リポジトリのセットアップが完了したら：

1. **即座にテスト**: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/jxta/jupyter-sage-mcp/main)
2. **デモノートブック実行**: `notebooks/sage_mcp_demo.ipynb`
3. **Claude Desktop連携**: `config/claude_config.json`の設定
4. **独自プロジェクト開始**: 新しいノートブックを作成

---

**質問やサポートが必要な場合は、GitHubのIssueでお気軽にお声がけください！**
