#!/bin/bash

echo "=========================================="
echo "Testing ASR Baseline Setup"
echo "=========================================="
echo ""

# Test 1: Check Python dependencies
echo "[1/5] Checking Python dependencies..."
python3 << 'EOF'
try:
    import torch
    import torchaudio
    import transformers
    import datasets
    import jiwer
    print("✓ All Python dependencies installed")
except ImportError as e:
    print(f"✗ Missing dependency: {e}")
    exit(1)
EOF

if [ $? -ne 0 ]; then
    echo "Please run: bash setup_environment.sh"
    exit 1
fi

# Test 2: Check model implementations
echo ""
echo "[2/5] Testing model implementations..."
python3 << 'EOF'
import sys
sys.path.append('.')

try:
    from models.attention_variants import get_attention_module
    from models.conformer import Conformer
    from models.branchformer import Branchformer
    print("✓ Model implementations loaded successfully")
except Exception as e:
    print(f"✗ Error loading models: {e}")
    exit(1)
EOF

if [ $? -ne 0 ]; then
    exit 1
fi

# Test 3: Check config files
echo ""
echo "[3/5] Checking configuration files..."
CONFIG_COUNT=$(find configs -name "*.yaml" | wc -l)
if [ $CONFIG_COUNT -eq 8 ]; then
    echo "✓ Found all 8 config files"
else
    echo "✗ Expected 8 config files, found $CONFIG_COUNT"
    exit 1
fi

# Test 4: Quick model instantiation test
echo ""
echo "[4/5] Testing model instantiation..."
python3 << 'EOF'
import torch
import sys
sys.path.append('.')

from models.conformer import Conformer
from models.branchformer import Branchformer

try:
    # Test Conformer with different attentions
    for attn in ['mha', 'mla', 'gqa', 'linear']:
        kwargs = {'latent_dim': 128} if attn == 'mla' else {}
        kwargs = {'num_kv_heads': 2} if attn == 'gqa' else kwargs
        
        model = Conformer(
            d_model=128,
            num_heads=4,
            num_layers=2,
            attention_type=attn,
            attention_kwargs=kwargs if kwargs else None
        )
        
        x = torch.randn(1, 50, 80)
        logits, _ = model(x)
        assert logits.shape[1] == 50
    
    print("✓ All attention mechanisms work correctly")

except Exception as e:
    print(f"✗ Model instantiation failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
EOF

if [ $? -ne 0 ]; then
    exit 1
fi

# Test 5: Check directory structure
echo ""
echo "[5/5] Checking directory structure..."
REQUIRED_DIRS=("models" "scripts" "configs" "data" "results" "logs")
ALL_EXIST=true

for dir in "${REQUIRED_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        echo "  ✓ $dir/"
    else
        echo "  ✗ $dir/ missing"
        ALL_EXIST=false
    fi
done

if [ "$ALL_EXIST" = false ]; then
    echo ""
    echo "Please create missing directories or run: bash setup_environment.sh"
    exit 1
fi

echo ""
echo "=========================================="
echo "✓ All tests passed!"
echo "=========================================="
echo ""
echo "Your setup is ready. Next steps:"
echo ""
echo "1. Download data:"
echo "   bash scripts/prepare_data.sh"
echo ""
echo "2. Run a single model (quick test):"
echo "   bash scripts/train_model.sh conformer mha"
echo ""
echo "3. Or run all baselines:"
echo "   bash run_all_baselines.sh"
echo ""
