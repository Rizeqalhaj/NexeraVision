#!/bin/bash
# Transfer all necessary files to Vast.ai instance
# Run this script AFTER adding your SSH key to Vast.ai

set -e

SSH_HOST="root@70.77.113.32"
SSH_PORT="40964"

echo "========================================"
echo "NexaraVision - Transfer Files to Vast.ai"
echo "========================================"
echo ""

# Test connection
echo "🔌 Testing SSH connection..."
if ! ssh -p $SSH_PORT -o ConnectTimeout=10 $SSH_HOST "echo '✅ Connection successful'"; then
    echo "❌ ERROR: Cannot connect to Vast.ai instance"
    echo ""
    echo "Please ensure:"
    echo "1. You've added your SSH public key to Vast.ai"
    echo "2. The instance is running"
    echo "3. The connection details are correct"
    echo ""
    echo "Your public key:"
    cat ~/.ssh/id_rsa.pub
    exit 1
fi

echo ""
echo "📂 Creating workspace directory on Vast.ai..."
ssh -p $SSH_PORT $SSH_HOST "mkdir -p /workspace"

echo ""
echo "📤 Transferring setup script..."
scp -P $SSH_PORT VASTAI_SETUP_COMPLETE.sh $SSH_HOST:/workspace/

echo ""
echo "📤 Transferring dataset download script..."
scp -P $SSH_PORT VASTAI_DOWNLOAD_DATASETS.py $SSH_HOST:/workspace/

echo ""
echo "📤 Transferring PROGRESS.md (documentation)..."
scp -P $SSH_PORT PROGRESS.md $SSH_HOST:/workspace/

echo ""
echo "✅ All files transferred successfully!"
echo ""
echo "========================================"
echo "Next Steps:"
echo "========================================"
echo ""
echo "1. SSH into Vast.ai:"
echo "   ssh -p $SSH_PORT $SSH_HOST"
echo ""
echo "2. Run setup script:"
echo "   cd /workspace"
echo "   chmod +x VASTAI_SETUP_COMPLETE.sh"
echo "   ./VASTAI_SETUP_COMPLETE.sh"
echo ""
echo "3. Start downloading datasets:"
echo "   python3 VASTAI_DOWNLOAD_DATASETS.py"
echo ""
echo "4. Monitor progress with:"
echo "   watch -n 5 'du -sh /workspace/datasets/*'"
echo ""
echo "========================================"
