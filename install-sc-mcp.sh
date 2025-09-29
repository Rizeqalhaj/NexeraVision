#!/bin/bash

# SuperClaude MCP Server Installation Script
# This script installs all recommended MCP servers for the SC framework

echo "═══════════════════════════════════════════════════"
echo "SuperClaude MCP Server Installation"
echo "═══════════════════════════════════════════════════"

# Create MCP directory if it doesn't exist
MCP_DIR="$HOME/.mcp"
mkdir -p "$MCP_DIR"

# Function to check if npm package exists
check_npm_package() {
    npm list -g "$1" &>/dev/null
    return $?
}

# Function to install MCP server via npm
install_mcp() {
    local package="$1"
    local name="$2"

    echo ""
    echo "📦 Installing $name..."

    if check_npm_package "$package"; then
        echo "✅ $name already installed"
    else
        npm install -g "$package"
        if [ $? -eq 0 ]; then
            echo "✅ $name installed successfully"
        else
            echo "❌ Failed to install $name"
            return 1
        fi
    fi
}

# Install Node.js MCP servers
echo ""
echo "🔧 Installing MCP Servers..."
echo "════════════════════════════════════════"

# Core MCP servers for SuperClaude
install_mcp "@modelcontextprotocol/server-sequential-thinking" "Sequential Thinking (for deep analysis)"
install_mcp "@modelcontextprotocol/server-memory" "Memory (for session persistence)"
install_mcp "@modelcontextprotocol/server-playwright" "Playwright (for browser automation)"

# Additional recommended servers
install_mcp "@modelcontextprotocol/server-filesystem" "Filesystem (enhanced file operations)"
install_mcp "@modelcontextprotocol/server-git" "Git (version control operations)"
install_mcp "@modelcontextprotocol/server-puppeteer" "Puppeteer (alternative browser control)"

# Python-based MCP servers (if Python is available)
if command -v python3 &>/dev/null; then
    echo ""
    echo "🐍 Setting up Python-based MCP servers..."

    # Create virtual environment for Python MCPs
    python3 -m venv "$MCP_DIR/venv" 2>/dev/null
    source "$MCP_DIR/venv/bin/activate"

    # Install Python MCP packages
    pip install --upgrade pip &>/dev/null
    pip install mcp-server-fetch 2>/dev/null && echo "✅ Fetch server installed"
    pip install mcp-server-browser 2>/dev/null && echo "✅ Browser server installed"

    deactivate
fi

# Create MCP configuration file template
echo ""
echo "📄 Creating MCP configuration template..."

cat > "$MCP_DIR/sc-config.json" << 'EOF'
{
  "mcpServers": {
    "sequential": {
      "command": "npx",
      "args": ["@modelcontextprotocol/server-sequential-thinking"],
      "description": "Deep multi-step analysis and reasoning"
    },
    "memory": {
      "command": "npx",
      "args": ["@modelcontextprotocol/server-memory"],
      "description": "Session persistence and memory management"
    },
    "playwright": {
      "command": "npx",
      "args": ["@modelcontextprotocol/server-playwright"],
      "description": "Browser automation and testing"
    },
    "filesystem": {
      "command": "npx",
      "args": ["@modelcontextprotocol/server-filesystem", "--root", "/home/admin"],
      "description": "Enhanced file system operations"
    },
    "git": {
      "command": "npx",
      "args": ["@modelcontextprotocol/server-git"],
      "description": "Git version control operations"
    }
  },
  "defaults": {
    "autoStart": true,
    "restartOnFailure": true,
    "maxRestarts": 3
  }
}
EOF

echo "✅ Configuration template created at $MCP_DIR/sc-config.json"

# Create activation script
cat > "$MCP_DIR/activate-sc.sh" << 'EOF'
#!/bin/bash
# SuperClaude MCP Activation Script

echo "🚀 Activating SuperClaude MCP servers..."

# Export MCP configuration
export MCP_CONFIG="$HOME/.mcp/sc-config.json"

# Start essential services
npx @modelcontextprotocol/server-sequential-thinking &
npx @modelcontextprotocol/server-memory &

echo "✅ MCP servers activated"
echo "📝 Configuration: $MCP_CONFIG"
EOF

chmod +x "$MCP_DIR/activate-sc.sh"

echo ""
echo "═══════════════════════════════════════════════════"
echo "✨ Installation Complete!"
echo "═══════════════════════════════════════════════════"
echo ""
echo "Available MCP Servers:"
echo "  • Sequential Thinking (--seq): Deep analysis"
echo "  • Memory (--mem): Session persistence"
echo "  • Playwright (--play): Browser automation"
echo "  • Filesystem: Enhanced file operations"
echo "  • Git: Version control"
echo ""
echo "Configuration: $MCP_DIR/sc-config.json"
echo "Activation: source $MCP_DIR/activate-sc.sh"
echo ""
echo "Use flags to activate specific servers:"
echo "  --seq    Sequential thinking"
echo "  --play   Playwright browser"
echo "  --all-mcp All servers"
echo "═══════════════════════════════════════════════════"