#!/bin/bash

# SuperClaude MCP Server Local Installation Script
# Installs MCP servers locally without requiring sudo

echo "═══════════════════════════════════════════════════"
echo "🚀 SuperClaude MCP Server Setup"
echo "═══════════════════════════════════════════════════"

# Setup directories
PROJECT_DIR="/home/admin/Desktop/NexaraVision"
MCP_DIR="$PROJECT_DIR/.mcp"
NODE_MODULES="$PROJECT_DIR/node_modules"

mkdir -p "$MCP_DIR"
cd "$PROJECT_DIR"

# Initialize npm if needed
if [ ! -f "package.json" ]; then
    echo "📦 Initializing npm project..."
    npm init -y >/dev/null 2>&1
fi

echo ""
echo "📦 Installing MCP Servers Locally..."
echo "════════════════════════════════════════════════════"

# Install available MCP packages locally
npm install --save-dev \
    @modelcontextprotocol/server-memory \
    @modelcontextprotocol/server-filesystem \
    mcp-server-fetch \
    2>/dev/null

# Check installations
echo ""
echo "✅ Installed MCP Servers:"
[ -d "node_modules/@modelcontextprotocol/server-memory" ] && echo "  • Memory Server ✓"
[ -d "node_modules/@modelcontextprotocol/server-filesystem" ] && echo "  • Filesystem Server ✓"
[ -d "node_modules/mcp-server-fetch" ] && echo "  • Fetch Server ✓"

# Create activation script
cat > "$MCP_DIR/activate-local.sh" << 'EOF'
#!/bin/bash
# SuperClaude MCP Local Activation

PROJECT_DIR="/home/admin/Desktop/NexaraVision"
export MCP_PROJECT_DIR="$PROJECT_DIR"
export PATH="$PROJECT_DIR/node_modules/.bin:$PATH"

echo "✅ MCP servers configured for local use"
echo "📍 Project: $PROJECT_DIR"
EOF

chmod +x "$MCP_DIR/activate-local.sh"

# Create SC command helpers
cat > "$MCP_DIR/sc-commands.sh" << 'EOF'
#!/bin/bash
# SuperClaude Command Helpers

# Sequential thinking activation
sc-seq() {
    echo "🧠 Activating Sequential Thinking..."
    npx @modelcontextprotocol/server-sequential-thinking
}

# Memory server activation
sc-mem() {
    echo "💾 Activating Memory Server..."
    npx @modelcontextprotocol/server-memory
}

# Filesystem server activation
sc-fs() {
    echo "📁 Activating Filesystem Server..."
    npx @modelcontextprotocol/server-filesystem --root /home/admin
}

# Help command
sc-help() {
    echo "SuperClaude MCP Commands:"
    echo "  sc-seq   - Sequential thinking for deep analysis"
    echo "  sc-mem   - Memory for session persistence"
    echo "  sc-fs    - Enhanced filesystem operations"
    echo "  sc-help  - Show this help message"
}

echo "SuperClaude helpers loaded. Type 'sc-help' for commands."
EOF

chmod +x "$MCP_DIR/sc-commands.sh"

echo ""
echo "═══════════════════════════════════════════════════"
echo "✨ Setup Complete!"
echo "═══════════════════════════════════════════════════"
echo ""
echo "📍 MCP Configuration: $MCP_DIR"
echo ""
echo "🔧 Currently Available MCP Servers:"
echo "  • Sequential Thinking (built-in)"
echo "  • Memory (built-in)"
echo "  • Postgres (built-in)"
echo "  • Context7 (built-in)"
echo "  • Playwright (built-in)"
echo ""
echo "📝 To activate helpers:"
echo "  source $MCP_DIR/sc-commands.sh"
echo ""
echo "🚀 Usage with flags:"
echo "  --seq       Sequential thinking"
echo "  --c7        Context7 documentation"
echo "  --play      Playwright browser"
echo "  --all-mcp   All servers"
echo ""
echo "═══════════════════════════════════════════════════"