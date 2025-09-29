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
