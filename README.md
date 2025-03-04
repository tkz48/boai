<div align="center">
  <img src="./public/logo-dark-styled.png" alt="Sidebolt Logo" width="200"/>
  <h1>Sidebolt</h1>
  <p>An AI Agent with integrated Sidecar functionality</p>
</div>

## What is Sidebolt?

Sidebolt is a powerful AI agent that combines the user-friendly interface of Bolt with the advanced code analysis capabilities of Sidecar. This integration provides a seamless experience for developers who want to leverage AI for code generation, analysis, and improvement.

### Key Features

- **Chat Interface**: Interact with AI models through a clean, intuitive chat interface
- **Code Analysis**: Leverage Sidecar's advanced code analysis capabilities
- **Repository Mapping**: Understand code relationships and dependencies
- **Inline Completions**: Get intelligent code suggestions as you type
- **Agentic Capabilities**: Use AI agents to solve complex coding tasks
- **Multiple AI Providers**: Support for various AI providers including OpenAI, Anthropic, and more

## Architecture

Sidebolt consists of two main components:

1. **Bolt Frontend**: A React-based web application that provides the user interface
2. **Sidecar Backend**: A Rust-based service that provides advanced code analysis and AI capabilities

The integration allows the frontend to communicate with the Sidecar service through a dedicated API endpoint, enabling seamless access to Sidecar's powerful features.

## Getting Started

### Prerequisites

- Node.js 18.18.0 or higher
- Rust 1.73 or higher
- pnpm 9.4.0 or higher

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/sidebolt.git
   cd sidebolt
   ```

2. Install dependencies:
   ```bash
   pnpm install
   ```

3. Build Sidecar:
   ```bash
   pnpm run build:sidecar
   ```

4. Start the development server:
   ```bash
   pnpm run dev:all
   ```

5. Open your browser and navigate to `http://localhost:5173`

## Development

### Project Structure

- `/app`: The main React application
- `/sidecar`: The Sidecar Rust service
- `/public`: Static assets
- `/functions`: Cloudflare Functions

### Available Scripts

- `pnpm run dev:all`: Start both the frontend and Sidecar service
- `pnpm run build`: Build the frontend
- `pnpm run build:sidecar`: Build the Sidecar service
- `pnpm run start:all`: Start the production build with Sidecar

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.