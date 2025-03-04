FROM rust:1.73 as builder

WORKDIR /app

# Copy the Cargo.toml and Cargo.lock files
COPY Cargo.toml Cargo.lock ./
COPY rust-toolchain.toml ./

# Copy the source code
COPY sidecar/ ./sidecar/
COPY llm_client/ ./llm_client/
COPY llm_prompts/ ./llm_prompts/
COPY logging/ ./logging/

# Build the application
RUN cargo build --release --bin webserver

# Create a smaller image for running the application
FROM debian:bullseye-slim

WORKDIR /app

# Install dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the binary from the builder stage
COPY --from=builder /app/target/release/webserver /app/webserver

# Create a directory for data
RUN mkdir -p /app/data

# Set environment variables
ENV RUST_LOG=info
ENV SIDECAR_PORT=3000
ENV SIDECAR_HOST=0.0.0.0
ENV SIDECAR_SCRATCH_PAD_DIR=/app/data

# Expose the port
EXPOSE 3000

# Run the application
CMD ["/app/webserver"]