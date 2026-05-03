#! /bin/bash

set -eu

cat > .env << EOF
UID=$(id -u)
GID=$(id -g)
EOF

echo ".env file created successfully"
